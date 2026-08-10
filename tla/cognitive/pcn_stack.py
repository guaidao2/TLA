"""预测编码表征层叠（③④）：每层预测下层表征，误差逐层传播。

标准 PCN（Rao-Ballard / Whittington-Bogacz 式）：
- 生成模型：p_{l-1} = tanh(W_l μ_l + b_l)（层 l 预测层 l-1，非线性在预测侧），顶层 prior=0；
- μ 本身自由演化（不逐层硬压），推理 = 对 free energy 的梯度下降；
- readout：p_out = W_out·gate(μ_L) + W_lin·x + b_out（线性捷径防 tanh 饱和封顶）；
- 学习：ΔW_l = η·(g'(a_l)⊙e_{l-1})·μ_lᵀ、Δb_l = η·(g'(a_l)⊙e_{l-1}) —— 全部局部误差驱动，无 BP/autograd。

容量门（⑨）：readout 经 gate（mask·gain）输出——由 meta/CapacityManager 提供。
"""
import torch


class PCNStack:
    def __init__(self, dims, out_dim, lr_inf=0.1, prior=0.0, mu_max=5.0,
                 use_lin_shortcut=True, seed=None):
        assert len(dims) >= 2
        self.dims = dims
        self.out_dim = out_dim
        self.L = len(dims) - 1          # 隐藏层数（含顶层）
        self.lr_inf = lr_inf
        self.prior = prior
        self.mu_max = mu_max            # μ 安全界（物理有界性，①）
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        g = [1.0 / dims[l] ** 0.5 for l in range(1, len(dims))]
        # W_l: dims[l-1] x dims[l] —— 层 l 预测层 l-1
        self.Ws = [torch.randn(dims[l - 1], dims[l], generator=gen) * g[l - 1]
                   for l in range(1, self.L + 1)]
        self.bs = [torch.zeros(dims[l - 1]) for l in range(1, self.L + 1)]
        self.W_out = torch.randn(out_dim, dims[-1], generator=gen) / dims[-1] ** 0.5
        self.use_lin = use_lin_shortcut
        self.W_lin = (torch.randn(out_dim, dims[0], generator=gen) * 0.1
                      if use_lin_shortcut else torch.zeros(out_dim, dims[0]))
        self.b_out = torch.zeros(out_dim)
        self.mus = [torch.zeros(d) for d in dims]
        self._gate = None               # 容量门（mask·gain 向量），由 CapacityManager 注入
        self.last_max_err = 0.0

    # ---- 容量门（⑨）----
    def set_gate(self, gate_fn):
        self._gate = gate_fn

    def _gated_top(self):
        mu = self.mus[self.L]
        if self._gate is not None:
            mu = mu * self._gate()
        return mu

    # ---- 误差与预测（非线性在预测侧）----
    def errors(self, target=None):
        mu = self.mus
        a = [None] * (self.L + 1)               # a_l（层 l 预测 l-1 的预激活）
        p = [None] * self.L                     # p_{l-1} = tanh(a_l)
        for l in range(1, self.L + 1):
            a[l] = self.Ws[l - 1] @ mu[l] + self.bs[l - 1]
            p[l - 1] = torch.tanh(a[l])
        e = [mu[0] - p[0]]
        for l in range(1, self.L):
            e.append(mu[l] - p[l])
        e.append(mu[self.L] - self.prior)       # e_L（顶层 prior）
        p_out = self.W_out @ self._gated_top() + self.W_lin @ mu[0] + self.b_out
        e_out = (target - p_out) if target is not None else None
        return e, e_out, p_out, a, p
    def max_err(self, e):
        return max(torch.max(torch.abs(ee)).item() for ee in e)

    # ---- 推理（settle 一步：对 free energy 做梯度下降，μ 自由演化）----
    def settle_step(self, x, target=None, lr_inf=None):
        lr = lr_inf if lr_inf is not None else self.lr_inf
        mu = self.mus
        mu[0] = x
        e, e_out, _, a, p = self.errors(target)
        for l in range(1, self.L + 1):
            gprime = 1.0 - p[l - 1] ** 2                      # tanh'(a_l)
            grad = e[l] - self.Ws[l - 1].T @ (gprime * e[l - 1])
            if l == self.L and e_out is not None:
                gate = self._gate() if self._gate is not None else 1.0
                grad = grad - gate * (self.W_out.T @ e_out)   # target 注入顶层（带容量门，与学习梯度一致）
            mu[l] = (mu[l] - lr * grad).clamp(-self.mu_max, self.mu_max)   # 安全界（①）
        self.last_max_err = self.max_err(e)
        return self.last_max_err

    # ---- 推理（完整 settling，时序暖启动：μ 跨时刻继承，轨迹边界才重置）----
    def settle(self, x, target=None, steps=4):
        self.mus[0] = x
        last = 0.0
        for _ in range(steps):
            last = self.settle_step(x, target)
        return last

    def reset_mus(self):
        for l in range(1, self.L + 1):
            self.mus[l] = torch.zeros_like(self.mus[l])

    def readout(self):
        return self.W_out @ self._gated_top() + self.W_lin @ self.mus[0] + self.b_out

    # ---- 学习（局部误差驱动，无 BP；权重衰减防线性路径漂移）----
    def learn_weights(self, target, lr=0.01, wd=1e-4):
        e, e_out, _, a, p = self.errors(target)
        mu = self.mus
        for l in range(1, self.L + 1):
            gprime = 1.0 - p[l - 1] ** 2
            delta = gprime * e[l - 1]
            self.Ws[l - 1] = (1.0 - lr * wd) * self.Ws[l - 1] + lr * torch.outer(delta, mu[l])
            self.bs[l - 1] = self.bs[l - 1] + lr * delta
        if e_out is not None:
            self.W_out = (1.0 - lr * wd) * self.W_out + lr * torch.outer(e_out, self._gated_top())
            if self.use_lin:    # 无捷径模式：W_lin 恒 0（输出完全依赖精化后的 μ，循环承重）
                self.W_lin = (1.0 - lr * wd) * self.W_lin + lr * torch.outer(e_out, self.mus[0])
            self.b_out = self.b_out + lr * e_out
            return float(torch.mean(e_out ** 2).item())
        return None

    def learn_step(self, x, target, lr=0.01, settle_steps=4):
        """两阶段：free phase settle（输入+target 双钳制）→ 局部权重更新。"""
        self.settle(x, target, steps=settle_steps)
        return self.learn_weights(target, lr)
