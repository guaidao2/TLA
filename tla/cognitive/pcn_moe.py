"""MoE-PCN：任务分离容量（专家混合）——共享底 + K 个专家顶层。

背景（"线性捷径悖论"，见文档 §14）：共享 W_lin 线性捷径在单任务上易学，但作为任务无关的
单一线性映射同时杀死"会琢磨"（P-COG-3 空转）与"防遗忘"（P-LEARN-1 权重级冲突）。
本文件是修复方向的第一次实现：**无捷径 + 任务分离专家**——

- 共享层 0..L-1（输入 + 隐藏）；
- K 个专家，各自拥有顶层表示 μ_top,e 与自己的读出 W_out,e（无 W_lin 捷径）；
- **自由能路由**：每个专家单独解释共享层（重建误差 E_e 最小者胜），责任 r = softmax(−E/T)；
- 输出 = Σ_e r_e · readout_e（软路由）；
- 学习：局部误差驱动，专家权重按责任 r_e 加权更新（ΔW ∝ r_e·e·a）——无 BP/autograd；
- 任务分离：A/B 输入路由到不同专家 → 无权重级冲突 → P-LEARN-1 修复方向；
  每专家内 settle 承重 → P-COG-3 正证据场地。
"""
import torch


class MoEPCNStack:
    def __init__(self, dims, out_dim, n_experts=2, lr_inf=0.1, prior=0.0,
                 mu_max=5.0, route_mode="proto", seed=None):
        """dims = [in, hidden, top]（L=2：共享一层隐藏 + 专家顶层）。

        route_mode="proto"：原型路由（专家持输入原型，就近竞争，原型 EMA 局部更新）；
        route_mode="free_energy"：自由能路由（重建误差最小者胜）。
        """
        assert len(dims) == 3, "MoE v1: dims=[in, hidden, top]"
        self.dims = dims
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.L = 2                      # 顶层索引（专家）
        self.lr_inf = lr_inf
        self.prior = prior
        self.mu_max = mu_max
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        g1 = 1.0 / dims[1] ** 0.5
        # 共享：W_1: dims[0] x dims[1]（隐藏层预测输入）
        self.W_1 = torch.randn(dims[0], dims[1], generator=gen) * g1
        self.b_1 = torch.zeros(dims[0])
        # 专家
        self.expert = []
        for e in range(n_experts):
            ge = torch.Generator().manual_seed((seed if seed is not None else 0) + 100 + e)
            gu = 1.0 / dims[2] ** 0.5
            self.expert.append(dict(
                W_up=torch.randn(dims[1], dims[2], generator=ge) * gu,   # 专家顶层预测隐藏层
                b_up=torch.zeros(dims[1]),
                W_out=torch.randn(out_dim, dims[2], generator=ge) / dims[2] ** 0.5,
                b_out=torch.zeros(out_dim),
                mu=torch.zeros(dims[2]),
            ))
        self.mu_1 = torch.zeros(dims[1])
        self.last_max_err = 0.0
        self.last_routing = torch.ones(n_experts) / n_experts
        self.r_learn = torch.ones(n_experts) / n_experts
        self.route_mode = route_mode
        # 原型路由：每专家一个输入原型（init 取随机方向，训练中 EMA 局部更新）
        self.proto = [torch.randn(dims[0], generator=gen) * 0.3 for _ in range(n_experts)]

    def reset(self):
        self.mu_1 = torch.zeros_like(self.mu_1)
        for e in self.expert:
            e["mu"] = torch.zeros_like(e["mu"])

    # ---- 误差 ----
    def errors(self, x=None, target=None):
        """返回共享层误差、每专家误差、每专家读出与路由（x=None 时只算读出与路由）。"""
        mu_1 = self.mu_1
        e_0 = (x - torch.tanh(self.W_1 @ mu_1 + self.b_1)) if x is not None else None
        e_top, e_out, reads, fe = [], [], [], []
        for ex in self.expert:
            p_1 = torch.tanh(ex["W_up"] @ ex["mu"] + ex["b_up"])  # 专家预测隐藏层
            et = ex["mu"] - self.prior                            # 专家顶层 prior 误差
            read = ex["W_out"] @ ex["mu"] + ex["b_out"]
            eo = (target - read) if target is not None else None
            # 自由能（路由信号）：专家单独解释隐藏层的重建误差 + 顶层 prior
            fe_e = float((torch.mean((mu_1 - p_1) ** 2) + torch.mean(et ** 2)).item())
            e_top.append(et)
            e_out.append(eo)
            reads.append(read)
            fe.append(fe_e)
        fe = torch.tensor(fe)
        if self.route_mode == "proto" and x is not None:
            # 原型路由：距离最小者胜（竞争更直接，防自由能信号不具判别性）
            d = torch.stack([torch.norm(x - p) for p in self.proto])
            r = torch.softmax(-d, dim=0)
            # 硬路由学习也必须用原型距离 argmin（用 fe 是 bug：fe 不具判别性，
            # 专家学习永远收不到分离信号——MoE 终审发现）
            hard_idx = torch.argmin(d)
        else:
            r = torch.softmax(-fe / 1.0, dim=0)                   # 软路由（自由能低者胜）
            hard_idx = torch.argmin(fe)
        # 硬路由主导学习（防 MoE 对称性：两专家学成一样，路由卡 0.5）：
        # 赢者 one-hot 为主 + 20% 软路由探索（防死专家）
        onehot = torch.zeros_like(r)
        onehot[hard_idx] = 1.0
        self.r_learn = 0.8 * onehot + 0.2 * r
        # 共享层误差 e_1 = μ_1 − 混合顶层预测
        mix_p_1 = sum(torch.tanh(ex["W_up"] @ ex["mu"] + ex["b_up"]) * r[e]
                      for e, ex in enumerate(self.expert))
        e_1 = mu_1 - mix_p_1
        self.last_routing = r
        # 原型 EMA 更新（路由信号：哪个专家服务当前输入，其原型就近拉）
        if self.route_mode == "proto" and x is not None:
            for e in range(self.n_experts):
                self.proto[e] = self.proto[e] + 0.05 * self.r_learn[e] * (x - self.proto[e])
        return e_0, e_1, e_top, e_out, reads, r

    # ---- 推理（settle 一步）----
    def settle_step(self, x, target=None, lr_inf=None):
        lr = lr_inf if lr_inf is not None else self.lr_inf
        e_0, e_1, e_top, e_out, reads, r = self.errors(x, target)
        # 更新共享隐藏层 μ_1：grad = e_1 − W_1ᵀ(g'⊙e_0)
        g1 = 1.0 - torch.tanh(self.W_1 @ self.mu_1 + self.b_1) ** 2
        grad_1 = e_1 - self.W_1.T @ (g1 * e_0)
        self.mu_1 = (self.mu_1 - lr * grad_1).clamp(-self.mu_max, self.mu_max)
        # 更新每个专家顶层 μ_top,e
        for e, ex in enumerate(self.expert):
            p_1 = torch.tanh(ex["W_up"] @ ex["mu"] + ex["b_up"])
            g_up = 1.0 - p_1 ** 2
            grad_top = e_top[e] - ex["W_up"].T @ (g_up * (r[e] * e_1))
            if e_out[e] is not None:
                grad_top = grad_top - ex["W_out"].T @ e_out[e]
            ex["mu"] = (ex["mu"] - lr * grad_top).clamp(-self.mu_max, self.mu_max)
        self.last_max_err = float(max(torch.max(torch.abs(e_0)).item(),
                                      torch.max(torch.abs(e_1)).item(),
                                      max(torch.max(torch.abs(et)).item() for et in e_top)))
        return self.last_max_err

    # ---- 推理（完整 settling，时序暖启动）----
    def settle(self, x, target=None, steps=4):
        for _ in range(steps):
            last = self.settle_step(x, target)
        return last

    def readout(self):
        _, _, _, _, reads, r = self.errors(None)
        return sum(reads[e] * r[e] for e in range(self.n_experts))

    # ---- 学习（软路由局部误差驱动，无 BP）----
    def learn_step(self, x, target, lr=0.01, settle_steps=4, wd=1e-4):
        self.settle(x, target, steps=settle_steps)
        e_0, e_1, e_top, e_out, reads, r = self.errors(x, target)
        # 共享层
        g1 = 1.0 - torch.tanh(self.W_1 @ self.mu_1 + self.b_1) ** 2
        self.W_1 = (1.0 - lr * wd) * self.W_1 + lr * torch.outer(g1 * e_0, self.mu_1)
        self.b_1 = self.b_1 + lr * (g1 * e_0)
        # 专家（按硬路由主导的责任加权，防对称性）
        for e, ex in enumerate(self.expert):
            p_1 = torch.tanh(ex["W_up"] @ ex["mu"] + ex["b_up"])
            g_up = 1.0 - p_1 ** 2
            re = self.r_learn[e]
            ex["W_up"] = ((1.0 - lr * wd) * ex["W_up"]
                          + lr * re * torch.outer(g_up * e_1, ex["mu"]))
            ex["b_up"] = ex["b_up"] + lr * re * (g_up * e_1)
            if e_out[e] is not None:
                ex["W_out"] = ((1.0 - lr * wd) * ex["W_out"]
                               + lr * re * torch.outer(e_out[e], ex["mu"]))
                ex["b_out"] = ex["b_out"] + lr * re * e_out[e]
        return float(torch.mean(
            sum((e_out[e] ** 2).mean() * r[e] for e in range(self.n_experts)
                if e_out[e] is not None)).item()) if any(eo is not None for eo in e_out) else None
