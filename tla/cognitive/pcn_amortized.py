"""原则一：摊销首猜 + 残差修正（Amortized Residual PCN）。

哲学诊断（文档 §14"捷径悖论"之后）：两个优化器抢同一目标导致分工失序——
- 有捷径：慢优化器（权重）抢了快优化器（settle）的活 → 琢磨空转（P-COG-3 负结果）；
- 无捷径：快优化器稀释了慢优化器的信号 → 弱学习（~0.11 vs 0.004）。

本实现把分工写死：
- **摊销首猜** W_base@x：对"自己的首猜误差" e_base = target − pred_base 负责——
  学得快（线性通路，局部更新不被 settle 稀释），提供学习强度；
- **残差通路**（无捷径 PCN，经 settle）：读出层对"总误差" e_total = target − pred_total 负责，
  隐藏层按重建误差 e_0 更新（PCN 本体）——学"首猜不够用的部分"（上下文相关的 Δ），
  输出为加性残差，settle 对输出有直接贡献，琢磨在构造上不可旁路；
- 最终输出 pred = pred_base + Δ（Δ 经推理环迭代精化 → 自适应深度真实影响输出）。

分工防摆烂（对读出层严格成立）：W_base 只按 e_base 更新（残差救不了它，必须自己学）；
残差读出层只按 e_total 更新（首猜错时 e_total 大 → 残差有活干，不会空转）。
"""
import torch


class AmortizedResidualPCN:
    def __init__(self, dims, out_dim, lr_inf=0.1, prior=0.0, mu_max=5.0, seed=None):
        """dims = [in, hidden]（单隐藏层残差通路）。"""
        assert len(dims) == 2
        self.dims = dims
        self.out_dim = out_dim
        self.lr_inf = lr_inf
        self.prior = prior
        self.mu_max = mu_max
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        g = 1.0 / dims[1] ** 0.5
        # 摊销首猜（学得快，学习强度来源）
        self.W_base = torch.randn(out_dim, dims[0], generator=gen) / dims[0] ** 0.5
        self.b_base = torch.zeros(out_dim)
        # 残差通路：隐藏层预测输入（重建）+ 顶层读出残差
        self.W_1 = torch.randn(dims[0], dims[1], generator=gen) * g
        self.b_1 = torch.zeros(dims[0])
        self.W_out = torch.randn(out_dim, dims[1], generator=gen) / dims[1] ** 0.5
        self.b_out = torch.zeros(out_dim)
        self.mu_1 = torch.zeros(dims[1])
        self.last_max_err = 0.0
        # ---- 表示坍缩防护（文本世界防预注册；默认关，机制就绪）----
        self.rep_cov_reg = 0.0          # >0 激活；0=关
        self.sigma_target = 1e-3        # 表示方差目标下限
        self._rep_mean = None           # 表示 EMA 均值（跨样本）
        self._rep_var = None            # 表示 EMA 方差

    def update_rep_stats(self):
        """跨样本 EMA 表示统计（检测坍缩：所有输入映射到同一表示 → 方差→0）。"""
        mu = self.mu_1.detach()
        if self._rep_mean is None:
            self._rep_mean = mu.clone()
            self._rep_var = torch.zeros_like(mu)
        else:
            self._rep_mean = 0.99 * self._rep_mean + 0.01 * mu
            self._rep_var = 0.99 * self._rep_var + 0.01 * (mu - self._rep_mean) ** 2

    def anti_collapse(self, lr=0.01):
        """坍缩防护（默认关）：表示方差低于目标时，把 μ 推离 EMA 均值 + 小扰动。

        面向文本世界/目标表示学习的防预注册机制（JEPA 坍缩问题的局部版本）——
        当前无坍缩任务可行为验证，仅提供机制 hook 与统计；激活阈值/强度待真实场景标定。
        """
        if self.rep_cov_reg <= 0:
            return
        if self._rep_var is None:
            return
        sigma2 = float(self._rep_var.mean())
        if sigma2 < self.sigma_target:
            push = (self.mu_1 - self._rep_mean) + 0.1 * torch.randn_like(self.mu_1)
            self.mu_1 = (self.mu_1 + self.rep_cov_reg * push).clamp(-self.mu_max, self.mu_max)

    def reset(self):
        self.mu_1 = torch.zeros_like(self.mu_1)

    # ---- 误差 ----
    def errors(self, x, target=None):
        pred_base = self.W_base @ x + self.b_base
        p_0 = torch.tanh(self.W_1 @ self.mu_1 + self.b_1)      # 隐藏层重建输入
        e_0 = x - p_0                                          # 输入层重建误差
        e_1 = self.mu_1 - self.prior                           # 顶层 prior 误差
        res = self.W_out @ self.mu_1 + self.b_out              # 残差读出
        pred_total = pred_base + res
        e_out = (target - pred_total) if target is not None else None
        return e_0, e_1, e_out, pred_base, res, pred_total

    def max_err(self, e_0, e_1):
        return max(torch.max(torch.abs(e_0)).item(), torch.max(torch.abs(e_1)).item())

    # ---- 推理（settle 一步：残差通路精化 μ，直接改输出残差）----
    def settle_step(self, x, target=None, lr_inf=None):
        lr = lr_inf if lr_inf is not None else self.lr_inf
        e_0, e_1, e_out, _, _, _ = self.errors(x, target)
        g1 = 1.0 - torch.tanh(self.W_1 @ self.mu_1 + self.b_1) ** 2
        grad = e_1 - self.W_1.T @ (g1 * e_0)
        if e_out is not None:
            grad = grad - self.W_out.T @ e_out                  # target 注入（残差通路学总误差）
        self.mu_1 = (self.mu_1 - lr * grad).clamp(-self.mu_max, self.mu_max)
        self.update_rep_stats()
        self.anti_collapse(lr)
        self.last_max_err = self.max_err(e_0, e_1)
        return self.last_max_err

    def settle(self, x, target=None, steps=4):
        last = 0.0
        for _ in range(steps):
            last = self.settle_step(x, target)
        return last

    def readout(self, x):
        _, _, _, pred_base, res, _ = self.errors(x)
        return pred_base + res

    # ---- 突触巩固（EWC 式）：A 训练累计 importance，B 训练按重要性拉回 A 状态 ----
    def start_consolidation(self):
        self._imp = {k: torch.zeros_like(v) for k, v in self._params().items()}

    def finalize_consolidation(self):
        """快照 A 训练后的权重为参考（须在 B 训练前调用）；importance 归一化到 [0,1]（防无界累加）。"""
        self._ref = {k: v.clone() for k, v in self._params().items()}
        for k in self._imp:
            mx = self._imp[k].max()
            if mx > 0:
                self._imp[k] = self._imp[k] / mx

    def _params(self):
        return dict(W_base=self.W_base, b_base=self.b_base,
                    W_1=self.W_1, b_1=self.b_1, W_out=self.W_out, b_out=self.b_out)

    # ---- 学习（分工写死：首猜对 e_base，残差对 e_total；无 BP）----
    def learn_step(self, x, target, lr=0.01, settle_steps=4, wd=1e-4,
                   freeze_base=False, consolidate=False, protect=False, lam=1.0):
        self.settle(x, target, steps=settle_steps)
        e_0, e_1, e_out, pred_base, res, pred_total = self.errors(x, target)
        e_base = target - pred_base                              # 首猜自己的误差
        # 摊销首猜：只按自己的误差更新（残差救不了它 → 必须自己学，防摆烂）
        # freeze_base=True：冻结首猜（遗忘定位诊断用——只让残差通路学新任务）
        if not freeze_base:
            self.W_base = (1.0 - lr * wd) * self.W_base + lr * torch.outer(e_base, x)
            self.b_base = self.b_base + lr * e_base
        # 残差通路：按总误差更新（首猜错 → e_total 大 → 残差有活干，防空转）
        g1 = 1.0 - torch.tanh(self.W_1 @ self.mu_1 + self.b_1) ** 2
        self.W_1 = (1.0 - lr * wd) * self.W_1 + lr * torch.outer(g1 * e_0, self.mu_1)
        self.b_1 = self.b_1 + lr * (g1 * e_0)
        self.W_out = (1.0 - lr * wd) * self.W_out + lr * torch.outer(e_out, self.mu_1)
        self.b_out = self.b_out + lr * e_out
        # 突触巩固：A 训练时累计每个权重的更新量级（importance=Fisher 式，全局部）
        if consolidate and hasattr(self, "_imp"):
            self._imp["W_base"] += torch.outer(e_base, x).abs()
            self._imp["b_base"] += e_base.abs()
            self._imp["W_1"] += torch.outer(g1 * e_0, self.mu_1).abs()
            self._imp["b_1"] += (g1 * e_0).abs()
            self._imp["W_out"] += torch.outer(e_out, self.mu_1).abs()
            self._imp["b_out"] += e_out.abs()
        # 突触巩固：B 训练时按重要性把权重拉回 A 状态（保护重要突触，次要的随便改）
        if protect and hasattr(self, "_ref"):
            for k, v in self._params().items():
                pull = lr * lam * self._imp[k] * (self._ref[k] - v)
                self.__dict__[k] = v + pull
        return float(torch.mean(e_out ** 2).item())

    # ---- 批训练（mini-batch 局部梯度等价）----
    def learn_batch(self, xs, targets, lr=0.01, settle_steps=4, wd=1e-4):
        """批量训练：逐样本 settle（批内暖启动，同序列训练语义），更新项按批累计后
        除以批大小一次性应用——等价于对局部更新规则做 mini-batch 梯度（ΔW = (η/B)Σδ）。"""
        B = len(xs)
        acc = {k: torch.zeros_like(v) for k, v in self._params().items()}
        e_out_sum = 0.0
        for x, t in zip(xs, targets):
            self.settle(x, t, steps=settle_steps)
            e_0, e_1, e_out, pred_base, res, pred_total = self.errors(x, t)
            e_base = t - pred_base
            acc["W_base"] += torch.outer(e_base, x)
            acc["b_base"] += e_base
            g1 = 1.0 - torch.tanh(self.W_1 @ self.mu_1 + self.b_1) ** 2
            acc["W_1"] += torch.outer(g1 * e_0, self.mu_1)
            acc["b_1"] += g1 * e_0
            acc["W_out"] += torch.outer(e_out, self.mu_1)
            acc["b_out"] += e_out
            e_out_sum += float(torch.mean(e_out ** 2).item())
        for k, v in self._params().items():
            # 与 learn_step 对齐：仅权重矩阵衰减 wd，偏置不衰减（两路径一致性）
            if k.startswith("b_"):
                self.__dict__[k] = v + (lr / B) * acc[k]
            else:
                self.__dict__[k] = (1.0 - lr * wd) * v + (lr / B) * acc[k]
        return e_out_sum / B
