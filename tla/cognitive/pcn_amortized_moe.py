"""原则一 × MoE 专家分离 v2：每专家=完整摊销预测器（Amortized MoE-PCN）。

v1 教训（2026-08-11 实测）：v1 只把残差读出层按专家分离，W_base 与共享隐藏层仍是
全局的——B 训练覆盖 W_base → A 的首猜被毁 → 结构性防遗忘失败（无 EWC 保留率 2.8%）。

v2 设计（"消遗忘"在构造上成立）：
- 每个专家 = **完整 AmortizedResidualPCN**（自己的 W_base 首猜 + 自己的隐藏层 W_1 +
  自己的残差读出 W_out），只有 LTC 基板在堆栈外共享；
- **原型路由**：专家持输入原型，就近竞争（硬路由主导学习 0.8/0.2，防对称/防死专家）；
- 输入路由到赢家专家 → **只有赢家 settle、只有赢家学** → B 训练物理上不触碰专家 A 的
  任何权重 → 遗忘在构造上不存在；
- **琢磨只该用时用**：只赢家 settle（便宜），深度按误差收敛自适应，双过程回退兜底
  （琢磨失败 → 赢家自己的首猜）；
- 学习强度 = 原则一（每专家都是摊销首猜+残差，不被 settle 稀释）。

无 BP：全部权重更新为局部误差驱动；EWC（importance=更新量级）按专家独立累计/保护，
作为路由不完美分离时的兜底（专家内残留冲突仍受保护）。
"""
import torch
from tla.cognitive.pcn_amortized import AmortizedResidualPCN


class AmortizedMoEPCN:
    """dims = [in, hidden]（每专家独立）。接口与 AmortizedResidualPCN 对齐。"""

    def __init__(self, dims, out_dim, n_experts=2, lr_inf=0.1, prior=0.0,
                 mu_max=5.0, seed=None, obs_dim=None):
        assert len(dims) == 2
        self.dims = dims
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.lr_inf = lr_inf
        self.prior = prior
        self.mu_max = mu_max
        # 路由只看观测部分（前 obs_dim 维）的重建误差——共享 LTC 的 h 分量对两专家
        # 同步起伏，会把任务相关信号淹没（B 域 50/50 的根因）
        self.obs_dim = obs_dim if obs_dim is not None else min(out_dim, dims[0])
        self.expert = [AmortizedResidualPCN(dims=dims, out_dim=out_dim, lr_inf=lr_inf,
                                            prior=prior, mu_max=mu_max,
                                            seed=(seed if seed is not None else 0) + 100 + e)
                       for e in range(n_experts)]
        self.last_routing = torch.ones(n_experts) / n_experts
        self.r_learn = torch.ones(n_experts) / n_experts
        self.usage = torch.zeros(n_experts) + 1e-3      # 使用率 EMA（平手时打破对称/防死专家）
        self.winner = 0
        self.last_max_err = 0.0
        self.n_novel = 0                     # 诊断：平手路由触发计数
        self.n_route = 0

    def reset(self):
        for ex in self.expert:
            ex.reset()

    # ---- 路由：暖状态**观测部分**重建误差（“哪个专家的内部状态最能重建当前观测”→
    #      任务对齐；排除共享 LTC h 分量的同步噪声）；平手（误差几乎相同）时用使用率
    #      打破对称（冷启动交替、防死专家）。----
    def _route(self, x, update_proto=True):
        errs = [float(torch.mean(
                    (x[:self.obs_dim] - torch.tanh(ex.W_1 @ ex.mu_1 + ex.b_1)[:self.obs_dim]) ** 2
                ).item()) for ex in self.expert]
        e_min = max(min(errs), 1e-9)
        tie = abs(errs[0] - errs[1]) < 0.05 * e_min
        if tie and update_proto:
            self.n_novel += 1
            hard_idx = int(torch.argmin(self.usage).item())   # 平手 → 最少使用的专家
        else:
            hard_idx = int(torch.argmin(torch.tensor(errs)).item())
        if update_proto:
            self.n_route += 1
            self.usage = 0.99 * self.usage
            self.usage[hard_idx] += 0.01
        r = torch.softmax(-torch.tensor(errs) / e_min, dim=0)  # 按最小误差归一化温度
        onehot = torch.zeros_like(r)
        onehot[hard_idx] = 1.0
        self.r_learn = 0.8 * onehot + 0.2 * r
        self.last_routing = r
        self.winner = hard_idx
        return r, hard_idx

    # ---- 误差（只算赢家；路由只定一次，proto 不在此更新）----
    def errors(self, x, target=None):
        _, hard = self._route(x, update_proto=False)
        ex = self.expert[hard]
        e_0, e_1, e_out, pred_base, res, pred_total = ex.errors(x, target)
        return e_0, e_1, e_out, pred_base, res, pred_total

    def max_err(self, e_0, e_1):
        return max(torch.max(torch.abs(e_0)).item(), torch.max(torch.abs(e_1)).item())

    # ---- 推理（只赢家 settle；推理时 proto 冻结防中途翻转）----
    def settle_step(self, x, target=None, lr_inf=None):
        _, hard = self._route(x, update_proto=False)
        ex = self.expert[hard]
        last = ex.settle_step(x, target, lr_inf=lr_inf)
        self.last_max_err = last
        return last

    def settle(self, x, target=None, steps=4):
        last = 0.0
        for _ in range(steps):
            last = self.settle_step(x, target)
        return last

    def readout(self, x):
        _, hard = self._route(x, update_proto=False)
        return self.expert[hard].readout(x)

    def guess(self, x):
        """系统1：赢家专家的摊销首猜（双过程回退的兜底输出）。"""
        _, hard = self._route(x, update_proto=False)
        return self.expert[hard].W_base @ x + self.expert[hard].b_base

    # ---- EWC 突触巩固（按专家独立累计/保护；importance=更新量级，全局部）----
    def start_consolidation(self):
        self._imp = {k: torch.zeros_like(v) for k, v in self._params().items()}

    def finalize_consolidation(self):
        self._ref = {k: v.clone() for k, v in self._params().items()}
        for k in self._imp:
            mx = self._imp[k].max()
            if mx > 0:
                self._imp[k] = self._imp[k] / mx

    def _params(self):
        p = {}
        for e, ex in enumerate(self.expert):
            for k, v in ex._params().items():
                p[f"e{e}_{k}"] = v
        return p

    def _apply(self, k, new):
        e_idx = int(k[1:k.index("_")])
        setattr(self.expert[e_idx], k[k.index("_") + 1:], new)

    # ---- 学习（只赢家学；路由定一次、赢家冻结整个 settle+learn，防中途翻转）----
    def learn_step(self, x, target, lr=0.01, settle_steps=4, wd=1e-4,
                   freeze_base=False, consolidate=False, protect=False, lam=1.0):
        _, hard = self._route(x, update_proto=True)    # 路由定一次 + proto push-pull 同步发生
        ex = self.expert[hard]
        ex.settle(x, target, steps=settle_steps)        # 冻结赢家 settle（内部不重路由）
        mse = ex.learn_step(x, target, lr=lr, settle_steps=0, wd=wd,
                            freeze_base=freeze_base)
        if consolidate and hasattr(self, "_imp"):
            # 赢家专家的更新量级累计（importance 按专家独立）
            e_0, e_1, e_out, pred_base, _, _ = ex.errors(x, target)
            e_base = target - pred_base
            g1 = 1.0 - torch.tanh(ex.W_1 @ ex.mu_1 + ex.b_1) ** 2
            imp = {
                "W_base": torch.outer(e_base, x).abs(),
                "b_base": e_base.abs(),
                "W_1": torch.outer(g1 * e_0, ex.mu_1).abs(),
                "b_1": (g1 * e_0).abs(),
                "W_out": torch.outer(e_out, ex.mu_1).abs(),
                "b_out": e_out.abs(),
            }
            for k, v in imp.items():
                self._imp[f"e{hard}_{k}"] += v
        if protect and hasattr(self, "_ref"):
            for k, v in self._params().items():
                if k.startswith(f"e{hard}_"):
                    pull = lr * lam * self._imp[k] * (self._ref[k] - v)
                    self._apply(k, v + pull)
        return mse

    # ---- 批训练（mini-batch：按赢家分桶累计，逐专家应用；路由逐样本定一次）----
    def learn_batch(self, xs, targets, lr=0.01, settle_steps=4, wd=1e-4):
        B = len(xs)
        acc = {k: torch.zeros_like(v) for k, v in self._params().items()}
        e_out_sum = 0.0
        for x, t in zip(xs, targets):
            _, hard = self._route(x, update_proto=True)   # 路由定一次 + proto push-pull
            ex = self.expert[hard]
            ex.settle(x, t, steps=settle_steps)
            e_0, e_1, e_out, pred_base, _, _ = ex.errors(x, t)
            e_base = t - pred_base
            g1 = 1.0 - torch.tanh(ex.W_1 @ ex.mu_1 + ex.b_1) ** 2
            acc[f"e{hard}_W_base"] += torch.outer(e_base, x)
            acc[f"e{hard}_b_base"] += e_base
            acc[f"e{hard}_W_1"] += torch.outer(g1 * e_0, ex.mu_1)
            acc[f"e{hard}_b_1"] += g1 * e_0
            acc[f"e{hard}_W_out"] += torch.outer(e_out, ex.mu_1)
            acc[f"e{hard}_b_out"] += e_out
            e_out_sum += float(torch.mean(e_out ** 2).item())
        cur = self._params()
        for k in acc:
            if "W" in k:                       # 仅权重矩阵衰减 wd，偏置不衰减
                self._apply(k, (1.0 - lr * wd) * cur[k] + (lr / B) * acc[k])
            else:
                self._apply(k, cur[k] + (lr / B) * acc[k])
        return e_out_sum / B
