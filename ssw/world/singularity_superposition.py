"""奇点-薛定谔叠加世界模型（Singularity-Superposition World Model, SSW）。

组合设计（判据先于实现）：
  - **分支 = 状态基板 + 日程读出头**：基板（奇点细胞 / LTC / 无状态）固定，读出头可训练；
    分支 i 先验 = 规则 i 的事件间隔日程（监督校准——"生来带着假设"）；
  - **振幅 = 预测误差 softmax**（薛定谔原版坍缩机制）：观测到达 → 分支预测与实际越吻合
    振幅越高 → 证据坍缩掉错误假设；
  - **时间戳**：奇点基板的衰减相编码"距上次事件多久"——读出头从状态解码下次事件时刻；
    LTC 泄漏积分给近似时间信息；无状态基板没有时间信息（对照）；
  - **分裂**：熵持续接近上限（所有分支都无法区分观测）+ 最差分支 miss 超阈值 → 从最强
    分支克隆+扰动出新分支（生长）。

训练纪律：判据先于实现、固定种子可复现；分支日程校准（phase 1）与叠加在线训练
（phase 2）都允许 BP（这是独立新架构，非 TLA——TLA 是否引用另行决定）。
"""
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from tla.substrate.singularity_cell import SingularityCell
from tla.substrate.singularity_substrate import SingularitySubstrate
from tla.substrate.ltc_cell import LTCCell
from ssw.env.event_world import RULES

OBS_DIM = 2
TARGET_DIM = 2
RULE_BASE = [RULES[r][0] for r in ("fast", "mid", "slow")]   # [5, 10, 20]


class SingularityClockBank:
    """奇点时钟银行：N 个独立奇点细胞（无循环——避免 W_h 级联恒热），每细胞固定增益 w_i。
    事件（event_flag=1）→ 各细胞以不同增益暴胀到不同高度；安静 → 独立指数衰减
    （λ=0.08，τ≈12.5）→ 银行状态 = 衰减剖面 = 自包含时间戳（距上次事件多久）。
    特征 = [h, ln(h+ε)]：衰减是指数的 → 对数线性化 → 时间解码为线性问题
    （奇点的结构优势；LTC 泄漏积分无此性质）。无状态基板（'none'）不建此类。"""

    def __init__(self, n=16, seed=0, alpha=1.5, beta=3.0, lam=0.08,
                 eps=0.01, w_lo=0.5, w_hi=1.6):
        rng = np.random.default_rng(seed)
        self.w = torch.tensor(np.linspace(w_lo, w_hi, n), dtype=torch.float32)
        self.cells = [
            SingularityCell(alpha=alpha, beta=beta, eps=eps, lam=lam,
                            gamma=1.0, lam_dark=1e-3, dt=1.0,
                            input_gated=True)
            for _ in range(n)
        ]
        self.h = torch.zeros(n)
        self.f0 = None                     # 上次事件后的峰值特征（解析反演需 h0）
        self.h0_h = None                   # 上次事件后的 h 空间峰值（均值反演用，稳健）
        self.feature_dim = n               # f = -ln(h+eps)/4.6（对数线性化特征）

    def forward(self, obs):
        event = float(obs[0])
        for i, c in enumerate(self.cells):
            c.step(float(self.w[i]) * event)
            self.h[i] = c.h
        if event > 0.5:
            self.f0 = self.features().clone()   # 事件 tick：记录本事件后的峰值特征
            self.h0_h = self.h.clone()          # h 空间峰值（均值反演用）
        return self.features()

    def features(self):
        # f = -ln(h+eps)/4.6：衰减时 f 线性增长（指数→对数线性化），值域 [0,2] 不饱和
        return torch.log(1.0 / (self.h + 1e-4)) / 4.6

    def reset(self):
        # reset 到幽灵态（h*）而非 0：f 从 0.95 起（与完全衰减后一致，避免 f∈[1,2]
        # 越界外推——首测发现前 20 tick 全错的根因）
        for c in self.cells:
            c.reset(c.ghost())
        self.h = torch.tensor([c.h for c in self.cells], dtype=torch.float32)
        self.f0 = None
        self.h0_h = None


class LTCRecencyBank:
    """LTC 适配：与奇点时钟银行同接口（forward/features/reset/feature_dim）。
    特征 = h（泄漏积分状态；无 log 线性化性质——衰减非指数纯形式）。"""

    def __init__(self, hidden=16, seed=0):
        self._ltc = LTCCell(in_dim=OBS_DIM, hidden=hidden, seed=seed)
        self.h = self._ltc.h
        self.feature_dim = hidden

    def forward(self, obs):
        self.h = self._ltc.forward(obs)
        return self.features()

    def features(self):
        return self.h

    def reset(self):
        self._ltc.reset()
        self.h = self._ltc.h


def make_substrate(kind, hidden=16, seed=0, **kw):
    """分支状态基板（固定）。kind: 'singularity'（时钟银行）| 'ltc' | 'none'。
    SSW 专用标定（2026-08-11，判据先于实现）：
      事件是单 tick 脉冲 → 时钟银行（α=1.5, β=3.0, 增益 0.5-1.6）：事件 tick 各细胞
      暴胀到 0.13-1.0 分级高度；安静 → 独立衰减（λ=0.08, τ≈12.5）覆盖 5-20 tick 间隔；
      特征含 ln(h+ε) → 指数衰减对数线性化 → 时间解码线性化（奇点结构优势）。
      （用含 W_h 的 substrate 有复现级联恒热问题——实验首测发现，已弃用。）"""
    if kind == "singularity":
        return SingularityClockBank(n=hidden, seed=seed, **kw)
    if kind == "ltc":
        return LTCRecencyBank(hidden=hidden, seed=seed)
    if kind == "none":
        return None
    raise ValueError(kind)


def make_opt(module, lr=1e-3, interval_lr=0.1):
    """AdamW 参数分组：analytic 头的 interval 标量梯度被 /t_norm 稀释（~0.024/步），
    但 Adam 步长≈lr（梯度已归一化），lr 太大 → 目标附近振荡（0.5 实测振荡 ±0.5、
    SW-1 时间误差 ±0.02）；interval_lr=0.1 折中：振荡 ±0.1（时间误差 ±0.004，
    SW-1 余量充足）且分裂新分支 10→20 在 ~150 步内可达（0.05 实测 231 步不足，
    SW-3b 新分支振幅 0.259 < 0.3）。每组存 base_lr 供可塑性门控缩放（gated 模式）。"""
    main = [p for n, p in module.named_parameters() if "interval" not in n]
    iv = [p for n, p in module.named_parameters() if "interval" in n]
    groups = [{"params": main, "lr": lr, "base_lr": lr}]
    if iv:
        groups.append({"params": iv, "lr": interval_lr, "base_lr": interval_lr})
    return torch.optim.AdamW(groups)


class ScheduleBranch(nn.Module):
    """一个"日程专家"分支：基板状态 + 读出头 → [下一事件概率, 归一化 time_to_next]。
    读出头输入 [h, obs]（无状态基板：仅 [obs]）。
    head_kind="mlp"：自由形式 MLP（LTC/无状态）。
    head_kind="analytic"：奇点解析反演读出头——状态 f 可精确反演为"距上次事件 tick"
    （指数衰减 → 解析可逆，奇点结构优势显式化；LTC 泄漏积分不可解析反演，只能学）。
    解析头唯一可学参数 = 分支间隔 I_i（标量，分裂时新分支可学新间隔）。"""

    def __init__(self, substrate, obs_dim=OBS_DIM, head_hidden=32, seed=0,
                 head_kind="mlp", interval_init=10.0, t_norm=25.0):
        super().__init__()
        self.substrate = substrate
        self.head_kind = head_kind
        self.t_norm = t_norm
        g = torch.Generator().manual_seed(seed)
        if head_kind == "analytic":
            # 解析反演：f = -ln(h+eps)/4.6 → h = e^{-4.6f} - eps；
            # h(t) = h* + (h0-h*)e^{-λt} → k̂ = -(1/λ)ln((h-h*)/(h0-h*))
            # time = (I_i - k̂)/t_norm；next_event = σ((k̂ - I_i + 1)·5)（soft 阈值）
            self.interval = nn.Parameter(torch.tensor(float(interval_init)))
            self.register_buffer("lam", torch.tensor(0.08))
            self.register_buffer("h_star", torch.tensor(1e-3 / 0.08))
            self.register_buffer("eps_f", torch.tensor(1e-4))
            self.head = None
        else:
            in_dim = (substrate.feature_dim if substrate is not None else 0) + obs_dim
            self.head = nn.Sequential(
                nn.Linear(in_dim, head_hidden), nn.Tanh(),
                nn.Linear(head_hidden, TARGET_DIM))
            for p in self.head.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, generator=g)
                else:
                    nn.init.zeros_(p)
        self.state = None

    def reset(self):
        if self.substrate is not None:
            self.substrate.reset()
        self.state = None

    def readout(self, obs):
        """用当前状态读出头（不推进状态）。状态 None → 基板初始特征。"""
        if self.head_kind == "analytic":
            # 均值状态反演（稳健）：h̄ = mean(h)，h̄0 = mean(h0_h)（事件峰值），
            # k̂ = -(1/λ)ln((h̄-h*)/(h̄0-h*))。逐细胞反演再平均有偏（低增益细胞
            # h0-h* 极小 → k̂ 爆炸），均值状态反演与探针一致（slow 区间 20±5）。
            if self.substrate.h0_h is None:
                k = torch.zeros(())
            else:
                h_bar = self.substrate.h.mean()
                h0_bar = self.substrate.h0_h.mean()
                num = (h_bar - self.h_star).clamp(min=1e-9)
                den = (h0_bar - self.h_star).clamp(min=1e-9)
                # 注：λ=0.08 名义 vs 欧拉离散实际 -ln(0.92)≈0.0834 → k̂ ~4% 偏小，
                # 由可学 interval 自补偿（"解析可逆"是近似，如实披露）
                k = (-(torch.log(num / den)) / self.lam).clamp(min=0.0)
            time = ((self.interval - k) / self.t_norm).clamp(0.0, 1.0)
            nxt = torch.sigmoid((k - self.interval + 1.0) * 5.0)
            return torch.stack([nxt, time])
        if self.substrate is not None:
            h = self.state if self.state is not None else self.substrate.features()
            return self.head(torch.cat([h, obs]))
        return self.head(obs)

    def advance(self, obs):
        """推进状态：substrate.forward(obs) 更新特征。无状态基板为空操作。"""
        if self.substrate is not None:
            self.state = self.substrate.forward(obs)

    def forward(self, obs):
        self.advance(obs)
        return self.readout(obs)


class SSWModel(nn.Module):
    """奇点-薛定谔叠加世界模型。"""

    def __init__(self, substrate_kind="singularity", n_branches=3,
                 max_branches=5, temp=0.02, seed=0, hidden=16, head_hidden=64,
                 head_kind="mlp", plasticity="uniform"):
        """head_kind="mlp" 默认（LTC/无状态）；"analytic" 仅奇点（解析反演读出头）。
        plasticity="uniform"（默认全可塑）| "gated"（可塑性门控：η_i = η·(1−amp_i)——
        非主导分支保持可塑朝新规则漂移、主导分支提交防遗忘，SW-5 实验）。
        temp=0.02（2026-08-11 标定）：坍缩误差是归一化 time MSE（量级 0.005-0.16），
        原 temp=1.0 的 softmax(−err) 几乎平坦（振幅≈1/3，坍缩不集中——实测
        fast 数据上正确分支仅 0.36）；temp=0.02 使正确/错误分支 logit 差 ~2 → 收敛
        （0.85/0.14/0.00）。判据未改，此为坍缩温度标定。"""
        super().__init__()
        self.kind = substrate_kind
        self.temp = temp
        self.max_branches = max_branches
        self.seed = seed
        self.hidden = hidden
        self.head_hidden = head_hidden
        self.head_kind = head_kind
        self.plasticity = plasticity
        assert plasticity in ("uniform", "gated", "frozen"), \
            f"未知 plasticity 策略 {plasticity}（拼写错误会静默落到 uniform，安全审查补防）"
        self.branches = nn.ModuleList([
            ScheduleBranch(make_substrate(substrate_kind, hidden=hidden,
                                          seed=seed + 100 * i),
                           head_hidden=head_hidden,
                           seed=seed + 1000 * i,
                           head_kind=head_kind,
                           interval_init=RULE_BASE[i] if head_kind == "analytic" else 10.0)
            for i in range(n_branches)
        ])
        self.amps = torch.ones(n_branches) / n_branches
        self.misses = torch.zeros(n_branches)
        self.err_hist = deque(maxlen=20)      # 最近坍缩的"最优分支误差"（分裂信号）
        self.last_entropy = float(np.log(n_branches))
        self.opt = make_opt(self)

    # ---- 预测/坍缩 ----
    def branch_errors(self, obs, target):
        """各分支对 (obs→预测) vs target 的误差（坍缩依据）。"""
        errs = []
        with torch.no_grad():
            for b in self.branches:
                p = b(obs)
                e = float(torch.mean((p - target) ** 2).item())
                errs.append(e)
        return errs

    def collapse(self, errs, tick=0):
        """观测到达 → 按预测误差更新振幅（softmax 坍缩）+ 熵记录 + 分裂信号。"""
        logits = torch.tensor([-e / max(0.05, self.temp) for e in errs])
        self.amps = torch.softmax(logits, dim=0)
        probs = self.amps.numpy()
        self.last_entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        self.err_hist.append(min(errs))
        for i, e in enumerate(errs):
            if e > 0.05:
                self.misses[i] += 1.0

    # ---- 单 tick 流程：advance（状态纳入本 tick 事件）→ readout → 坍缩（监督信号）----
    # target_t = 事件 t 之后的时间状态；预测用处理完 obs_t 的状态（teacher-forced 坍缩：
    # 世界提供真实时间状态作监督，文档如实披露）。
    # 坍缩误差只用 time 维（pred[1] vs target[1]）：时间戳是判别规则的唯一证据，
    # next_event 输出是派生的二值信号，纳入误差会稀释坍缩判别力（实测 SW-2 恶化）。
    def step(self, obs, target):
        for b in self.branches:
            b.advance(obs)
        preds = [b.readout(obs) for b in self.branches]
        errs = [float((p[1] - target[1]) ** 2) for p in preds]
        self.collapse(errs)
        out = torch.zeros(TARGET_DIM)
        for i, p in enumerate(preds):
            out = out + self.amps[i] * p
        return out

    def train_step(self, obs, target):
        """一步在线训练（振幅加权损失；advance→readout→loss→backward）。
        plasticity="gated"：逐分支学习率 η_i = η·(1−amp_i)（非主导分支可塑、
        主导分支提交）——用分支级优化器组实现（SW-5 实验）。"""
        for b in self.branches:
            b.advance(obs)
        preds = [b.readout(obs) for b in self.branches]
        total = torch.zeros(())
        for i, b in enumerate(self.branches):
            loss = torch.mean((preds[i] - target) ** 2)
            total = total + self.amps[i] * loss
        total.backward()
        if self.plasticity == "frozen":
            # 全冻结（η=0）：只算损失不更新（SW-5 极端对照——终审修复：
            # 原实现落到 uniform 分支，frozen=uniform 是 bug 而非机制）
            self.opt.zero_grad(set_to_none=True)
        elif self.plasticity == "gated":
            # 逐分支更新：η_i = base_lr·(1−amp_i)（非主导可塑、主导提交）
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            for i, b in enumerate(self.branches):
                gate = 1.0 - float(self.amps[i])
                if gate <= 1e-3 or not any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in b.parameters()):
                    continue
                if not hasattr(b, "_opt") or b._opt is None:
                    b._opt = make_opt(b)
                for g in b._opt.param_groups:
                    g["lr"] = g["base_lr"] * gate
                b._opt.step()
            self.opt.zero_grad(set_to_none=True)
        else:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return total.item()

    # ---- 生长 ----
    def should_split(self, best_err_threshold=0.03, fail_ratio=0.6):
        """何时分裂：最近坍缩中"最优分支"的误差**持久**高于阈值（窗口内失败占比
        ≥ fail_ratio——最好的假设持续失败 → 容量不足）且最差分支 miss 超阈值。
        均值触发对偶发匹配敏感（专家偶尔碰巧对上不规则间隔 → 均值掉线）；
        失败率更稳健（2026-08-11 修正，判据未改）。"""
        if len(self.branches) >= self.max_branches:
            return False
        if len(self.err_hist) < 10:
            return False
        fails = sum(1.0 for e in self.err_hist if e > best_err_threshold)
        worst = min(self.misses.tolist())
        return fails / len(self.err_hist) >= fail_ratio and worst > 5

    def split(self):
        """从最强分支克隆+扰动出新分支（继承日程近似 + 小扰动，头参数重训适应）。"""
        if len(self.branches) >= self.max_branches:
            return False
        parent_idx = int(torch.argmax(self.amps).item())
        parent = self.branches[parent_idx]
        child = ScheduleBranch(make_substrate(self.kind, hidden=self.hidden,
                                              seed=self.seed + 100 * len(self.branches)),
                               head_hidden=self.head_hidden,
                               seed=self.seed + 1000 * len(self.branches),
                               head_kind=self.head_kind,
                               interval_init=10.0)
        child.load_state_dict(parent.state_dict())   # 克隆（含 analytic 的 interval）
        with torch.no_grad():
            for p in child.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        self.branches.append(child)
        self.amps = torch.cat([self.amps, torch.tensor([0.1])])
        self.misses = torch.cat([self.misses, torch.zeros(1)])
        self.opt = make_opt(self)
        return True

    def reset(self):
        for b in self.branches:
            b.reset()
        self.amps = torch.ones(len(self.branches)) / len(self.branches)

    def branch_stats(self):
        return [(i, round(float(a), 3), int(m))
                for i, (a, m) in enumerate(zip(self.amps, self.misses))]


# ---- 日程校准（phase 1）：分支 i 只学规则 i 的日程 ----
def calibrate_schedule(model, world, rule_idx, rule, n_ep=20, T=60,
                       epochs=4, seed=0):
    """把第 rule_idx 个分支的读出头监督校准为规则 rule 的日程。
    其余分支冻结（不参与）。返回校准后该分支在测试集上的 time MSE。
    epochs=4：实测 2 欠训（sing≈ff）、8+ 过拟合训练噪声（校准预算，非判据）。"""
    torch.manual_seed(seed)
    eps = world.episodes(n=n_ep, T=T, rule=rule, seed_shift=seed)
    opt = make_opt(model.branches[rule_idx])
    target_branch = model.branches[rule_idx]
    # 时间解码只训练/评估"首事件后"的衰减相（时钟已启动才有时间证据；
    # 首事件前的相位对所有基板不可知——无事件=无时钟，2026-08-11 统一披露）
    for _ in range(epochs):
        for obs_seq, tgt_seq in eps:
            target_branch.reset()
            started = False
            for o, t in zip(obs_seq, tgt_seq):
                if float(o[0]) > 0.5:
                    started = True
                if not started:
                    continue
                opt.zero_grad()
                p = target_branch(o)
                loss = torch.mean((p - t) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(target_branch.parameters(), 1.0)
                opt.step()
    # 评估：该分支在 rule 的未见 episode 上的 time 预测 MSE（仅首事件后）
    test = world.episodes(n=4, T=T, rule=rule, seed_shift=seed + 77)
    mse = 0.0
    cnt = 0
    with torch.no_grad():
        for obs_seq, tgt_seq in test:
            target_branch.reset()
            started = False
            for o, t in zip(obs_seq, tgt_seq):
                if float(o[0]) > 0.5:
                    started = True
                if not started:
                    continue
                p = target_branch(o)
                mse += float((p[1] - t[1]) ** 2)
                cnt += 1
    return mse / max(cnt, 1)
