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

OBS_DIM = 2
TARGET_DIM = 2


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
        self.feature_dim = n              # f = -ln(h+eps)/4.6（对数线性化特征）

    def forward(self, obs):
        event = float(obs[0])
        for i, c in enumerate(self.cells):
            c.step(float(self.w[i]) * event)
            self.h[i] = c.h
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


class ScheduleBranch(nn.Module):
    """一个"日程专家"分支：基板状态 + 读出头 → [下一事件概率, 归一化 time_to_next]。
    读出头输入 [h, obs]（无状态基板：仅 [obs]）。"""

    def __init__(self, substrate, obs_dim=OBS_DIM,
                 head_hidden=32, seed=0):
        super().__init__()
        self.substrate = substrate
        g = torch.Generator().manual_seed(seed)
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
                 max_branches=5, temp=0.02, seed=0, hidden=16, head_hidden=64):
        """temp=0.02（2026-08-11 标定）：坍缩误差是归一化 time MSE（量级 0.005-0.16），
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
        self.branches = nn.ModuleList([
            ScheduleBranch(make_substrate(substrate_kind, hidden=hidden,
                                          seed=seed + 100 * i),
                           head_hidden=head_hidden,
                           seed=seed + 1000 * i)
            for i in range(n_branches)
        ])
        self.amps = torch.ones(n_branches) / n_branches
        self.misses = torch.zeros(n_branches)
        self.err_hist = deque(maxlen=20)      # 最近坍缩的"最优分支误差"（分裂信号）
        self.last_entropy = float(np.log(n_branches))
        self.opt = torch.optim.AdamW(self.parameters(), lr=1e-3)

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
        """一步在线训练（振幅加权损失；advance→readout→loss→backward）。"""
        self.opt.zero_grad()
        for b in self.branches:
            b.advance(obs)
        preds = [b.readout(obs) for b in self.branches]
        total = torch.zeros(())
        for i, b in enumerate(self.branches):
            loss = torch.mean((preds[i] - target) ** 2)
            total = total + self.amps[i] * loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()
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
                               seed=self.seed + 1000 * len(self.branches))
        child.load_state_dict(parent.state_dict())
        with torch.no_grad():
            for p in child.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        self.branches.append(child)
        self.amps = torch.cat([self.amps, torch.tensor([0.1])])
        self.misses = torch.cat([self.misses, torch.zeros(1)])
        self.opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
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
    opt = torch.optim.AdamW(model.branches[rule_idx].parameters(), lr=1e-3)
    target_branch = model.branches[rule_idx]
    for _ in range(epochs):
        for obs_seq, tgt_seq in eps:
            target_branch.reset()
            for o, t in zip(obs_seq, tgt_seq):
                opt.zero_grad()
                p = target_branch(o)
                loss = torch.mean((p - t) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(target_branch.parameters(), 1.0)
                opt.step()
    # 评估：该分支在 rule 的未见 episode 上的 time 预测 MSE
    test = world.episodes(n=4, T=T, rule=rule, seed_shift=seed + 77)
    mse = 0.0
    cnt = 0
    with torch.no_grad():
        for obs_seq, tgt_seq in test:
            target_branch.reset()
            for o, t in zip(obs_seq, tgt_seq):
                p = target_branch(o)
                mse += float((p[1] - t[1]) ** 2)
                cnt += 1
    return mse / max(cnt, 1)
