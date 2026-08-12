"""隐藏规则事件世界（Hidden-Rule Event World）——奇点-薛定谔世界模型的实验场地。

哲学对应（薛定谔叠加态世界模型原版）：现实在"被观测之前"处于多可能状态叠加；
事件间隔（timing）是区分隐藏规则的唯一证据——单值世界模型学不会（预测恒错），
叠加模型需同时维持"规则 A/B/C"多个假设分支，靠事件时刻坍缩。

机制：
  - 每条规则 r 有特征事件间隔分布：interval = base_r + jitter（规则越慢越难学）；
  - 每 episode 激活一条规则（隐藏，不直接观测）；
  - 观测 = [event_flag（带噪声：5% 误报/漏报）, context（弱规则信号：P(码=规则码)=0.6）]；
  - 目标是时序性的：下一 tick 是否事件 + 距下次事件的 tick 数——预测后者需要
    "距上次事件多久"的时间状态（奇点衰减相 / LTC 泄漏积分 / 或无状态=不可知）。

时间语义：`remain` = 距下次事件的 tick 数（0 = 本 tick 触发事件）。事件触发后
重新采样间隔；观测反映本 tick 是否事件（含噪声）；目标 = [下一 tick 是否事件,
归一化 time_to_next]。
"""
import numpy as np
import torch


# 预注册规则集：事件间隔分布（base, jitter）——时间尺度分三档
RULES = {
    "fast": (5, 1),    # 规则 fast：事件约每 5±1 tick
    "mid": (10, 2),    # 规则 mid：约每 10±2 tick
    "slow": (20, 5),   # 规则 slow：约每 20±5 tick
}
RULE_CODES = {"fast": 0b01, "mid": 0b10, "slow": 0b11}
OBS_DIM = 2            # [event_flag, context]
TARGET_DIM = 2         # [next_event_flag, time_to_next 归一化]


class EventWorld:
    """隐藏规则事件世界。reset(rule) 开新 episode；step() 推进 1 tick 返回 (观测, 目标)。"""

    def __init__(self, rule="mid", noise=0.0, seed=0, t_norm=25.0):
        """noise=0.0（实验首测后修改，2026-08-11）：事件 flag 翻转噪声（原 5%）会重置
        所有基板的时钟，产生"状态说刚事件、目标说还早"的矛盾训练对，淹没基板时间解码
        差异（三基板同受害，LTC 亦然）。事件观测改为可靠；噪声鲁棒性列为后续工作
        （判据未改，此为测试设置修正，如实记录）。"""
        assert rule in RULES, f"未知规则 {rule}"
        self.rule = rule
        self.noise = noise
        self.seed = seed
        self.t_norm = t_norm              # time_to_next 归一化尺度（最大间隔≈slow 25）
        self.rng = np.random.default_rng(seed)
        self.remain = 0
        self.tick = 0
        self.reset(rule)

    def reset(self, rule=None):
        if rule is not None:
            assert rule in RULES
            self.rule = rule
        base, jit = RULES[self.rule]
        self.base, self.jit = base, jit
        self.tick = 0
        self._schedule_next()
        return self._obs(0.0)             # 起始 tick 无事件

    # ---- 内部 ----
    def _schedule_next(self):
        self.remain = max(1, int(round(self.rng.normal(self.base, self.jit))))

    def _obs(self, event_now):
        # event_flag：本 tick 是否事件（噪声下可能误报/漏报）
        raw = 1.0 if event_now > 0 else 0.0
        if self.rng.random() < self.noise:
            raw = 1.0 - raw
        # context：弱规则信号——P(输出规则码位)=0.6（不直接泄露规则）
        code = RULE_CODES[self.rule]
        bit = float(code & 1)
        ctx = bit if self.rng.random() < 0.6 else (1.0 - bit)
        return torch.tensor([raw, ctx], dtype=torch.float32)

    def step(self):
        """推进 1 tick。事件在本 tick 触发（remain==0）→ 观测含事件 → 重新调度。"""
        event_now = 1.0 if self.remain == 0 else 0.0
        obs = self._obs(event_now)
        if self.remain == 0:
            self._schedule_next()         # 刚触发 → 重采样下个间隔
        else:
            self.remain -= 1
        time_to_next = float(self.remain)             # 下一 tick 视角
        next_event = 1.0 if time_to_next <= 1 else 0.0
        target = torch.tensor([next_event, time_to_next / self.t_norm],
                              dtype=torch.float32)
        self.tick += 1
        return obs, target

    # ---- 数据生成 ----
    def episodes(self, n, T, rule=None, seed_shift=0):
        """n 个 episode（长度 T），返回 (obs_seq, target_seq) 列表。"""
        out = []
        for i in range(n):
            self.reset(rule)
            obs_seq, tgt_seq = [], []
            for _ in range(T):
                o, t = self.step()
                obs_seq.append(o)
                tgt_seq.append(t)
            out.append((torch.stack(obs_seq), torch.stack(tgt_seq)))
        return out


def rule_sequence(rule_list, T_each):
    """任务序列：规则按列表切换（如 ['fast','mid','slow']，每段 T_each tick）。"""
    return [(r, T_each) for r in rule_list]
