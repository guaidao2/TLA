"""CLS 重放缓冲（防遗忘，复用 NLA P9 思路：99.5% vs 18.1%）。

- 均匀抽样（P-LEARN-1 实测修订：surprise 加权在 A→B 任务切换时被新任务高 surprise 样本垄断，
  旧任务样本几乎抽不到，防遗忘失效；均匀抽样保证旧记忆稳定重放）；
- 重放时 meta_update=False（不污染生长统计/能量，同 NLA `_replay_step`），
  且**还原样本当时的身体上下文**（存 h_ctx 一并重放——重置 LTC 教的是"无上下文版本"的映射）。
"""
import torch


class ReplayBuffer:
    def __init__(self, capacity=8192, replay_prob=0.3, batch=8, seed=None):
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.batch = batch
        self.items = []  # (s_t, s_next, h_ctx, surprise)
        self.rng = torch.Generator().manual_seed(seed) if seed is not None else None

    def push(self, s_t, s_next, h_ctx, surprise):
        self.items.append((s_t.clone(), s_next.clone(), h_ctx.clone(), float(surprise)))
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def __len__(self):
        return len(self.items)

    def maybe_replay(self, trainer):
        if len(self.items) < self.batch:
            return 0
        r = torch.rand(1, generator=self.rng).item() if self.rng is not None else torch.rand(1).item()
        if r >= self.replay_prob:
            return 0
        idx = torch.randint(len(self.items), (self.batch,), generator=self.rng)
        n = 0
        for i in idx.tolist():
            s_t, s_next, h_ctx, _ = self.items[i]
            trainer.replay_step(s_t, s_next, h_ctx)
            n += 1
        return n
