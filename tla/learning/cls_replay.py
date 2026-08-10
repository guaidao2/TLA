"""CLS 重放缓冲（防遗忘，复用 NLA P9 思路：99.5% vs 18.1%）。

- surprise 加权抽样（重放"最意外"的旧样本）；
- 重放时 meta_update=False（不污染生长统计/能量，同 NLA `_replay_step`）。
"""
import torch


class ReplayBuffer:
    def __init__(self, capacity=2048, replay_prob=0.3, batch=8, seed=None):
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.batch = batch
        self.items = []  # (s_t, s_next, surprise)
        self.rng = torch.Generator().manual_seed(seed) if seed is not None else None

    def push(self, s_t, s_next, surprise):
        self.items.append((s_t.clone(), s_next.clone(), float(surprise)))
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
        surp = torch.tensor([it[2] for it in self.items[-self.capacity:]])
        w = torch.softmax(surp / max(surp.max().item(), 1e-9), dim=0)
        idx = torch.multinomial(w, self.batch, replacement=False, generator=self.rng)
        n = 0
        for i in idx.tolist():
            s_t, s_next, _ = self.items[-self.capacity:][i]
            trainer.replay_step(s_t, s_next)
            n += 1
        return n
