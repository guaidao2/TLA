"""生长/修剪门 + 校准期（⑨）。

- 生长门：grow ⟺ (avg_error > ε_e) ∧ (novelty > ε_n) ∧ (energy > ε_E)
  —— 惊奇 × 新奇 × 能量，只有"新情境 + 有资源"才长（防已知情境冗余生长）；
- 新单元：低增益 calib_min=0.3 → 校准期线性升到 1.0（防噪声污染）；
- 修剪：importance 低于阈值 → 释放容量（mask 置 0，权重保留，重新激活即恢复）。
"""
import torch
from tla.meta.importance import ImportanceTracker


class CapacityManager:
    def __init__(self, n_units, calib_min=0.3, calib_window=100,
                 prune_threshold=1e-6, prune_rel=0.1, grow_thresholds=(0.3, 0.3, 0.3),
                 prune_interval=200, grow_interval=200):
        self.n = n_units
        self.mask = torch.ones(n_units, dtype=torch.bool)
        self.gain = torch.ones(n_units)
        self.age = torch.full((n_units,), float(calib_window))  # 既有单元初始即已校准（gain=1.0）
        self.calib_min, self.calib_window = calib_min, calib_window
        self.prune_threshold = prune_threshold
        self.prune_rel = prune_rel      # 相对阈值：importance < prune_rel×均值 才剪（尺度无关）
        self.eps_e, self.eps_n, self.eps_E = grow_thresholds
        self.prune_interval, self.grow_interval = prune_interval, grow_interval
        self.tick = 0
        self.imp = ImportanceTracker(n_units)
        self.last_pruned = 0
        self.last_grown = None

    def update(self, activations, error_contrib, avg_error, energy_level):
        self.tick += 1
        self.imp.update(activations, error_contrib)
        # 校准 ramp：新单元 gain 0.3 → 1.0
        self.age = self.age + self.mask.float()
        ramp = (self.age / self.calib_window).clamp(0.0, 1.0)
        self.gain = self.calib_min + (1.0 - self.calib_min) * ramp
        self.gain = self.gain * self.mask.float()

    def maybe_prune(self):
        if self.tick % self.prune_interval == 0:
            # 相对阈值（防绝对阈值与数据尺度不匹配导致批量误剪）：
            # importance < max(绝对下限, prune_rel × 活跃单元均值) 才剪
            active_mean = self.imp.imp[self.mask].mean().item() if self.mask.any() else 0.0
            thr = max(self.prune_threshold, self.prune_rel * active_mean)
            cand = self.mask & (self.imp.imp < thr)
            if cand.sum() <= 0.3 * self.mask.sum():   # 防级联：单次最多剪 30%
                self.mask[cand] = False
                self.gain[cand] = 0.0
                self.last_pruned = int(cand.sum().item())
        return self.last_pruned

    def maybe_grow(self, avg_error, novelty, energy_level):
        if (self.tick % self.grow_interval == 0
                and avg_error > self.eps_e and novelty > self.eps_n
                and energy_level > self.eps_E):
            dormant = ~self.mask
            if dormant.any():
                i = dormant.nonzero()[self.imp.imp[dormant].argmin()].item()
                self.mask[i] = True
                self.age[i] = 0.0
                self.gain[i] = self.calib_min
                self.last_grown = i
                return i
        return None

    def gate_vector(self):
        return self.gain
