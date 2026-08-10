"""importance 长时窗累积（⑨）：滑动平均 激活×误差贡献。

α 极慢（长时窗）→ 防"刚重放过的技能因近期不活跃就被剪"（修剪与 CLS 重放的打架解法）。
"""
import torch


class ImportanceTracker:
    def __init__(self, n_units, alpha=1e-3):
        self.imp = torch.zeros(n_units)
        self.alpha = float(alpha)

    def update(self, activations, error_contrib):
        contrib = activations.abs() * error_contrib.abs()
        self.imp = self.imp + self.alpha * (contrib - self.imp)

    def prune_mask(self, threshold):
        return self.imp < threshold
