"""奇点基板（Singularity Substrate）：向量化奇点神经元库，LTC 的换装替代。

设计：hidden 个奇点细胞并联，每个细胞一个标量状态 h_i ∈ [0, h_max]（幽灵态 ~0.02 /
饱和态 ~0.9 双不动点）；输入 I_i = W_in@s + W_h@h + b（固定随机投影，与 LTC 相同语义——
基板参数不参与学习，只做固定非线性动力学特征提取）。

与 LTC 的关键差异（行为层面）：
- LTC：h 是连续的（tanh 软饱和），时间常数由输入调制；
- 奇点：h 呈"幽灵态/饱和态"双簇（稀疏激活——I > 边界 ≈0.05 才暴胀），且状态携带
  可解码的时间戳（SN-3 能力）。
输入尺度标定：奇点暴胀边界 I≈0.05（远低于 LTC 工作区间），input_scale 需相应缩小
（默认 0.4 → 每细胞 I 的 std ≈ 0.09，混合暴胀/幽灵的稀疏区间）。判据复测见
criteria/substrate_swap.py（换装后学习/琢磨/防遗忘是否保持）。
"""
import torch
from tla.substrate.singularity_cell import SingularityCell


class SingularitySubstrate:
    def __init__(self, in_dim, hidden, input_scale=0.4, alpha=0.5, beta=1.0,
                 eps=1e-4, h_max=1.0, lam=0.05, gamma=1.0, lam_dark=1e-3,
                 seed=None):
        self.in_dim, self.hidden = in_dim, hidden
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        g = 1.0 / (hidden ** 0.5)
        self.W_in = torch.randn(hidden, in_dim, generator=gen) * g * input_scale
        self.W_h = torch.randn(hidden, hidden, generator=gen) * 0.3 * g * input_scale
        self.b = torch.zeros(hidden)
        self.cells = [SingularityCell(alpha=alpha, beta=beta, eps=eps, h_max=h_max,
                                      lam=lam, gamma=gamma, lam_dark=lam_dark,
                                      input_gated=True)
                      for _ in range(hidden)]
        self.h = torch.zeros(hidden)

    def reset(self):
        for c in self.cells:
            c.reset(0.0)
        self.h = torch.zeros_like(self.h)

    def forward(self, x):
        I = self.W_in @ x + self.W_h @ self.h + self.b
        hs = []
        for i, c in enumerate(self.cells):
            c.step(float(I[i].item()))
            hs.append(c.h)
        self.h = torch.tensor(hs, dtype=torch.float32)
        return self.h
