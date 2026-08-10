"""LTC 液态细胞（⑤ 时间是约束，公理⑤/①）。

τ(t) 输入调制时间常数；离散化 ODE；tanh 软饱和保证有界（P-PHY-3）。
零输入时 V → 静息 0（P-PHY-1，W_h 初始化很小保证不动点稳定）。
"""
import torch


class LTCCell:
    def __init__(self, in_dim, hidden, tau_min=1.0, tau_max=8.0, dt=0.2,
                 v_max=2.0, seed=None):
        self.in_dim, self.hidden = in_dim, hidden
        self.dt, self.v_max = dt, v_max
        self.tau_min, self.tau_max = tau_min, tau_max
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        g = 1.0 / (hidden ** 0.5)
        self.W_tau = torch.randn(hidden, in_dim, generator=gen) * g
        self.W_in = torch.randn(hidden, in_dim, generator=gen) * g
        self.W_h = torch.randn(hidden, hidden, generator=gen) * 0.3 * g  # 小 recurrent
        self.b = torch.zeros(hidden)
        self.h = torch.zeros(hidden)

    def reset(self):
        self.h = torch.zeros_like(self.h)

    def forward(self, x):
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.W_tau @ x)
        I = self.W_in @ x + self.W_h @ self.h + self.b
        dh = (self.dt / tau) * (-self.h + I)
        h = self.h + dh
        h = self.v_max * torch.tanh(h / self.v_max)  # 软饱和有界（P-PHY-3）
        self.h = h
        return h
