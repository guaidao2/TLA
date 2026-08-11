"""奇点神经元（Singularity Cell）——宇宙演化映射的时间细胞。

数学推导见 docs/奇点神经元_数学推导.md（v0.1，判据先于实现）。

核心 ODE：
    dh/dt = α·e^{βI}·(h+ε)·(1 − h/h_max) − λ·h^γ + Λ
    - ε 奇点涨落：h=0 时增长项 = αe^{βI}ε > 0（能炸，修复原版乘性死亡）；
    - (1 − h/h_max) 饱和：增长项在 h_max 处归零（有界）；
    - λ·h^γ 红移衰减（遗忘）；Λ 暗能量（幽灵态保底）。

离散化：显式欧拉（默认，dt=1 tick）；γ=1 时可选半隐式
    h_{t+1} = (h + dt·αe^{βI}ε + dt·Λ) / (1 + dt·αe^{βI} + dt·λ)
    （线性化"无条件稳定"——饱和项 (1−h/h_max) 按常数处理，诚实标注）。

input_gated=True 为修复变体（推导 v0.1 的相容性问题，见 criteria/singularity.py）：
    增长项改为 α·(e^{βI}−1)·(h+ε)·(1−h/h_max)——I=0 时增长归零，幽灵态 h*=(Λ/λ)^{1/γ}
    成为真正的全局吸引子（原文 e^{βI} 在 I=0 时仍有基线生长，细胞恒热 ~0.902）。

待定（推导 v0.1 不一致点）：自调制 τ = τ_min + (τ_max−τ_min)e^{−k·h} 已推导但尚未进入
动力学（仅作诊断属性输出），留待耦合层间时间尺度实验。
"""
import math


class SingularityCell:
    def __init__(self, alpha=0.5, beta=1.0, eps=1e-4, h_max=1.0,
                 lam=0.05, gamma=1.0, lam_dark=1e-3, k=1.0,
                 tau_min=0.1, tau_max=1.0, dt=1.0, input_gated=False):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.h_max = h_max
        self.lam = lam
        self.gamma = gamma
        self.lam_dark = lam_dark
        self.k = k
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dt = dt
        self.input_gated = input_gated
        self.h = 0.0
        self.t = 0

    # ---- 语义量 ----
    def ghost(self):
        """暗能量不动点 h* = (Λ/λ)^{1/γ}（幽灵态高度）。"""
        return (self.lam_dark / self.lam) ** (1.0 / self.gamma)

    def tau(self):
        """自调制时间常数（诊断：尚未进入动力学，推导 v0.1 不一致点）。
        h 大（刚炸完/已凝固）→ τ 大 → 慢演化长记忆；h 小（奇点）→ τ 小 → 快响应。"""
        return self.tau_min + (self.tau_max - self.tau_min) * math.exp(-self.k * self.h)

    def i_th(self, h=None):
        """暴胀阈值 I_th(h)：增长项 > 衰减项 的临界输入（§6.2）。"""
        h = self.h if h is None else h
        arg = self.lam * h ** self.gamma / (self.alpha * self.eps)
        if self.input_gated:
            arg = arg + 1.0
        return (1.0 / self.beta) * math.log(arg)

    # ---- 动力学 ----
    def growth(self, I):
        base = self.alpha * (math.exp(self.beta * I) - 1.0) if self.input_gated \
            else self.alpha * math.exp(self.beta * I)
        return base * (self.h + self.eps) * (1.0 - self.h / self.h_max)

    def decay(self):
        return self.lam * self.h ** self.gamma

    def dhdt(self, I):
        return self.growth(I) - self.decay() + self.lam_dark

    def step(self, I):
        """显式欧拉（dt=1 tick 默认）。数值安全 clip 到 [0, h_max]（连续 ODE 的不变集，
        离散化用 clip 防欧拉越界——SN-5 数值验证的诚实标注）。"""
        self.h = min(max(self.h + self.dt * self.dhdt(I), 0.0), self.h_max)
        self.t += 1
        return self.h

    def step_semi_implicit(self, I):
        """γ=1 半隐式（线性化稳定；饱和项按常数——诚实标注）。"""
        assert self.gamma == 1.0, "半隐式仅定义于 γ=1"
        eI = math.exp(self.beta * I)
        grow = self.alpha * (eI - 1.0) if self.input_gated else self.alpha * eI
        num = self.h + self.dt * grow * self.eps + self.dt * self.lam_dark
        den = 1.0 + self.dt * grow + self.dt * self.lam
        self.h = min(max(num / den, 0.0), self.h_max)
        self.t += 1
        return self.h

    def reset(self, h=0.0):
        self.h = float(h) if isinstance(h, (int, float)) else h
        self.t = 0
