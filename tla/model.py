"""组合体装配（TLA 单一网络）：LTC 身体 + PCN 层叠 + Self_Slot + 工作记忆 + 能量 + 容量管理 + CLS。

一次 train_step 的完整数据流（对应 v0.2 §3）：
  obs_t → LTC（⑤ 时间演化）→ x=[obs,h] → PCN settle+局部更新（③④）
  → readout p_out（预测 obs_{t+1}）→ Self_Slot 自监督（⑦）→ 元层统计（⑨）→ CLS 入队。
"""
import torch
from tla.substrate.ltc_cell import LTCCell
from tla.substrate.energy import EnergyBudget
from tla.cognitive.pcn_stack import PCNStack
from tla.cognitive.inference import InferenceLoop
from tla.cognitive.learning import ErrorDrivenTrainer
from tla.cognitive.self_slot import SelfSlot
from tla.cognitive.memory import Scratchpad
from tla.meta.growth import CapacityManager
from tla.learning.cls_replay import ReplayBuffer


class TLAModel:
    def __init__(self, obs_dim, out_dim=2, ltc_hidden=16, hidden_dims=(24, 24),
                 seed=0, lr=0.01, lr_inf=0.1, settle_steps=4,
                 infer_max_steps=8, infer_tol=0.02, energy_capacity=20.0,
                 use_lin_shortcut=True):
        self.obs_dim, self.out_dim = obs_dim, out_dim
        self.ltc = LTCCell(in_dim=obs_dim, hidden=ltc_hidden, seed=seed)
        self.pcn = PCNStack(dims=[obs_dim + ltc_hidden, *hidden_dims],
                            out_dim=out_dim, lr_inf=lr_inf,
                            use_lin_shortcut=use_lin_shortcut, seed=seed + 1)
        self.self_slot = SelfSlot(in_dim=obs_dim + ltc_hidden, out_dim=out_dim,
                                  seed=seed + 2)
        self.scratchpad = Scratchpad()
        self.energy = EnergyBudget(capacity=energy_capacity)
        self.meta = CapacityManager(n_units=hidden_dims[-1])
        self.pcn.set_gate(self.meta.gate_vector)
        self.trainer = ErrorDrivenTrainer(self, lr=lr, lr_inf=lr_inf,
                                          settle_steps=settle_steps)
        self.replay = ReplayBuffer(seed=seed + 3)
        self.infer_loop = InferenceLoop(max_steps=infer_max_steps, tol=infer_tol)

    def reset(self):
        self.ltc.reset()
        self.energy.reset()
        self.pcn.reset_mus()

    # ---- 训练 ----
    def train_step(self, s_t, s_next):
        mse, self_loss, h_in = self.trainer.step(s_t, s_next)
        surprise = self.trainer.ema_err
        self.replay.push(s_t, s_next, h_in, surprise)
        self.replay.maybe_replay(self.trainer)
        return mse, self_loss

    # ---- 推理（会琢磨：误差小即停 + 预算 + doubtful）----
    def infer(self, obs, reset_energy=True, max_steps=None, self_consistency_gate=None):
        if reset_energy:
            self.energy.reset()
        h = self.ltc.forward(obs)
        x = torch.cat([obs, h])
        gate = True if self_consistency_gate is None else self_consistency_gate
        return self.infer_loop.run(self, x, self.energy, self_consistency_gate=gate,
                                   max_steps=max_steps)
