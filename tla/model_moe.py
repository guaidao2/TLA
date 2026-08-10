"""TLAMoEModel：无捷径 + 任务分离专家（MoE）的组合体装配。

接口与 TLAModel 对齐（train_step / infer / reset / ltc / pcn / self_slot / replay），
供 criteria 实验直接复用。MoE v1 暂不接容量管理（专家本身就是容量分离）。
"""
import torch
from tla.substrate.ltc_cell import LTCCell
from tla.substrate.energy import EnergyBudget
from tla.cognitive.pcn_moe import MoEPCNStack
from tla.cognitive.self_slot import SelfSlot
from tla.cognitive.memory import Scratchpad
from tla.learning.cls_replay import ReplayBuffer


class TLAMoEModel:
    def __init__(self, obs_dim=3, out_dim=2, ltc_hidden=16, hidden=24, top=16,
                 n_experts=2, seed=0, lr=0.01, lr_inf=0.1, settle_steps=4,
                 infer_max_steps=8, infer_tol=0.02, tol_rel=0.5,
                 tol_progress=0.05, tol_out=0.005, energy_capacity=20.0):
        self.obs_dim, self.out_dim = obs_dim, out_dim
        self.ltc = LTCCell(in_dim=obs_dim, hidden=ltc_hidden, seed=seed)
        self.pcn = MoEPCNStack(dims=[obs_dim + ltc_hidden, hidden, top],
                               out_dim=out_dim, n_experts=n_experts,
                               lr_inf=lr_inf, seed=seed + 1)
        self.self_slot = SelfSlot(in_dim=obs_dim + ltc_hidden, out_dim=out_dim,
                                  seed=seed + 2)
        self.scratchpad = Scratchpad()
        self.energy = EnergyBudget(capacity=energy_capacity)
        self.replay = ReplayBuffer(seed=seed + 3)
        self.lr = lr
        self.settle_steps = settle_steps
        self.infer_max_steps = infer_max_steps
        self.infer_tol = infer_tol
        self.tol_rel = tol_rel
        self.tol_progress = tol_progress
        self.tol_out = tol_out

    def reset(self):
        self.ltc.reset()
        self.energy.reset()
        self.pcn.reset()

    # ---- 训练 ----
    def _core_step(self, s_t, s_next):
        """核心学习步（不 push、不重放）——训练与重放共用。"""
        h = self.ltc.forward(s_t)
        x = torch.cat([s_t, h])
        target = s_next[: self.out_dim]
        mse = self.pcn.learn_step(x, target, lr=self.lr, settle_steps=self.settle_steps)
        p_out = self.pcn.readout().detach()
        self_loss = self.self_slot.learn(x, p_out)
        self.scratchpad.write(self.pcn.last_max_err)
        return mse, self_loss

    def train_step(self, s_t, s_next):
        h_in = self.ltc.h.clone()
        mse, self_loss = self._core_step(s_t, s_next)
        self.replay.push(s_t, s_next, h_in, float(mse) if mse is not None else 1.0)
        self.replay.maybe_replay(self)
        return mse, self_loss

    def replay_step(self, s_t, s_next, h_ctx):
        """CLS 重放：还原样本当时的身体上下文（与 TLAModel 语义一致），不 push/不重放。"""
        self.ltc.h = h_ctx.clone()
        return self._core_step(s_t, s_next)

    # ---- 推理（会琢磨：误差/输出收敛/停滞即停 + doubtful + 预算）----
    def infer(self, obs, reset_energy=True, max_steps=None):
        if reset_energy:
            self.energy.reset()
        h = self.ltc.forward(obs)
        x = torch.cat([obs, h])
        pcn = self.pcn
        budget = self.infer_max_steps if max_steps is None else max_steps
        steps, max_err, prev_err = 0, 0.0, float("inf")
        prev_pred, err_first = None, None
        doubtful = False
        if budget <= 0:
            pred = pcn.readout().detach()
            return pred, dict(steps=0, max_err=0.0, doubtful=False,
                              suppressed=False, confidence=0.0)
        for k in range(1, budget + 1):
            max_err = pcn.settle_step(x, target=None)
            pred = pcn.readout().detach()
            steps = k
            if err_first is None:
                err_first = max_err
            out_change = float(torch.norm(pred - prev_pred).item()) if prev_pred is not None else float("inf")
            prev_pred = pred
            if not self.energy.consume(n_active=pcn.n_experts + 1):
                doubtful = True
                break
            tol_eff = max(self.infer_tol, self.tol_rel * err_first)
            converged = (max_err < tol_eff
                         or out_change < self.tol_out
                         or (prev_err - max_err < self.tol_progress * prev_err))
            prev_err = max_err
            if converged:
                break
        tol_final = max(self.infer_tol, (self.tol_rel * err_first) if err_first else 0)
        if steps >= budget and max_err >= tol_final:
            doubtful = True
        pred = pcn.readout().detach()
        confidence = float(min(max(1.0 - max_err / max(err_first, 1e-9), 0.0), 1.0))
        if confidence < 0.35:
            doubtful = True
        return pred, dict(steps=steps, max_err=max_err, doubtful=doubtful,
                          suppressed=False, confidence=confidence)
