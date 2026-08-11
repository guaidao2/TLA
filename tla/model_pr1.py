"""TLAPR1Model：原则一（摊销首猜 + 残差修正）组合体装配。

接口与 TLAModel 对齐（train_step / infer / reset / ltc / pcn / self_slot / replay）。
"""
import torch
from tla.substrate.ltc_cell import LTCCell
from tla.substrate.energy import EnergyBudget
from tla.cognitive.pcn_amortized import AmortizedResidualPCN
from tla.cognitive.self_slot import SelfSlot
from tla.cognitive.memory import Scratchpad
from tla.learning.cls_replay import ReplayBuffer


class TLAPR1Model:
    def __init__(self, obs_dim=3, out_dim=2, ltc_hidden=16, hidden=24,
                 seed=0, lr=0.01, lr_inf=0.1, settle_steps=4,
                 infer_max_steps=8, infer_tol=0.02, tol_rel=0.5,
                 tol_progress=0.05, tol_out=0.005, energy_capacity=20.0,
                 lam=1.0, substrate_cls=None):
        self.obs_dim, self.out_dim = obs_dim, out_dim
        if substrate_cls is None:
            from tla.substrate.ltc_cell import LTCCell
            substrate_cls = LTCCell
        self.ltc = substrate_cls(in_dim=obs_dim, hidden=ltc_hidden, seed=seed)
        self.pcn = AmortizedResidualPCN(dims=[obs_dim + ltc_hidden, hidden],
                                        out_dim=out_dim, lr_inf=lr_inf,
                                        seed=seed + 1)
        self.self_slot = SelfSlot(in_dim=obs_dim + ltc_hidden, out_dim=out_dim,
                                  seed=seed + 2)
        self.scratchpad = Scratchpad()
        self.energy = EnergyBudget(capacity=energy_capacity)
        self.replay = ReplayBuffer(seed=seed + 3)
        self.lr = lr
        self.lam = lam                     # EWC 突触巩固强度（protect 时用）
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
    def _core_step(self, s_t, s_next, freeze_base=False, consolidate=False,
                   protect=False):
        h = self.ltc.forward(s_t)
        x = torch.cat([s_t, h])
        target = s_next[: self.out_dim]
        mse = self.pcn.learn_step(x, target, lr=self.lr, settle_steps=self.settle_steps,
                                  freeze_base=freeze_base, consolidate=consolidate,
                                  protect=protect, lam=self.lam)
        p_out = self.pcn.readout(x).detach()
        self_loss = self.self_slot.learn(x, p_out)
        self.scratchpad.write(self.pcn.last_max_err)
        return mse, self_loss

    def train_step(self, s_t, s_next, freeze_base=False, consolidate=False,
                   protect=False):
        h_in = self.ltc.h.clone()
        mse, self_loss = self._core_step(s_t, s_next, freeze_base=freeze_base,
                                         consolidate=consolidate, protect=protect)
        self.replay.push(s_t, s_next, h_in, float(mse) if mse is not None else 1.0)
        self.replay.maybe_replay(self)
        return mse, self_loss

    def replay_step(self, s_t, s_next, h_ctx):
        """CLS 重放：还原样本当时的身体上下文，不 push/不重放。"""
        self.ltc.h = h_ctx.clone()
        return self._core_step(s_t, s_next)

    # ---- 批训练（mini-batch 局部更新，等价性由 tests/test_batch.py 验证）----
    def train_batch(self, traj, batch_size=8):
        """在一个轨迹上按 batch 累积局部更新（LTC 沿轨迹滚动，PCN 每 batch 应用一次）。"""
        total, n = 0.0, 0
        for start in range(0, len(traj) - 1, batch_size):
            chunk = traj[start:start + batch_size + 1]
            xs, targets = [], []
            for t in range(len(chunk) - 1):
                h = self.ltc.forward(chunk[t])
                xs.append(torch.cat([chunk[t], h]))
                targets.append(chunk[t + 1][: self.out_dim])
            mse = self.pcn.learn_batch(xs, targets, lr=self.lr,
                                       settle_steps=self.settle_steps)
            total += mse * len(xs)
            n += len(xs)
        return total / max(n, 1)

    # ---- 推理（会琢磨：残差迭代精化直接改输出；琢磨失败→回退首猜=瞎猜，双过程系统2→系统1）----
    # fallback 默认 False（opt-in）：保既有判据（pr1/retest/ablation/lifelong/moe）的
    # "自适应"语义不被静默混入回退——回退实验（pr1_fallback.py）显式传入 fallback=True。
    def infer(self, obs, reset_energy=True, max_steps=None, fallback=False,
              fallback_conf=0.35, drift_cap=0.02):
        if reset_energy:
            self.energy.reset()
        h = self.ltc.forward(obs)
        x = torch.cat([obs, h])
        pcn = self.pcn
        budget = self.infer_max_steps if max_steps is None else max_steps
        steps, max_err, prev_err = 0, 0.0, float("inf")
        prev_pred, err_first = None, None
        doubtful = False
        guess = pcn.readout(x).detach()          # 系统1：摊销首猜（瞎猜基线）
        if budget <= 0:
            return guess, dict(steps=0, max_err=0.0, doubtful=False,
                               suppressed=False, confidence=0.0, fell_back=bool(fallback))
        for k in range(1, budget + 1):
            max_err = pcn.settle_step(x, target=None)
            pred = pcn.readout(x).detach()
            steps = k
            if err_first is None:
                err_first = max_err
            out_change = float(torch.norm(pred - prev_pred).item()) if prev_pred is not None else float("inf")
            prev_pred = pred
            if not self.energy.consume(n_active=1):
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
        reasoned = pcn.readout(x).detach()
        confidence = float(min(max(1.0 - max_err / max(err_first, 1e-9), 0.0), 1.0))
        if confidence < 0.35:
            doubtful = True
        # 琢磨失败判定（双过程：系统2失败 → 系统1兜底=瞎猜）：
        #  ① 琢磨没进展（confidence 低，误差没降下来）；
        #  ② 琢磨想歪了（输出偏离首猜过多 = 过度精化）。
        drift = float(torch.norm(reasoned - guess).item())
        fell_back = False
        if fallback and (confidence < fallback_conf or drift > drift_cap):
            pred = guess                                  # 瞎猜（直觉兜底）
            fell_back = True
            doubtful = True                               # 回退=琢磨失败=不确定（含 drift 情形）
        else:
            pred = reasoned
        return pred, dict(steps=steps, max_err=max_err, doubtful=doubtful,
                          suppressed=False, confidence=confidence, fell_back=fell_back)
