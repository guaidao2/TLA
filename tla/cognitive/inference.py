"""推理环（会琢磨，⑪）：误差小即停 + 步数上限 + 预算耗尽 doubtful（不硬猜）。

- 停止条件 = 误差阈值 + 步数上限 + 能量预算；
- 琢磨不透（预算耗尽仍不收敛）→ doubtful 标记 + 置信度；
- 能量耗尽 → suppressed（物理崩=全崩语义，②）；
- Self_Slot 一致性门：自预测与当前输出分歧大 → 不提前停（⑦）。
"""
import torch


class InferenceLoop:
    def __init__(self, max_steps=8, tol=0.02, tol_rel=0.5, tol_progress=0.05,
                 tol_out=0.005, doubtful_conf=0.35):
        self.max_steps = max_steps
        self.tol = tol
        self.tol_rel = tol_rel       # 相对停止：误差降到初始的 tol_rel 以下即停
        self.tol_progress = tol_progress  # 停滞停止：单步改进 < 5% 即停（DEQ 式）
        self.tol_out = tol_out       # 输出收敛停止：预测不再变化即停（琢磨收益≈0）
        self.doubtful_conf = doubtful_conf  # 低置信度 → 琢磨不透标记（P-COG-4 校准）

    def run(self, model, x, energy, self_consistency_gate=False, max_steps=None):
        pcn = model.pcn
        pcn.mus[0] = x
        budget = self.max_steps if max_steps is None else max_steps
        steps, max_err, prev_err = 0, 0.0, float("inf")
        prev_pred, err_first = None, None
        doubtful, suppressed = False, False
        if budget <= 0:                       # 纯初猜（关掉推理环，P-COG-3/5 消融基线）
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
            if not energy.consume(n_active=len(pcn.mus) - 1):
                suppressed, doubtful = True, True      # 断电（②）
                break
            tol_eff = max(self.tol, self.tol_rel * err_first)
            converged = (max_err < tol_eff
                         or out_change < self.tol_out
                         or (prev_err - max_err < self.tol_progress * prev_err))
            if self_consistency_gate and not converged:
                self_pred = model.self_slot.predict(x)
                divergence = float(torch.norm(self_pred - pred).item())
                if divergence > 0.2:                  # 自预测分歧大（OOD/异常）→ 继续琢磨
                    prev_err = max_err
                    continue
            prev_err = max_err
            if converged:
                break
        tol_final = max(self.tol, (self.tol_rel * err_first) if err_first else 0)
        if steps >= budget and max_err >= tol_final:
            doubtful = True                            # 步数预算耗尽仍不收敛 → 标记不确定
        pred = pcn.readout().detach() if not suppressed else None
        # 置信度 = 误差消除率（settle 把初始误差消掉多少；0=没琢磨动，1=全消掉）
        confidence = 0.0 if suppressed else float(min(max(1.0 - max_err / max(err_first, 1e-9), 0.0), 1.0))
        if not suppressed and confidence < self.doubtful_conf:
            doubtful = True                            # 低置信度 = 琢磨不透（P-COG-4 校准信号）
        return pred, dict(steps=steps, max_err=max_err, doubtful=doubtful,
                          suppressed=suppressed, confidence=confidence)
