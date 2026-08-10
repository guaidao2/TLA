"""推理环（会琢磨，⑪）：误差小即停 + 步数上限 + 预算耗尽 doubtful（不硬猜）。

- 停止条件 = 误差阈值 + 步数上限 + 能量预算；
- 琢磨不透（预算耗尽仍不收敛）→ doubtful 标记 + 置信度；
- 能量耗尽 → suppressed（物理崩=全崩语义，②）；
- Self_Slot 一致性门：自预测与当前输出分歧大 → 不提前停（⑦）。
"""
import torch


class InferenceLoop:
    def __init__(self, max_steps=8, tol=0.02, tol_rel=0.5, tol_progress=0.05,
                 tol_out=0.005):
        self.max_steps = max_steps
        self.tol = tol
        self.tol_rel = tol_rel       # 相对停止：误差降到初始的 tol_rel 以下即停
        self.tol_progress = tol_progress  # 停滞停止：单步改进 < 5% 即停（DEQ 式）
        self.tol_out = tol_out       # 输出收敛停止：预测不再变化即停（琢磨收益≈0）

    def run(self, model, x, energy, self_consistency_gate=False):
        pcn = model.pcn
        pcn.mus[0] = x
        steps, max_err, prev_err = 0, 0.0, float("inf")
        prev_pred, err_first = None, None
        doubtful, suppressed = False, False
        for k in range(1, self.max_steps + 1):
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
        if steps >= self.max_steps and max_err >= tol_final:
            doubtful = True                            # 步数预算耗尽仍不收敛 → 标记不确定
        pred = pcn.readout().detach() if not suppressed else None
        confidence = 0.0 if suppressed else (1.0 - min(max_err / max(tol_final, 1e-9), 1.0))
        return pred, dict(steps=steps, max_err=max_err, doubtful=doubtful,
                          suppressed=suppressed, confidence=confidence)
