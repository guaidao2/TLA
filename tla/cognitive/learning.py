"""训练环（会学习，③④）：两阶段——free phase settle → 局部误差驱动权重更新（无 BP）。

组合体装配：LTC 身体（⑤）→ PCN 层叠（③④）→ Self_Slot 自监督（⑦）→ 元层生长/修剪（⑨）。
Self_Slot 与元层统计只在 meta_update=True 时更新（CLS 重放时 meta_update=False，防污染，同 NLA）。
"""
import torch


class ErrorDrivenTrainer:
    def __init__(self, model, lr=0.01, lr_inf=0.1, settle_steps=4):
        self.model = model
        self.lr = lr
        self.lr_inf = lr_inf
        self.settle_steps = settle_steps
        self.ema_err = 0.3              # 惊奇度 EMA（生长门 novelty 信号）
        self.ema_novelty = 0.3

    def step(self, s_t, s_next, meta_update=True):
        model = self.model
        h = model.ltc.forward(s_t)                      # 身体：时间演化（⑤）
        x = torch.cat([s_t, h])
        target = s_next[: model.pcn.out_dim]            # readout 只预测状态维（dt 是输入不是目标）
        mse = model.pcn.learn_step(x, target, lr=self.lr, settle_steps=self.settle_steps)
        p_out = model.pcn.readout().detach()

        # Self_Slot（⑦）：从完整输入状态预测"自己会输出什么"（自监督，无外部标签）
        self_loss = model.self_slot.learn(x, p_out)

        # 工作记忆槽：写本次推理误差痕迹（供"主动采样"读取）
        model.scratchpad.write(model.pcn.last_max_err)

        # 元层统计（重放时跳过）
        if meta_update:
            err = float(torch.mean((p_out - s_next[: model.pcn.out_dim]) ** 2).item())
            novelty = float(torch.mean(torch.abs(p_out - s_next[: model.pcn.out_dim])).item())
            self.ema_err = 0.95 * self.ema_err + 0.05 * err
            self.ema_novelty = 0.95 * self.ema_novelty + 0.05 * novelty
            top = model.pcn.mus[model.pcn.L].detach()
            e_out = (p_out - s_next[: model.pcn.out_dim]).detach()
            contrib = torch.abs(model.pcn.W_out.T @ e_out)   # 每顶层单元的误差贡献（Wᵀ 投影）
            model.meta.update(activations=top, error_contrib=contrib,
                              avg_error=self.ema_err, energy_level=model.energy.report().integrity)
            model.meta.maybe_prune()
            model.meta.maybe_grow(self.ema_err, self.ema_novelty,
                                  model.energy.report().integrity)
        return mse, self_loss

    def replay_step(self, s_t, s_next):
        """CLS 重放：同学习步但 meta_update=False（不污染生长统计）。"""
        return self.step(s_t, s_next, meta_update=False)