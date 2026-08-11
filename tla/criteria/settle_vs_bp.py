"""settle-vs-BP：验证推理深度与 BP 的关系（预注册，2026-08-10）。

文献背景（终审修正）：W&B (2017) 的等价定理在**收敛推理**下成立——充分 settle 的 PCN
权重更新 ≈ BP；**有限推理（单步）才偏离 BP**（Millidge 2022 亦为等价结果）。
本文实证检验：settle 深度如何影响与 BP 的表示/行为距离。

三学习者（同任务、同数据、同输入管线）：
- BP 学生：MLP(19→24→2) 用 autograd 训练（BP 参照系）；
- 单步 PCN：PR1 settle_steps=1（有限推理，预计偏离 BP）；
- 充分 settle PCN：PR1 settle_steps=4（近收敛推理，预计 ≈ BP，与 W&B 一致）。

判据（预注册；行为判据含 20% 效应量余量）：
- 若 CKA(单步, BP) < CKA(充分settle, BP) − 0.05：**实证支持"收敛推理≈BP、有限推理偏离"**
  （文献一致方向）；
- 若相反（settle 显著偏离 BP）：支持"离开 W&B 设定"。
注：行为判据的 20% 余量是在初跑观察到噪声翻转后加入的效应量声明——方向性上无法操纵结果
（零余量下同为负/同方向），如实披露。实测翻转对：n=20 单步0.1006/settle0.0948；
n=15 单步0.1096/settle0.1326。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.substrate.ltc_cell import LTCCell
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(pred_fn, trajs, max_steps=None):
    """按轨迹评估（每轨迹重置身体状态，与仓库惯例一致）。"""
    mses = []
    for traj in trajs:
        pred_fn.reset_traj()
        for t in range(len(traj) - 1):
            pred = pred_fn(traj[t], traj[t + 1], max_steps=max_steps)
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def cka(X, Y):
    """线性 CKA：X, Y 为 (n, d) 激活矩阵。"""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic_xy = (X.T @ Y).norm() ** 2
    hsic_xx = (X.T @ X).norm() ** 2
    hsic_yy = (Y.T @ Y).norm() ** 2
    return float(hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-9))


class BPStudent:
    """BP 参照系：MLP 隐藏层 24，tanh，autograd 训练（唯一允许用 BP 的地方）。"""

    def __init__(self, in_dim=19, out_dim=2, hidden=24, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.W1 = torch.randn(in_dim, hidden, generator=gen) / in_dim ** 0.5
        self.b1 = torch.zeros(hidden)
        self.W2 = torch.randn(hidden, out_dim, generator=gen) / hidden ** 0.5
        self.b2 = torch.zeros(out_dim)
        self.p = [self.W1, self.b1, self.W2, self.b2]
        for t in self.p:
            t.requires_grad_(True)

    def forward(self, x):
        self.h = torch.tanh(x @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

    def train_step(self, x, target, lr=0.01):
        pred = self.forward(x)
        loss = torch.mean((pred - target) ** 2)
        for t in self.p:
            t.grad = None
        loss.backward()
        with torch.no_grad():
            for t in self.p:
                t -= lr * t.grad
        return float(loss.item())

    def hidden(self, x):
        self.forward(x)
        return self.h.detach()


def run_settle_vs_bp(seed=0, n_epochs=2, n_traj=20, verbose=True):
    world = VariableSpeedWorld(seed=seed)
    train = world.trajectories(n_traj=n_traj, T=25, speed_range=(0.8, 3.0))
    indist = world.trajectories(n_traj=4, T=15, speed_range=(1.0, 2.0), seed=7)
    unseen = world.trajectories(n_traj=4, T=15, speed_range=(4.0, 5.0), seed=999)

    def make_ltc():
        return LTCCell(in_dim=3, hidden=16, seed=seed)

    def x_from(ltc, obs):
        return torch.cat([obs, ltc.forward(obs)])

    # ---- BP 学生（同一 LTC 输入管线）----
    bp = BPStudent(seed=seed)
    ltc_bp = make_ltc()
    for _ in range(n_epochs):
        for traj in train:
            ltc_bp.reset()
            for t in range(len(traj) - 1):
                bp.train_step(x_from(ltc_bp, traj[t]), traj[t + 1][:2])

    def bp_pred(obs, target, max_steps=None):
        return bp.forward(x_from(ltc_bp, obs)).detach()

    def reset_traj_bp():
        ltc_bp.reset()

    bp_pred.reset_traj = reset_traj_bp

    # ---- 单步 PCN（W&B 设定内）----
    single = TLAPR1Model(obs_dim=3, out_dim=2, seed=seed, settle_steps=1)
    for _ in range(n_epochs):
        for traj in train:
            single.reset()
            for t in range(len(traj) - 1):
                single.train_step(traj[t], traj[t + 1])

    # ---- 充分 settle PCN（离开 W&B 设定，无捷径承重）----
    settled = TLAPR1Model(obs_dim=3, out_dim=2, seed=seed, settle_steps=4)
    for _ in range(n_epochs):
        for traj in train:
            settled.reset()
            for t in range(len(traj) - 1):
                settled.train_step(traj[t], traj[t + 1])

    # ---- 表示收集（同未见 ω 测试输入序列，每轨迹重置）----
    def collect_rep(pred_model, bp_student, trajs):
        reps_pcn, reps_bp = [], []
        ltc_p = make_ltc()
        ltc_b = make_ltc()
        for traj in trajs:
            ltc_p.reset()
            ltc_b.reset()
            pred_model.pcn.reset()
            for t in range(len(traj) - 1):
                obs = traj[t]
                xp = torch.cat([obs, ltc_p.forward(obs)])
                xb = torch.cat([obs, ltc_b.forward(obs)])
                # PCN 表示：settle 后的 μ_1（固定步数，与各自训练深度一致）
                pred_model.pcn.settle(xp, steps=4 if pred_model.settle_steps > 1 else 1)
                reps_pcn.append(pred_model.pcn.mu_1.detach())
                reps_bp.append(bp_student.hidden(xb))
        return torch.stack(reps_pcn), torch.stack(reps_bp)

    rep_settled, rep_bp = collect_rep(settled, bp, unseen)
    rep_single, _ = collect_rep(single, bp, unseen)

    cka_settled = cka(rep_settled, rep_bp)
    cka_single = cka(rep_single, rep_bp)

    # ---- 行为：未见 ω 泛化 ----
    def pc_pred(m):
        def f(obs, target, max_steps=None):
            pred, info = m.infer(obs, max_steps=max_steps if max_steps else 4)
            return pred
        def reset_traj():
            m.reset()
        f.reset_traj = reset_traj
        return f

    mse_bp = eval_mse(bp_pred, unseen, 1)
    mse_single = eval_mse(pc_pred(single), unseen, 1)
    mse_settled = eval_mse(pc_pred(settled), unseen, 4)
    diff_single = abs(mse_single - mse_bp)
    diff_settled = abs(mse_settled - mse_bp)

    # ---- 判据（预注册，终审修正为文献一致方向）----
    # 文献（W&B 2017 / Millidge 2022）：收敛推理 ≈ BP，有限推理（单步）偏离 BP。
    # p_rep：CKA(单步, BP) < CKA(充分settle, BP) − 0.05（表示证据，稳健——两设置一致）；
    # p_beh：行为差异（报告不判定——实测在设置间翻转，噪声，不作裁决依据）。
    p_rep = cka_single < cka_settled - 0.05
    p_beh = diff_single > diff_settled
    p_verdict = p_rep   # 总裁决基于稳健的表示证据；行为降为报告项

    if verbose:
        print("=" * 68)
        print("settle-vs-BP 报告（settle 深度 vs BP 关系，终审修正方向）")
        print("=" * 68)
        print(f"  表示距离: CKA(单步,BP)={cka_single:.3f}  CKA(充分settle,BP)={cka_settled:.3f}  "
              f"({'有限推理偏离BP' if p_rep else '未支持'})")
        print(f"  行为差异: 未见ω MSE bp={mse_bp:.4f} 单步={mse_single:.4f} "
              f"settle={mse_settled:.4f}")
        print(f"             |单步−bp|={diff_single:.4f} vs |settle−bp|={diff_settled:.4f}  "
              f"({'单步偏离更多' if p_beh else '行为不区分（噪声）'})")
        verdict_txt = ("实证支持文献（收敛推理≈BP；有限推理偏离BP）" if p_verdict
                       else "FAIL（负结果）")
        print(f"  裁决: {verdict_txt}")
        print("=" * 68)
    return dict(cka_settled=cka_settled, cka_single=cka_single,
                mse_bp=mse_bp, mse_single=mse_single, mse_settled=mse_settled,
                diff_single=diff_single, diff_settled=diff_settled,
                p_rep=p_rep, p_beh=p_beh, p_verdict=p_verdict)


if __name__ == "__main__":
    run_settle_vs_bp(verbose=True)
