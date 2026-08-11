"""settle-vs-BP：验证"充分 settle 后局部学习是否离开 W&B 设定"（预注册，2026-08-10）。

背景：W&B (2017) 证明单步更新 PCN 的局部学习 ≈ BP；TLA 的推理环是迭代 settle。
本实验检验开放问题：**充分 settle 后，局部学习还等价 BP 吗？**

三学习者（同任务、同数据、同输入管线）：
- BP 学生：MLP(19→24→2) 用 autograd 训练（BP 参照系）；
- 单步 PCN：PR1 settle_steps=1（仍在 W&B 设定内，"不琢磨"）；
- 充分 settle PCN：PR1 settle_steps=4（离开 W&B 设定，"会琢磨"，无捷径承重）。

判据（预注册，行为判据含 20% 效应量余量防噪声翻转）：
- CKA(充分settle, BP) < CKA(单步, BP) − 0.05：充分 settle 的表示比单步更偏离 BP；
- 行为差异（需 ≥20% 才判偏离）：diff_settled > diff_single × 1.2（未见 ω）。
两个判据至少一个成立 → "离开 W&B 设定"实证支持；都不成立 → 负结果如实记录。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.substrate.ltc_cell import LTCCell
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(pred_fn, trajs, max_steps=None):
    mses = []
    for traj in trajs:
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
        ltc_bp.reset()
        return bp.forward(x_from(ltc_bp, obs)).detach()

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

    # ---- 表示收集（同未见 ω 测试输入序列）----
    def collect_rep(pred_model, bp_student, trajs):
        reps_pcn, reps_bp = [], []
        ltc_p = make_ltc()
        ltc_b = make_ltc()
        for traj in trajs:
            ltc_p.reset()
            ltc_b.reset()
            for t in range(len(traj) - 1):
                obs = traj[t]
                xp = torch.cat([obs, ltc_p.forward(obs)])
                xb = torch.cat([obs, ltc_b.forward(obs)])
                # PCN 表示：settle 后的 μ_1（固定步数）
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
        return f

    mse_bp = eval_mse(bp_pred, unseen, 1)
    mse_single = eval_mse(pc_pred(single), unseen, 1)
    mse_settled = eval_mse(pc_pred(settled), unseen, 4)
    diff_single = abs(mse_single - mse_bp)
    diff_settled = abs(mse_settled - mse_bp)

    # ---- 判据（预注册；行为判据需 ≥20% 效应量，防噪声翻转）----
    p_rep = cka_settled < cka_single - 0.05
    p_beh = diff_settled > diff_single * 1.2
    p_verdict = p_rep or p_beh

    if verbose:
        print("=" * 68)
        print("settle-vs-BP 报告（充分 settle 是否离开 W&B 设定）")
        print("=" * 68)
        print(f"  表示距离: CKA(单步,BP)={cka_single:.3f}  CKA(充分settle,BP)={cka_settled:.3f}  "
              f"({'偏离' if p_rep else '未偏离'})")
        print(f"  行为差异: 未见ω MSE bp={mse_bp:.4f} 单步={mse_single:.4f} "
              f"settle={mse_settled:.4f}")
        print(f"             |settle−bp|={diff_settled:.4f} vs |单步−bp|={diff_single:.4f}  "
              f"({'偏离' if p_beh else '未偏离'})")
        print(f"  裁决: {'PASS（实证离开 W&B 设定）' if p_verdict else 'FAIL（负结果，如实记录）'}")
        print("=" * 68)
    return dict(cka_settled=cka_settled, cka_single=cka_single,
                mse_bp=mse_bp, mse_single=mse_single, mse_settled=mse_settled,
                diff_single=diff_single, diff_settled=diff_settled,
                p_rep=p_rep, p_beh=p_beh, p_verdict=p_verdict)


if __name__ == "__main__":
    run_settle_vs_bp(verbose=True)
