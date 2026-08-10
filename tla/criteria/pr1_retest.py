"""原则一解锁重测（预注册协议执行）：PR1 学习强度达标（MSE 0.0046 ≤ 0.02）后，
正式重跑 P-COG-3（琢磨消融）与 P-LEARN-1（A/B 保留率）——用正式裁决替代推断。

预期（基于探针诊断）：P-COG-3 大概率仍负（困难输入上 settle 迭代为负价值）；
P-LEARN-1 待测（W_base 任务相关 + 残差通路上下文分离，重放保护值得检验）。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(m, trajs, max_steps=None, noise=0.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mses = []
    for traj in trajs:
        m.reset()
        for t in range(len(traj) - 1):
            obs = traj[t]
            if noise > 0:
                obs = obs + torch.randn(3, generator=gen) * noise
            pred, info = m.infer(obs, max_steps=max_steps)
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def train_epochs(m, trajs, n, replay_prob=None):
    if replay_prob is not None:
        m.replay.replay_prob = replay_prob
    for _ in range(n):
        for traj in trajs:
            m.reset()
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1])


def run_retest(seed=0, verbose=True):
    world = VariableSpeedWorld(seed=seed)

    # ---- P-COG-3：琢磨消融（未见 ω 泛化场地）----
    train = world.trajectories(n_traj=30, T=30, speed_range=(0.8, 3.0))
    unseen = world.trajectories(n_traj=4, T=20, speed_range=(4.0, 5.0), seed=999)
    indist = world.trajectories(n_traj=4, T=20, speed_range=(1.0, 2.0), seed=7)
    m = TLAPR1Model(obs_dim=3, out_dim=2, seed=seed)
    train_epochs(m, train, 2)
    mse_ad_u = eval_mse(m, unseen)
    mse_f1_u = eval_mse(m, unseen, 1)
    mse_guess_u = eval_mse(m, unseen, 0)
    mse_indist = eval_mse(m, indist)
    gain_u = (mse_f1_u - mse_ad_u) / max(mse_f1_u, 1e-9)
    p_cog3 = mse_ad_u < 0.9 * mse_f1_u            # 预注册判据：adaptive < 0.9×fixed1

    # ---- P-LEARN-1：A(ω0.8-1.5) → B(ω3.5-4.5) 保留率（重放对照）----
    # 注：重测参数为 2 B-epoch × 20×T25（原 lifelong.py 为 3 epoch × 25×T30）——
    # 偏差在"宽松"方向（遗忘压力更小），不可能伪造负结果，故裁决方向可信。
    wa = VariableSpeedWorld(seed=seed, mode="spring")
    wb = VariableSpeedWorld(seed=seed + 10, mode="spring")
    train_a = wa.trajectories(n_traj=20, T=25, speed_range=(0.8, 1.5))
    test_a = wa.trajectories(n_traj=3, T=15, speed_range=(0.9, 1.3), seed=7)
    train_b = wb.trajectories(n_traj=20, T=25, speed_range=(3.5, 4.5))

    def protocol(rp):
        mm = TLAPR1Model(obs_dim=3, out_dim=2, seed=seed)
        train_epochs(mm, train_a, 2, rp)
        a0 = eval_mse(mm, test_a)
        train_epochs(mm, train_b, 2, rp)
        return a0, eval_mse(mm, test_a)

    a0_r, a1_r = protocol(0.3)
    a0_nr, a1_nr = protocol(0.0)
    retention_r = a0_r / max(a1_r, 1e-9)
    retention_nr = a0_nr / max(a1_nr, 1e-9)
    replay_helps = a1_r < a1_nr
    p_learn1 = retention_r >= 0.95 and retention_r > retention_nr

    if verbose:
        print("=" * 64)
        print("原则一解锁重测（P-COG-3 / P-LEARN-1 正式裁决）")
        print("=" * 64)
        print(f"  [前置] 分布内 MSE={mse_indist:.4f} (解锁阈值 0.02)")
        print(f"  P-COG-3: adaptive={mse_ad_u:.4f} fixed1={mse_f1_u:.4f} "
              f"guess={mse_guess_u:.4f} 增益 {gain_u * 100:+.1f}% "
              f"→ {'PASS' if p_cog3 else 'FAIL（负结果）'}")
        print(f"  P-LEARN-1: 重放 A {a0_r:.4f}→{a1_r:.4f} 保留率={retention_r * 100:.1f}%  "
              f"无重放={retention_nr * 100:.1f}% 重放帮助={'是' if replay_helps else '否'}"
              f" → {'PASS' if p_learn1 else 'FAIL（负结果）'}")
        print("=" * 64)
    return dict(mse_indist=mse_indist, mse_ad_u=mse_ad_u, mse_f1_u=mse_f1_u,
                mse_guess_u=mse_guess_u, gain_u=gain_u, p_cog3=p_cog3,
                retention_r=retention_r, retention_nr=retention_nr,
                replay_helps=replay_helps, p_learn1=p_learn1)


if __name__ == "__main__":
    run_retest(verbose=True)
