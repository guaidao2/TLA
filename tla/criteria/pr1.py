"""原则一（摊销首猜 + 残差修正）三探针（预注册，2026-08-10）。

- 探针 A（学习强度）：分布内 MSE 应显著优于无捷径变体（~0.11）并接近捷径基线（~0.004）——
  即 <0.02（重测解锁阈值）为"强学习"成立；
- 探针 B（琢磨增益）：自适应（残差迭代）vs 固定 1 步，在噪声/未见 ω 输入上的增益
  应显著 >0（残差通路让 settle 直接改输出，琢磨真实）——对比无捷径之前的负增益；
- 探针 C（对偶性）：A ∧ B 同时成立 = 学习强度与琢磨真实不再互斥（跷跷板解开）。
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


def run_pr1(seed=0, n_epochs=2, n_traj=25, verbose=True):
    world = VariableSpeedWorld(seed=seed)
    train = world.trajectories(n_traj=n_traj, T=30, speed_range=(0.8, 3.0))
    indist = world.trajectories(n_traj=4, T=20, speed_range=(1.0, 2.0), seed=7)
    unseen = world.trajectories(n_traj=4, T=20, speed_range=(4.0, 5.0), seed=999)

    m = TLAPR1Model(obs_dim=3, out_dim=2, seed=seed)
    for _ in range(n_epochs):
        for traj in train:
            m.reset()
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1])

    # 探针 A：分布内学习强度
    mse_indist = eval_mse(m, indist)
    # 探针 B：琢磨增益（噪声 / 未见 ω）
    mse_ad_n = eval_mse(m, indist, None, noise=0.3)
    mse_f1_n = eval_mse(m, indist, 1, noise=0.3)
    mse_ad_u = eval_mse(m, unseen, None)
    mse_f1_u = eval_mse(m, unseen, 1)
    gain_n = (mse_f1_n - mse_ad_n) / max(mse_f1_n, 1e-9)
    gain_u = (mse_f1_u - mse_ad_u) / max(mse_f1_u, 1e-9)

    # 基线对照（记录值）：无捷径 ~0.11，捷径 ~0.004
    p_A = mse_indist < 0.02            # 强学习（达重测解锁阈值）
    p_B = gain_n > 0.05 or gain_u > 0.05   # 琢磨增益显著 >0（任一轴）
    p_C = p_A and p_B                  # 对偶性

    if verbose:
        print("=" * 64)
        print("原则一三探针报告（摊销首猜 + 残差修正）")
        print("=" * 64)
        print(f"  A 学习强度: 分布内 MSE={mse_indist:.4f}  (无捷径~0.11 / 捷径~0.004 / 解锁阈值 0.02)")
        print(f"  B 琢磨增益: 噪声 adaptive={mse_ad_n:.4f} vs fixed1={mse_f1_n:.4f} "
              f"(增益 {gain_n * 100:+.1f}%)")
        print(f"             未见ω adaptive={mse_ad_u:.4f} vs fixed1={mse_f1_u:.4f} "
              f"(增益 {gain_u * 100:+.1f}%)")
        print(f"  C 对偶性: 强学习{'成立' if p_A else '失败'} ∧ "
              f"琢磨增益{'成立' if p_B else '失败'} → "
              f"{'对偶解开' if p_C else '仍未解开'}")
        print("=" * 64)
    return dict(mse_indist=mse_indist, mse_ad_n=mse_ad_n, mse_f1_n=mse_f1_n,
                mse_ad_u=mse_ad_u, mse_f1_u=mse_f1_u, gain_n=gain_n, gain_u=gain_u,
                p_A=p_A, p_B=p_B, p_C=p_C)


if __name__ == "__main__":
    run_pr1(verbose=True)
