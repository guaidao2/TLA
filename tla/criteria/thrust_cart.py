"""2D 推力体世界模型判据（预注册，2026-08-11）：含动作通道的持续学习第一仗。

背景：弹簧世界（3 维纯观测）是"热身"，本任务验证 TLA 在**含动作、非线性动力学、
物理参数任务序列**上的能力——世界模型真正要面对的场景（s,a → s'）。

判据（跑数前锁死）：
  TC-1 学习成立（无 BP）：未见重力（g 分布外）测试 MSE < 0.7×随机基线 且 < 恒等基线；
  TC-2 动作条件化：恒推 a=(+1,+1) 与 a=(−1,−1) 下，模型预测的 vx' 均值必须显著不同
     （模型真正使用动作信息，而非只看状态）——|mean_vx'_p1 − mean_vx'_m1| > 0.05；
  TC-3 防遗忘：A(g=1.0) → B(g=2.0) 任务序列，EWC 保留率 ≥ 0.95（A 测试集）。
判据锁死：跑数后任何裁决不得篡改，只许改代码。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.tasks.thrust_cart import ThrustCartWorld


def eval_mse(model, trajs):
    mses = []
    for traj in trajs:
        model.reset()
        for s, a, s_next in traj:
            pred, _ = model.infer(torch.cat([s, a]))
            mses.append(float(torch.mean((pred - s_next) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def eval_mean_vx(model, trajs, act):
    """恒推 act 下预测的 vx' 均值（TC-2 动作条件化信号）。"""
    vxs = []
    for traj in trajs:
        model.reset()
        for s, _, _ in traj:
            pred, _ = model.infer(torch.cat([s, torch.tensor(act, dtype=torch.float32)]))
            vxs.append(pred[2].item())
    return float(torch.tensor(vxs).mean().item())


def run(seed=0, verbose=True, n_traj=30, T=40, n_ep=2):
    world = ThrustCartWorld(g=1.0, seed=seed)
    train = world.task(g=1.0, n=n_traj, T=T, seed_shift=1)
    test_other_g = world.task(g=2.5, n=4, T=20, seed_shift=99)   # 未见重力
    test_act_p1 = world.task(g=1.0, n=4, T=20, seed_shift=7)
    test_act_m1 = world.task(g=1.0, n=4, T=20, seed_shift=8)

    m = TLAPR1Model(obs_dim=6, out_dim=4, seed=seed)
    # 随机基线
    m_rand = TLAPR1Model(obs_dim=6, out_dim=4, seed=seed)
    mse_rand = eval_mse(m_rand, test_other_g)
    # 恒等基线
    id_mses = [float(torch.mean((s - s_next) ** 2).item())
               for traj in test_other_g for s, _, s_next in traj]
    mse_id = float(torch.tensor(id_mses).mean().item())
    # 训练
    for _ in range(n_ep):
        for traj in train:
            for s, a, s_next in traj:
                m.train_step(torch.cat([s, a]), s_next)
    mse_g = eval_mse(m, test_other_g)
    p_tc1 = mse_g < 0.7 * mse_rand and mse_g < mse_id

    # TC-2 动作条件化
    vx_p1 = eval_mean_vx(m, test_act_p1, (1.0, 1.0))
    vx_m1 = eval_mean_vx(m, test_act_m1, (-1.0, -1.0))
    p_tc2 = abs(vx_p1 - vx_m1) > 0.05

    # TC-3 防遗忘（重力任务序列 + EWC）
    mm = TLAPR1Model(obs_dim=6, out_dim=4, seed=seed)
    train_a = world.task(g=1.0, n=n_traj, T=T, seed_shift=2)
    test_a = world.task(g=1.0, n=3, T=15, seed_shift=3)
    train_b = world.task(g=2.0, n=n_traj, T=T, seed_shift=4)
    mm.pcn.start_consolidation()
    for _ in range(n_ep):
        for traj in train_a:
            for s, a, s_next in traj:
                mm.train_step(torch.cat([s, a]), s_next, consolidate=True)
    mse_a0 = eval_mse(mm, test_a)
    mm.pcn.finalize_consolidation()
    mm.lam = 10.0
    for _ in range(n_ep):
        for traj in train_b:
            for s, a, s_next in traj:
                mm.train_step(torch.cat([s, a]), s_next, protect=True)
    mse_a1 = eval_mse(mm, test_a)
    ret = mse_a0 / max(mse_a1, 1e-12)
    p_tc3 = ret >= 0.95

    out = dict(mse_g=mse_g, mse_rand=mse_rand, mse_id=mse_id, p_tc1=p_tc1,
               vx_p1=vx_p1, vx_m1=vx_m1, p_tc2=p_tc2,
               ret=ret, p_tc3=p_tc3)
    if verbose:
        print("=" * 64)
        print(f"2D 推力体世界判据 (n_traj={n_traj}, T={T}, n_ep={n_ep})")
        print("=" * 64)
        print(f"TC-1 学习: 未见重力 MSE={mse_g:.4f} vs 随机={mse_rand:.4f} "
              f"vs 恒等={mse_id:.4f} → {'PASS' if p_tc1 else 'FAIL'}")
        print(f"TC-2 动作条件化: 恒推+1 vx'={vx_p1:.4f} vs 恒推−1 vx'={vx_m1:.4f} "
              f"→ {'PASS' if p_tc2 else 'FAIL'}")
        print(f"TC-3 防遗忘: g1.0→g2.0 EWC 保留率={ret:.1%} (≥95%) "
              f"→ {'PASS' if p_tc3 else 'FAIL'}")
        print("=" * 64)
    return out


if __name__ == "__main__":
    run(verbose=True)
