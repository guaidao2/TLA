"""奇点基板换装判据（预注册，2026-08-11）：SingularitySubstrate 代替 LTCCell 复测核心判据。

动机：奇点神经元单细胞机制成立（SN-1..6），但"换进网络能不能用"是另一回事——
LTC 的 h 连续（tanh 软饱和），奇点的 h 呈"幽灵态 ~0.02 / 饱和态 ~0.9"双簇（稀疏激活），
且输入暴胀边界 I≈0.05 远低于 LTC 工作区间（input_scale 需标定）。
判据（跑数前锁死，与 LTC 基线数字对比，如实记录）：
  S-B1 学习（P-LEARN-3 判据复测）：未见 ω MSE < 0.7×随机基线 且 < 恒等基线；
  S-B2 琢磨步数（P-COG-1/2 方向复测）：噪声均值步数 > 干净均值步数（差分方向）；
  S-B3 防遗忘（P-LEARN-1 EWC 复测）：有 EWC 保留率 ≥ 0.95。
判据锁死：跑数后任何裁决不得篡改；基板换装是代码改动，判据标准不变。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.substrate.singularity_substrate import SingularitySubstrate
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(model, trajs, noisy=False):
    mses = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            o = traj[t]
            if noisy:
                o = o + 0.1 * torch.randn_like(o)
            pred, _ = model.infer(o)
            mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def steps_stats(model, trajs, noisy=False):
    ss = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            o = traj[t]
            if noisy:
                o = o + 0.1 * torch.randn_like(o)
            _, info = model.infer(o)
            ss.append(info["steps"])
    ss = torch.tensor(ss, dtype=torch.float32)
    return float(ss.median().item()), float(ss.mean().item())


def activation_stats(substrate, trajs):
    """基板激活统计：h > 0.5 的比例（奇点双簇行为是否落在稀疏区间）。"""
    hot = 0
    total = 0
    for traj in trajs:
        substrate.reset()
        for t in range(len(traj)):
            substrate.forward(traj[t])
            hot += int((substrate.h > 0.5).sum().item())
            total += substrate.hidden
    return hot / max(total, 1)


def run_swap(seed=0, verbose=True, n_traj=20, T=30, n_ep=2, input_scale=0.4):
    world = VariableSpeedWorld(seed=seed, mode="spring")
    train = world.trajectories(n_traj=n_traj, T=T, speed_range=(0.8, 3.0))
    test_clean = world.trajectories(n_traj=3, T=20, speed_range=(1.0, 2.0), seed=7)
    test_unseen = world.trajectories(n_traj=3, T=20, speed_range=(3.5, 5.0), seed=999)

    def make_model(sd):
        # 换装奇点基板，指定 input_scale（暴胀边界标定）
        return TLAPR1Model(seed=sd,
                           substrate_cls=lambda **kw: SingularitySubstrate(
                               input_scale=input_scale, **kw))
    m = make_model(seed)
    m_rand = make_model(seed)
    mse_rand = eval_mse(m_rand, test_unseen)
    # 恒等基线（预测 s_t）
    id_mses = [float(torch.mean((t[0][:2] - t[1][:2]) ** 2).item())
               for tr in test_unseen for t in zip(tr, tr[1:])]
    mse_id = float(torch.tensor(id_mses).mean().item())
    # 训练
    for _ in range(n_ep):
        for traj in train:
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1])
    mse_unseen = eval_mse(m, test_unseen)
    p_b1 = mse_unseen < 0.7 * mse_rand and mse_unseen < mse_id
    # S-B2 步数
    med_c, mean_c = steps_stats(m, test_clean)
    _, mean_n = steps_stats(m, test_clean, noisy=True)
    p_b2 = mean_n > mean_c
    # S-B3 防遗忘（EWC）
    def run_ab():
        mm = make_model(seed)
        wa = VariableSpeedWorld(seed=seed, mode="spring")
        wb = VariableSpeedWorld(seed=seed + 10, mode="spring")
        train_a = wa.trajectories(n_traj=n_traj, T=T, speed_range=(0.8, 1.5))
        test_a = wa.trajectories(n_traj=3, T=15, speed_range=(0.9, 1.3), seed=7)
        train_b = wb.trajectories(n_traj=n_traj, T=T, speed_range=(3.5, 4.5))
        mm.pcn.start_consolidation()
        for _ in range(n_ep):
            for traj in train_a:
                for t in range(len(traj) - 1):
                    mm.train_step(traj[t], traj[t + 1], consolidate=True)
        mse_a0 = eval_mse(mm, test_a)
        mm.pcn.finalize_consolidation()
        mm.lam = 10.0
        for _ in range(n_ep):
            for traj in train_b:
                for t in range(len(traj) - 1):
                    mm.train_step(traj[t], traj[t + 1], protect=True)
        mse_a1 = eval_mse(mm, test_a)
        return mse_a0 / max(mse_a1, 1e-12)
    ret_ewc = run_ab()
    p_b3 = ret_ewc >= 0.95
    # 激活率
    act = activation_stats(SingularitySubstrate(in_dim=3, hidden=16, seed=0,
                                                input_scale=input_scale), train)

    out = dict(mse_unseen=mse_unseen, mse_rand=mse_rand, mse_id=mse_id, p_b1=p_b1,
               med_c=med_c, mean_c=mean_c, mean_n=mean_n, p_b2=p_b2,
               ret_ewc=ret_ewc, p_b3=p_b3, act=act)
    if verbose:
        print("=" * 64)
        print(f"奇点基板换装判据 (n_traj={n_traj}, T={T}, n_ep={n_ep})")
        print("=" * 64)
        print(f"S-B1 学习: 未见ω={mse_unseen:.4f} vs 随机={mse_rand:.4f} "
              f"vs 恒等={mse_id:.4f} → {'PASS' if p_b1 else 'FAIL'}")
        print(f"S-B2 步数: 干净 mean={mean_c:.2f} / 噪声 mean={mean_n:.2f} "
              f"(median={med_c}) → {'PASS' if p_b2 else 'FAIL'}")
        print(f"S-B3 防遗忘: EWC 保留率={ret_ewc:.1%} (≥95%) → {'PASS' if p_b3 else 'FAIL'}")
        print(f"基板激活率(h>0.5)={act:.1%}（LTC 参照：连续 h，无双簇）")
        print("=" * 64)
    return out


if __name__ == "__main__":
    run_swap(verbose=True)
