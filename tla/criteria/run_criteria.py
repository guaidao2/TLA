"""预注册判据 runner（最小可跑版）。

已实现：P-PHY-1/2/3、P-COG-1（干净输入少琢磨）、P-LEARN-3（误差驱动学习探针，最高优先）。
已报告未判：P-COG-2（噪声步数增加，report-only）、Self_Slot 损失下降（report-only）。

判据纪律：判据锁死，跑数前不改判据只改代码；不过的判据如实记录，不硬凑。
"""
import torch
from tla.model import TLAModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


def evaluate(model, trajs, noisy=False):
    """按轨迹顺序推理（每轨迹重置身体，LTC 状态滚动），返回 (mse, steps_list, doubtful_frac)。"""
    mses, steps_list, doubtful = [], [], 0
    n = 0
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            obs = traj[t]
            if noisy:
                obs = obs + torch.randn_like(obs) * 0.3
            pred, info = model.infer(obs)
            n += 1
            if pred is None:
                doubtful += 1
                continue
            mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
            steps_list.append(info["steps"])
            if info["doubtful"]:
                doubtful += 1
    return (float(torch.tensor(mses).mean().item()) if mses else float("nan"),
            steps_list, doubtful / max(n, 1))


def run_criteria(seed=0, n_epochs=3, verbose=True):
    world = VariableSpeedWorld(seed=seed)
    train_trajs = world.trajectories(n_traj=40, T=50, speed_range=(0.8, 3.0))
    test_trajs = world.trajectories(n_traj=10, T=40, speed_range=(4.0, 5.0), seed=999)  # 未见 ω

    model = TLAModel(obs_dim=3, out_dim=2, seed=seed)
    base = TLAModel(obs_dim=3, out_dim=2, seed=seed + 77)   # 随机初始化基线（不训练）

    # ---- 基线 ----
    mse_random, _, _ = evaluate(base, test_trajs)
    idents = [traj[t + 1][:2] - traj[t][:2] for traj in test_trajs for t in range(len(traj) - 1)]
    mse_identity = float(torch.mean(torch.stack(idents) ** 2).item())

    # ---- 训练（误差驱动，无 BP，按轨迹）----
    ss_losses = []
    for epoch in range(n_epochs):
        for traj in train_trajs:
            model.reset()
            for t in range(len(traj) - 1):
                mse, ss_loss = model.train_step(traj[t], traj[t + 1])
                if ss_loss is not None:
                    ss_losses.append(ss_loss)
        if verbose:
            print(f"epoch {epoch+1}/{n_epochs} done (last mse={mse:.5f})")

    train_mse = mse
    ss_head, ss_tail = (torch.tensor(ss_losses[:100]).mean(),
                        torch.tensor(ss_losses[-100:]).mean()) if len(ss_losses) else (0.0, 0.0)

    # ---- 判据 ----
    mse_test, steps_clean, doubtful_frac = evaluate(model, test_trajs)
    _, steps_noisy, _ = evaluate(model, test_trajs, noisy=True)

    median_clean = float(torch.tensor(steps_clean).median().item())
    median_noisy = float(torch.tensor(steps_noisy).median().item())
    mean_clean = float(torch.tensor(steps_clean, dtype=torch.float32).mean().item())
    mean_noisy = float(torch.tensor(steps_noisy, dtype=torch.float32).mean().item())

    p_learn3 = (mse_test < 0.7 * mse_random) and (mse_test < mse_identity)
    p_cog1 = 1 < median_clean <= 3      # 预注册 ≤1 为显式已知差距（1 < median）
    p_cog2 = mean_noisy > mean_clean    # report-only（分布签名）

    rows = [
        ("P-PHY-1~3", "基板有界/断电/静息", "见 tests/test_substrate.py", "-"),
        ("P-COG-1", "干净输入少步即停 (median≤3)", f"median={median_clean} (noisy={median_noisy})",
         "PASS" if p_cog1 else "FAIL"),
        ("P-COG-2", "噪声步数增加 (report-only)", f"clean_mean={mean_clean:.2f} vs noisy_mean={mean_noisy:.2f}",
         "PASS" if p_cog2 else "report"),
        ("P-LEARN-3", "误差驱动学习探针：无 BP 学得动", f"trained={mse_test:.4f} random={mse_random:.4f} identity={mse_identity:.4f} train={train_mse:.5f}",
         "PASS" if p_learn3 else "FAIL"),
        ("Self_Slot", "自监督损失下降 (report-only)", f"head={ss_head:.4f} tail={ss_tail:.4f}", "report"),
    ]
    print("\n" + "=" * 72)
    print("TLA 判据报告（预注册 · 最小可跑版）")
    print("=" * 72)
    for num, name, detail, verdict in rows:
        print(f"  {num:<10} {name}\n      {detail}\n      -> {verdict}")
    print("=" * 72)
    return dict(train_mse=train_mse, mse_test=mse_test, mse_random=mse_random,
                mse_identity=mse_identity, median_clean=median_clean,
                median_noisy=median_noisy, p_learn3=p_learn3, p_cog1=p_cog1,
                doubtful_frac=doubtful_frac)


if __name__ == "__main__":
    result = run_criteria()
    ok = result["p_learn3"] and result["p_cog1"]
    print(f"\n核心判据: {'全部通过' if ok else '有未过项（如实记录）'}")
