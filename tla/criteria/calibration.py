"""P-COG-4：doubtful 标记校准性（预注册判据，先于跑数锁死）。

- 混合难度测试集（分布内/未见ω × 干净/噪声）收集 (confidence, error, doubtful)；
- 判据：置信度最低分位样本的实际误差 显著高于 最高分位（可靠性曲线单调下降）；
- doubtful 标记（预算耗尽）对应实际错误率显著更高（标记与实际错误率校准）。
"""
import torch
from tla.model import TLAModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


def collect(model, trajs, noise=0.0, seed=0):
    """返回 (confs, errs, doubtful_flags)。"""
    gen = torch.Generator().manual_seed(seed)
    confs, errs, doubts = [], [], []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            obs = traj[t]
            if noise > 0:
                obs = obs + torch.randn(3, generator=gen) * noise
            pred, info = model.infer(obs)
            if pred is None:
                continue
            confs.append(info["confidence"])
            errs.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
            doubts.append(info["doubtful"])
    return (torch.tensor(confs), torch.tensor(errs),
            torch.tensor(doubts, dtype=torch.bool))


def run_calibration(seed=0, n_epochs=3, n_traj=40, T=50, verbose=True):
    world = VariableSpeedWorld(seed=seed)
    train = world.trajectories(n_traj=n_traj, T=T, speed_range=(0.8, 3.0))
    indist = world.trajectories(n_traj=max(4, n_traj // 6), T=30,
                                speed_range=(1.0, 2.0), seed=7)
    unseen = world.trajectories(n_traj=max(4, n_traj // 6), T=30,
                                speed_range=(4.0, 5.0), seed=999)

    model = TLAModel(obs_dim=3, out_dim=2, seed=seed)
    for _ in range(n_epochs):
        for traj in train:
            model.reset()
            for t in range(len(traj) - 1):
                model.train_step(traj[t], traj[t + 1])

    parts = [("indist-clean", indist, 0.0), ("indist-noisy", indist, 0.3),
             ("unseen-clean", unseen, 0.0), ("unseen-noisy", unseen, 0.3)]
    confs, errs, doubts = [], [], []
    for _, trajs, noise in parts:
        c, e, d = collect(model, trajs, noise)
        confs.append(c)
        errs.append(e)
        doubts.append(d)
    conf = torch.cat(confs)
    err = torch.cat(errs)
    doubt = torch.cat(doubts)

    # 可靠性：按置信度分位分桶，检查错误随置信度下降
    qs = torch.quantile(conf, torch.tensor([0.25, 0.5, 0.75]))
    lo = err[conf <= qs[0]]
    hi = err[conf >= qs[2]]
    mse_lo, mse_hi = lo.mean().item(), hi.mean().item()
    # doubtful 校准：标记样本 vs 未标记样本
    if doubt.any() and (~doubt).any():
        mse_doubt = err[doubt].mean().item()
        mse_nodoubt = err[~doubt].mean().item()
    else:
        mse_doubt, mse_nodoubt = float("nan"), err.mean().item()
    doubt_rate = doubt.float().mean().item()

    p_cog4 = (mse_lo > 2.0 * mse_hi) and (mse_doubt > 2.0 * mse_nodoubt if torch.isfinite(torch.tensor(mse_doubt)) else False)

    if verbose:
        print("=" * 64)
        print("P-COG-4 doubtful 校准报告")
        print("=" * 64)
        print(f"  低置信分位 MSE={mse_lo:.4f}  高置信分位 MSE={mse_hi:.4f}  "
              f"(比率 {mse_lo / max(mse_hi, 1e-9):.2f})")
        print(f"  doubtful 标记 MSE={mse_doubt:.4f}  未标记 MSE={mse_nodoubt:.4f}  "
              f"(doubtful 率={doubt_rate:.3f})")
        print(f"  P-COG-4 (校准成立: 低置信>>高置信 ∧ doubtful>>未标记): "
              f"{'PASS' if p_cog4 else 'FAIL'}")
        print("=" * 64)
    return dict(mse_lo=mse_lo, mse_hi=mse_hi, mse_doubt=mse_doubt,
                mse_nodoubt=mse_nodoubt, doubt_rate=doubt_rate, p_cog4=p_cog4)


if __name__ == "__main__":
    run_calibration()
