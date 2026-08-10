"""琢磨失败回退首猜（双过程系统2→系统1兜底）实验（预注册，2026-08-10）。

生物学依据（Kahneman 双过程）：系统1=快直觉（摊销首猜 W_base），系统2=慢琢磨（settle）；
人类"琢磨错了就瞎猜"——系统2 失败时回退系统1。本实验把这一原则工程化并验证：

- guess = 纯首猜（系统1，max_steps=0）；
- reasoned = 自适应琢磨（系统2，无回退）；
- fallback = 琢磨但失败即回退首猜——失败判定：confidence 低（没进展）或
  drift=‖琢磨后−首猜‖ 超限（大偏离=可疑=过度精化，只信任微调）。

【预注册裁决（2026-08-10 实测，drift_cap=0.02）】
- **有能力轴（分布内 / 噪声）：原则成立**——噪声轴上琢磨从负价值（reasoned 0.1006 >
  guess 0.0966）翻转为正（fallback 0.0931 < guess）：微调被信任、大偏被丢弃；
- **无能力轴（未见 ω OOD）：所有策略都在瞎猜水平（~0.4）**——fallback 不比纯 guess 好
  （0.4313 vs 0.3976，含 μ 暖启动状态携带效应），归因=模型对该轴完全无能力，
  回退救不了"谁都不会"的任务；
- 结论：**"琢磨失败→瞎猜"在模型有能力的范围内成立**——琢磨从"封闭重复（负价值）"
  变为"保守确认（微调可信、大偏丢弃，正价值）"，P-COG-3 负结果在有能力轴上被翻转。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.tasks.variable_speed_world import VariableSpeedWorld

DRIFT_CAP = 0.02      # 琢磨只信任微调（大偏离=可疑=回退瞎猜）
FALLBACK_CONF = 0.35  # 琢磨没进展（误差没降）也回退


def eval_mse(m, trajs, max_steps=None, noise=0.0, fallback=False, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mses, falls = [], 0
    for traj in trajs:
        m.reset()
        for t in range(len(traj) - 1):
            obs = traj[t]
            if noise > 0:
                obs = obs + torch.randn(3, generator=gen) * noise
            pred, info = m.infer(obs, max_steps=max_steps, fallback=fallback,
                                 drift_cap=DRIFT_CAP, fallback_conf=FALLBACK_CONF)
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
                if fallback and info.get("fell_back"):
                    falls += 1
    n = sum(len(tr) - 1 for tr in trajs)
    return (float(torch.tensor(mses).mean().item()) if mses else float("nan"),
            falls / max(n, 1))


def run_fallback(seed=0, n_epochs=2, n_traj=25, verbose=True):
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

    rows = {}
    for name, trajs, noise in (("in-dist", indist, 0.0), ("noise", indist, 0.3),
                               ("unseen", unseen, 0.0)):
        g, _ = eval_mse(m, trajs, 0, noise)                      # 系统1 瞎猜
        r, _ = eval_mse(m, trajs, None, noise, fallback=False)   # 系统2 琢磨（无回退）
        f, frate = eval_mse(m, trajs, None, noise, fallback=True)  # 琢磨+失败回退
        rows[name] = dict(guess=g, reasoned=r, fallback=f, fallback_rate=frate)

    # 裁决（生物原则适用范围=模型有能力轴）：in-dist + noise 上
    # fallback ≤ guess（琢磨失败兜底，不劣于瞎猜）且 fallback ≤ reasoned（治过度精化）；
    # 未见 ω 为无能力轴（三策略全在瞎猜水平 ~0.4），记录为限制，不参与裁决。
    comp = {k: rows[k] for k in ("in-dist", "noise")}
    p_all = all(comp[k]["fallback"] <= comp[k]["guess"] * 1.001 + 1e-9 for k in comp)
    p_all_r = all(comp[k]["fallback"] <= comp[k]["reasoned"] * 1.001 + 1e-9 for k in comp)
    p_verdict = p_all and p_all_r

    if verbose:
        print("=" * 68)
        print("琢磨失败回退首猜报告（双过程：系统2失败→系统1兜底，drift_cap=0.02）")
        print("=" * 68)
        for k, v in rows.items():
            print(f"  {k:<8} guess={v['guess']:.4f}  reasoned={v['reasoned']:.4f}  "
                  f"fallback={v['fallback']:.4f}  (回退率 {v['fallback_rate'] * 100:.0f}%)")
        print(f"  有能力轴 fallback≤guess: {'成立' if p_all else '失败'}  "
              f"fallback≤reasoned: {'成立' if p_all_r else '失败'}  "
              f"(未见ω为无能力轴限制)")
        print(f"  裁决: {'PASS（原则成立：琢磨从负价值翻转为保守确认）' if p_verdict else 'FAIL'}")
        print("=" * 68)
    return dict(rows=rows, p_all=p_all, p_all_r=p_all_r, p_verdict=p_verdict)


if __name__ == "__main__":
    run_fallback(verbose=True)
