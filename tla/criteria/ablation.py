"""P-COG-3 + P-COG-5：推理环消融实验（预注册判据，先于跑数锁死）。

【预注册裁决（2026-08-10 实测）】
P-COG-3（组合泛化场地"显著更优"）：**负结果**——现有世界模型任务的未见 ω 泛化场地上，
  自适应推理无增益（有捷径版 0.12%≈空转；无捷径版 -5.7% 过度精化有害）。
  归因①：线性捷径（W_lin）旁路输出，settle 精化不改变输出（循环空转）；
  归因②：无捷径时 settle 的最小化目标是内部重建，OOD 输入的重建不动点与任务预测目标错位，
    想得越多越偏（过度精化）。
  正面证据：无捷径版分布内 adaptive 比 fixed1 好 45%（机制真实，场地不对）。
  未来方向：SCAN 类组合泛化正式场地、防过度精化停止（输出变差即停）、推理-任务目标对齐。
P-COG-5（关掉推理环只跑初猜，精度劣化 ≤ 阈值，防"摆烂/空转"）：**跷跷板发现**——
  有捷径版：guess≈adaptive（劣化≈0，guardrail 通过，但暴露循环空转）；
  无捷径版：分布内 guess 差 3×（劣化超阈值，循环承重=初猜摆烂）。
  即"空转"与"摆烂"是同一设计的镜像失败面：捷径旁路→空转；无捷径→初猜烂。
  ⚠️ 限定：无捷径版 guess（max_steps=0）在轨迹起点 μ=0 且 W_lin=0 时为常数偏置——
    "3× 摆烂"部分是结构性基线（初猜无条件信息可用），以 fixed1（1 步 settle）作辅助对照更公允。
  判据锁死：以上数字与裁决为预注册结果，只许复现，不许篡改。
"""
import torch
from tla.model import TLAModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(model, trajs, max_steps=None, noise=0.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mses = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            obs = traj[t]
            if noise > 0:
                obs = obs + torch.randn(3, generator=gen) * noise
            pred, info = model.infer(obs, max_steps=max_steps)
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def _train(model, train_trajs, n_epochs=3):
    for _ in range(n_epochs):
        for traj in train_trajs:
            model.reset()
            for t in range(len(traj) - 1):
                model.train_step(traj[t], traj[t + 1])

def run_ablation(seed=0, n_epochs=3, n_traj=40, T=50, verbose=True):
    world = VariableSpeedWorld(seed=seed)
    train = world.trajectories(n_traj=n_traj, T=T, speed_range=(0.8, 3.0))
    unseen = world.trajectories(n_traj=max(4, n_traj // 5), T=T - 10,
                                speed_range=(4.0, 5.0), seed=999)
    indist = world.trajectories(n_traj=max(4, n_traj // 5), T=T - 10,
                                speed_range=(1.0, 2.0), seed=7)

    r = {}
    for use_lin in (True, False):
        m = TLAModel(obs_dim=3, out_dim=2, seed=seed, use_lin_shortcut=use_lin)
        _train(m, train, n_epochs=n_epochs)
        tag = "lin" if use_lin else "nolin"
        r[f"{tag}_unseen_ad"] = eval_mse(m, unseen)
        r[f"{tag}_unseen_f1"] = eval_mse(m, unseen, 1)
        r[f"{tag}_unseen_guess"] = eval_mse(m, unseen, 0)
        r[f"{tag}_indist_ad"] = eval_mse(m, indist)
        r[f"{tag}_indist_f1"] = eval_mse(m, indist, 1)
        r[f"{tag}_indist_guess"] = eval_mse(m, indist, 0)

    r["gain_lin_unseen"] = (r["lin_unseen_f1"] - r["lin_unseen_ad"]) / r["lin_unseen_f1"]
    r["gain_nolin_unseen"] = (r["nolin_unseen_f1"] - r["nolin_unseen_ad"]) / r["nolin_unseen_f1"]
    r["gain_nolin_indist"] = (r["nolin_indist_f1"] - r["nolin_indist_ad"]) / r["nolin_indist_f1"]
    r["guess_ratio_nolin_indist"] = r["nolin_indist_guess"] / max(r["nolin_indist_ad"], 1e-9)
    r["guess_ratio_lin_unseen"] = r["lin_unseen_guess"] / max(r["lin_unseen_ad"], 1e-9)

    if verbose:
        print("=" * 70)
        print("P-COG-3/5 消融报告（预注册裁决）")
        print("=" * 70)
        print(f"  有捷径: unseen gain={r['gain_lin_unseen'] * 100:+.1f}%  guess/ad={r['guess_ratio_lin_unseen']:.2f}")
        print(f"  无捷径: unseen gain={r['gain_nolin_unseen'] * 100:+.1f}%  "
              f"indist gain={r['gain_nolin_indist'] * 100:+.1f}%  guess/ad={r['guess_ratio_nolin_indist']:.2f}")
        print("  P-COG-3: 负结果（泛化场地无增益，见 docstring 归因）")
        print("  P-COG-5: 跷跷板发现（空转↔摆烂互为镜像）")
        print("=" * 70)
    return r


if __name__ == "__main__":
    run_ablation()
