"""TLA 消融实验（thrust_cart，预注册）：逐件关掉独有组件，看性能降不降。

目的：回答"TLA 的独有机制是否在起作用"——关掉后不降 = 该机制没干活（用处悬了）。

判据（跑数前锁死）：
  AB-1 推理环贡献：自适应（settle 多步）vs 固定 1 步 vs 纯首猜（0 步）——
     若 mse_fixed1 ≥ mse_adaptive × 1.05（关掉推理环显著降）→ 推理环在起作用；
     若 ≈（不降）→ 推理环空转（弹簧先例 P-COG-3 已示倾向）；
  AB-2 双过程回退：infer 默认 fallback=False（opt-in）——贡献=0 由配置决定，
     代码分析记录（无行为差异，非实验）；
  AB-3 Self_Slot：只学自模型、只影响默认关闭的一致性门（self_consistency_gate=None）
     与 scratchpad 诊断——不参与输出路径，贡献=0 由代码结构决定，代码分析记录；
  AB-4 摊销捷径（W_base）：freeze_base=True 训练 vs 正常——
     报告（历史先例：无捷径弱学习 0.11 vs 0.004，预期 W_base 是学习主力）。
判据锁死：AB-1 阈值（1.05×）预注册，不符→如实记录（推理环空转是预期内负结果）。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.tasks.thrust_cart import ThrustCartWorld


def eval_tla_at(model, trajs, max_steps=None):
    mses = []
    for traj in trajs:
        model.reset()
        for s, a, s_next in traj:
            pred, _ = model.infer(torch.cat([s, a]), max_steps=max_steps)
            mses.append(float(torch.mean((pred - s_next) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def run(seed=0, verbose=True, n_traj=30, T=40, n_ep=2):
    world = ThrustCartWorld(g=1.0, seed=seed)
    train = world.task(g=1.0, n=n_traj, T=T, seed_shift=1)
    test = world.task(g=2.5, n=4, T=20, seed_shift=99)

    # 正常训练（A0 基线）
    m = TLAPR1Model(obs_dim=6, out_dim=4, seed=seed)
    for _ in range(n_ep):
        for traj in train:
            for s, a, s_next in traj:
                m.train_step(torch.cat([s, a]), s_next)

    # AB-1 推理环：同一模型，不同推理深度
    mse_ad = eval_tla_at(m, test)              # 自适应（默认 settle）
    mse_f1 = eval_tla_at(m, test, max_steps=1) # 固定 1 步（关掉多步精化）
    mse_g0 = eval_tla_at(m, test, max_steps=0) # 纯首猜（0 步）
    p_ab1 = mse_f1 >= mse_ad * 1.05            # 关掉推理环显著降 → 在起作用

    # AB-4 摊销捷径：freeze_base 训练
    m_fb = TLAPR1Model(obs_dim=6, out_dim=4, seed=seed)
    for _ in range(n_ep):
        for traj in train:
            for s, a, s_next in traj:
                m_fb.train_step(torch.cat([s, a]), s_next, freeze_base=True)
    mse_fb = eval_tla_at(m_fb, test, max_steps=1)

    out = dict(mse_ad=mse_ad, mse_f1=mse_f1, mse_g0=mse_g0, p_ab1=p_ab1,
               mse_fb=mse_fb,
               ab2="双过程回退默认 off（fallback=False，opt-in）——贡献=0 由配置决定",
               ab3="Self_Slot 只影响默认关闭的一致性门与诊断——不参与输出路径，贡献=0")
    if verbose:
        print("=" * 70)
        print(f"TLA 消融（thrust_cart，seed={seed}，n_traj={n_traj}/T={T}/n_ep={n_ep}）")
        print("=" * 70)
        print(f"  AB-1 推理环: 自适应={mse_ad:.4f} / 固定1步={mse_f1:.4f} / "
              f"纯首猜={mse_g0:.4f}")
        print(f"     关掉推理环显著降(≥5%): {'PASS' if p_ab1 else 'FAIL（空转，预期内）'}")
        print(f"  AB-2 双过程回退: {out['ab2']}")
        print(f"  AB-3 Self_Slot: {out['ab3']}")
        print(f"  AB-4 摊销捷径: freeze_base={mse_fb:.4f} vs 正常={mse_f1:.4f} "
              f"（比值 {mse_fb / max(mse_f1, 1e-12):.1f}×）")
        print("=" * 70)
    return out


if __name__ == "__main__":
    run(verbose=True)
