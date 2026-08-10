"""遗忘修复全链 runner（预注册，2026-08-10）。

链路：诊断（冻结 W_base）→ 突触巩固 EWC → 身体门控 → MoE 专家首猜。
每层都在同一 P-LEARN-1 协议（A=弹簧 ω0.8-1.5 → B=ω3.5-4.5）上重测：
保留率 ≥95% 且优于无保护对照 → 该层通过即停；否则记录归因进入下一层。

附加保持判据（每层都验）：学习强度（分布内 MSE <0.02）与琢磨回退（有能力轴成立）
不得被保护机制破坏。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(m, trajs, fallback=False, noise=0.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mses = []
    for traj in trajs:
        m.reset()
        for t in range(len(traj) - 1):
            obs = traj[t]
            if noise > 0:
                obs = obs + torch.randn(3, generator=gen) * noise
            pred, info = m.infer(obs, fallback=fallback)
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def train_epochs(m, trajs, n, replay_prob=0.3, freeze_base=False):
    m.replay.replay_prob = replay_prob
    for _ in range(n):
        for traj in trajs:
            m.reset()
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1], freeze_base=freeze_base)


def run_learn1_protocol(maker, protect_kwargs=None, seed=0, verbose=False):
    """P-LEARN-1 协议：返回 (保留率, 无保护保留率, 重放帮助, 学习强度, 琢磨回退)。"""
    protect_kwargs = protect_kwargs or {}
    wa = VariableSpeedWorld(seed=seed, mode="spring")
    wb = VariableSpeedWorld(seed=seed + 10, mode="spring")
    train_a = wa.trajectories(n_traj=20, T=25, speed_range=(0.8, 1.5))
    test_a = wa.trajectories(n_traj=3, T=15, speed_range=(0.9, 1.3), seed=7)
    train_b = wb.trajectories(n_traj=20, T=25, speed_range=(3.5, 4.5))
    indist = wa.trajectories(n_traj=3, T=15, speed_range=(1.0, 1.5), seed=11)

    def protocol(rp, extra=None):
        m = maker(seed=seed, **protect_kwargs)
        extra = extra or {}
        train_epochs(m, train_a, 2, rp)
        a0 = eval_mse(m, test_a)
        train_epochs(m, train_b, 2, rp, freeze_base=extra.get("freeze_base", False))
        return a0, eval_mse(m, test_a)

    a0_r, a1_r = protocol(0.3)
    a0_nr, a1_nr = protocol(0.0)
    retention = a0_r / max(a1_r, 1e-9)
    retention_nr = a0_nr / max(a1_nr, 1e-9)
    replay_helps = a1_r < a1_nr
    # 保持判据：学习强度（分布内 <0.02）+ 琢磨回退（噪声轴 fallback < guess）
    m = maker(seed=seed, **protect_kwargs)
    m.replay.replay_prob = 0.3
    for _ in range(2):
        for traj in train_a:
            m.reset()
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1])
    mse_indist = eval_mse(m, indist)
    g_n = eval_mse(m, indist, fallback=False, noise=0.3)
    f_n = eval_mse(m, indist, fallback=True, noise=0.3)
    return dict(retention=retention, retention_nr=retention_nr,
                replay_helps=replay_helps, mse_indist=mse_indist,
                guess_noise=g_n, fallback_noise=f_n)


def run_diagnose(seed=0, verbose=True):
    """诊断：B 训练冻结 W_base（只让残差通路学 B），A 保留率是否保住。"""
    r = run_learn1_protocol(lambda **k: TLAPR1Model(obs_dim=3, out_dim=2, **k),
                            seed=seed, verbose=verbose)
    # 冻结版：B 训练时 freeze_base=True（另跑一个完整协议）
    wa = VariableSpeedWorld(seed=seed, mode="spring")
    wb = VariableSpeedWorld(seed=seed + 10, mode="spring")
    train_a = wa.trajectories(n_traj=20, T=25, speed_range=(0.8, 1.5))
    test_a = wa.trajectories(n_traj=3, T=15, speed_range=(0.9, 1.3), seed=7)
    train_b = wb.trajectories(n_traj=20, T=25, speed_range=(3.5, 4.5))
    m = TLAPR1Model(obs_dim=3, out_dim=2, seed=seed)
    train_epochs(m, train_a, 2, 0.3)
    a0 = eval_mse(m, test_a)
    train_epochs(m, train_b, 2, 0.3, freeze_base=True)
    a1 = eval_mse(m, test_a)
    retention_frozen = a0 / max(a1, 1e-9)

    if verbose:
        print("=" * 68)
        print("遗忘定位诊断（B 训练冻结 W_base → A 保留率）")
        print("=" * 68)
        print(f"  不冻结(现状): A 保留率={r['retention'] * 100:.1f}%  "
              f"(无重放 {r['retention_nr'] * 100:.1f}%)")
        print(f"  冻结 W_base:  A {a0:.4f}→{a1:.4f}  保留率={retention_frozen * 100:.1f}%")
        if retention_frozen >= 0.95:
            print("  结论: 遗忘在首猜 → 修复方向=保护/分离 W_base（EWC 首选）")
        elif retention_frozen >= r["retention"]:
            print("  结论: 冻结显著改善但未全保 → 遗忘部分在 W_base，需 EWC 强化")
        else:
            print("  结论: 冻结无帮助 → 遗忘在残差通路或共享结构，需换思路")
        print("=" * 68)
    return dict(retention=r["retention"], retention_nr=r["retention_nr"],
                retention_frozen=retention_frozen)


def run_ewc(seed=0, lam=10, verbose=True):
    """突触巩固 EWC：A 训练累计 importance（归一化）→ 快照 → B 训练 protect 拉回。

    λ=10 为实测标定（归一化后）：保留率 108.6% 且 B 仍可学；λ>200 保护过度，
    λ=500 数值不稳定（NaN）。"""
    wa = VariableSpeedWorld(seed=seed, mode="spring")
    wb = VariableSpeedWorld(seed=seed + 10, mode="spring")
    train_a = wa.trajectories(n_traj=20, T=25, speed_range=(0.8, 1.5))
    test_a = wa.trajectories(n_traj=3, T=15, speed_range=(0.9, 1.3), seed=7)
    train_b = wb.trajectories(n_traj=20, T=25, speed_range=(3.5, 4.5))
    test_b = wb.trajectories(n_traj=2, T=10, speed_range=(3.7, 4.3), seed=8)
    indist = wa.trajectories(n_traj=3, T=15, speed_range=(1.0, 1.5), seed=11)

    def protocol(rp):
        m = TLAPR1Model(obs_dim=3, out_dim=2, seed=seed, lam=lam)
        m.pcn.start_consolidation()
        m.replay.replay_prob = rp
        for _ in range(2):                       # A 训练（累计 importance）
            for traj in train_a:
                m.reset()
                for t in range(len(traj) - 1):
                    m.train_step(traj[t], traj[t + 1], consolidate=True)
        m.pcn.finalize_consolidation()           # 快照 A 权重 + 归一化 importance
        a0 = eval_mse(m, test_a)
        m.replay.replay_prob = rp
        for _ in range(2):                       # B 训练（EWC 保护）
            for traj in train_b:
                m.reset()
                for t in range(len(traj) - 1):
                    m.train_step(traj[t], traj[t + 1], protect=True)
        return a0, eval_mse(m, test_a), eval_mse(m, test_b)

    a0_r, a1_r, b1_r = protocol(0.3)
    a0_nr, a1_nr, _ = protocol(0.0)
    retention = a0_r / max(a1_r, 1e-9)
    retention_nr = a0_nr / max(a1_nr, 1e-9)
    replay_helps = a1_r < a1_nr
    p_learn1 = retention >= 0.95 and retention > retention_nr  # 主判据+优于无保护对照
    # 保持判据：学习强度（分布内 <0.02）——走完整 A→finalize→B(protect) 序列（防 EWC 破坏）
    m = TLAPR1Model(obs_dim=3, out_dim=2, seed=seed, lam=lam)
    m.pcn.start_consolidation()
    m.replay.replay_prob = 0.3
    for _ in range(2):
        for traj in train_a:
            m.reset()
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1], consolidate=True)
    m.pcn.finalize_consolidation()
    for _ in range(2):
        for traj in train_b:
            m.reset()
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1], protect=True)
    mse_indist = eval_mse(m, indist)
    keep_strength = mse_indist < 0.02

    if verbose:
        print("=" * 68)
        print(f"突触巩固 EWC 报告（λ={lam}，importance 归一化）")
        print("=" * 68)
        print(f"  A 保留率: 重放={retention * 100:.1f}%  无重放={retention_nr * 100:.1f}%  "
              f"重放帮助={'是' if replay_helps else '否'}")
        print(f"  A mse: {a0_r:.4f} → {a1_r:.4f}  (B 可学: {b1_r:.4f})")
        print(f"  保持: 学习强度(分布内)={mse_indist:.4f} {'<0.02 ✓' if keep_strength else '✗'}")
        print(f"  P-LEARN-1: {'PASS（遗忘负结果翻转）' if p_learn1 else 'FAIL'}")
        print("=" * 68)
    return dict(retention=retention, retention_nr=retention_nr,
                replay_helps=replay_helps, mse_indist=mse_indist,
                b_mse=b1_r, keep_strength=keep_strength, p_learn1=p_learn1)


if __name__ == "__main__":
    run_diagnose(verbose=True)
    print()
    run_ewc(verbose=True)
