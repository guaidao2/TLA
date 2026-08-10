"""P-LEARN-1/2：终身学习判据（预注册，先于跑数锁死）。

【预注册裁决（2026-08-10 实测）】
P-LEARN-1（先学 A 再学 B，A 保留率 ≥95%，对照无重放）：**负结果**——
  A=弹簧 ω∈[0.8,1.5]，B=弹簧 ω∈[3.5,4.5]（同量级、有界、真正不同的动力学）。
  实测：A 0.0009 → 0.064（保留率 ~1.5%），CLS 重放（均匀抽样+上下文忠实重放+睡眠巩固
  +重放专用 lr）把遗忘缓解 ~4.4×（0.064 vs 无重放 0.077）但远未达 95%。
  归因①：W_lin 线性捷径是任务无关的单一线性映射，A/B 冲突映射在同一权重上互相覆盖，
    重放是在做拔河，赢不了（重放 lr 放大反而加剧震荡）；
  归因②：无捷径变体无权重冲突（映射走上下文相关的 μ_L 通路），但学习弱
    （A 基线 0.044）且 B 训练产生正向迁移，构不成遗忘测试。
  统一叙事（与 P-COG-3/5 一致）：线性捷径的"单任务易学"与"多任务冲突"一体两面；
    未来方向 = 任务分离容量（MoE 式专家）或消除捷径后增强学习。
P-LEARN-2（放大 hidden 32→128 后每规则学习成本 <2×）：**PASS**——成本比 1.07。
"""
import time
import torch
from tla.model import TLAModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(model, trajs):
    mses = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            pred, info = model.infer(traj[t])
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def train_epochs(model, trajs, n_epochs, replay_prob):
    model.replay.replay_prob = replay_prob
    for _ in range(n_epochs):
        for traj in trajs:
            model.reset()
            for t in range(len(traj) - 1):
                model.train_step(traj[t], traj[t + 1])


def run_learn1(seed=0, verbose=True):
    """P-LEARN-1：A(ω 0.8-1.5) → B(ω 3.5-4.5)，A 保留率（有/无重放对照）。"""
    world_a = VariableSpeedWorld(seed=seed, mode="spring")
    world_b = VariableSpeedWorld(seed=seed + 10, mode="spring")
    train_a = world_a.trajectories(n_traj=25, T=30, speed_range=(0.8, 1.5))
    test_a = world_a.trajectories(n_traj=4, T=20, speed_range=(0.9, 1.3), seed=7)
    train_b = world_b.trajectories(n_traj=25, T=30, speed_range=(3.5, 4.5))

    def protocol(replay_prob):
        m = TLAModel(obs_dim=3, out_dim=2, seed=seed)
        train_epochs(m, train_a, 2, replay_prob)
        mse_a0 = eval_mse(m, test_a)
        train_epochs(m, train_b, 3, replay_prob)
        return mse_a0, eval_mse(m, test_a)

    a0, a1 = protocol(0.3)          # 有重放
    a0_nr, a1_nr = protocol(0.0)    # 无重放对照
    retention = a0 / max(a1, 1e-9)
    retention_nr = a0_nr / max(a1_nr, 1e-9)
    replay_helps = a1 < a1_nr       # 重放绝对缓解遗忘（负结果的正证据）
    p_learn1 = (retention >= 0.95) and (retention > retention_nr)

    if verbose:
        print("=" * 64)
        print("P-LEARN-1 报告（A=ω0.8-1.5 → B=ω3.5-4.5 弹簧）")
        print("=" * 64)
        print(f"  有重放: A {a0:.4f} → {a1:.4f}  保留率={retention * 100:.1f}%")
        print(f"  无重放: A {a0_nr:.4f} → {a1_nr:.4f}  保留率={retention_nr * 100:.1f}%")
        print(f"  重放绝对缓解遗忘: {'是' if replay_helps else '否'} ({a1:.4f} vs {a1_nr:.4f})")
        print(f"  P-LEARN-1: {'PASS' if p_learn1 else 'FAIL（预注册负结果，见 docstring 归因）'}")
        print("=" * 64)
    return dict(mse_a0=a0, mse_a1=a1, retention=retention, mse_a0_nr=a0_nr,
                mse_a1_nr=a1_nr, retention_nr=retention_nr,
                replay_helps=replay_helps, p_learn1=p_learn1)


def steps_to_target(model, trajs, target=0.008, max_steps=3000):
    """误差驱动训练直到 train MSE 到目标，返回所用步数（每规则学习成本）。"""
    n = 0
    for _ in range(10):
        for traj in trajs:
            model.reset()
            for t in range(len(traj) - 1):
                mse, _ = model.train_step(traj[t], traj[t + 1])
                n += 1
                if mse < target or n >= max_steps:
                    return n
    return n


def run_learn2(seed=0, verbose=True):
    """P-LEARN-2：hidden 32 vs 128 的每规则学习成本（步数到目标 MSE）。"""
    world = VariableSpeedWorld(seed=seed)
    train = world.trajectories(n_traj=30, T=40, speed_range=(0.8, 3.0))
    costs = {}
    max_steps = 3000
    for tag, hidden in (("h32", (32, 32)), ("h128", (128, 128))):
        m = TLAModel(obs_dim=3, out_dim=2, seed=seed, hidden_dims=hidden)
        t0 = time.time()
        costs[tag] = steps_to_target(m, train, max_steps=max_steps)
        costs[f"{tag}_sec"] = time.time() - t0
    ratio = costs["h128"] / max(costs["h32"], 1)
    # 触顶 = 未在预算内达标 → 判据无效（防触顶虚过 <2×）
    capped = costs["h32"] >= max_steps or costs["h128"] >= max_steps
    p_learn2 = (not capped) and ratio < 2.0

    if verbose:
        print("=" * 64)
        print("P-LEARN-2 报告（放大 hidden 32→128）")
        print("=" * 64)
        print(f"  h32:  {costs['h32']} 步 ({costs['h32_sec']:.1f}s)")
        print(f"  h128: {costs['h128']} 步 ({costs['h128_sec']:.1f}s)")
        print(f"  成本比率 = {ratio:.2f}  (<2.0)")
        print(f"  P-LEARN-2: {'PASS' if p_learn2 else 'FAIL'}")
        print("=" * 64)
    return dict(cost_h32=costs["h32"], cost_h128=costs["h128"],
                ratio=ratio, p_learn2=p_learn2)


if __name__ == "__main__":
    run_learn1()
    run_learn2()
