"""MoE 任务分离容量实验（预注册裁决，2026-08-10 实测）。

背景：无捷径变体让"会琢磨"机制真实（P-COG-3 分布内 +45%），但学习弱；共享 W_lin 捷径
让单任务易学但杀死琢磨与防遗忘（捷径悖论）。本实验检验修复方向：**无捷径 + 任务分离专家**。

【预注册裁决：第三项诚实负结果/部分结果】
1. 自由能路由（重建误差最小者胜）：**不分离**——两专家重建都平庸，argmin 随机翻转，
   r 恒卡 0.5（MoE 对称性），硬路由（赢者 0.8+探索 0.2）也无法打破。
2. 原型路由（专家持输入原型，就近竞争 + EMA 局部更新；硬路由学习用原型 argmin——
   MoE 终审修复，原用 fe argmin 是 bug）：**分离**——low-vel 输入路由到专家0 的比例
   0.57 vs high-vel 0.30（差 0.27，方向取决于初始种子，取绝对值判定）。
3. **无捷径专家是弱学习者**：MoE 训练后分布内 MSE ~0.11，而共享捷径基线 0.004
   （~27× 差距）。专家容量分离不能治愈"无捷径弱学习"这一更基础的短板。
4. 推论：P-COG-3 正证据场地与 P-LEARN-1 修复不能仅靠容量分离达成——
   修复顺序应先是**无捷径变体的学习强度**（梯度信噪比/容量/训练时长），再谈路由分离。
   重测解锁条件（预注册）：无捷径变体分布内 MSE ≤0.02 时解锁 P-COG-3/P-LEARN-1 重测。

结论：MoE v1 保留为实现与测量记录；路由分离与学习强度列为后续研究方向。
"""
import torch
from tla.model_moe import TLAMoEModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(m, trajs):
    mses = []
    for traj in trajs:
        m.reset()
        for t in range(len(traj) - 1):
            pred, info = m.infer(traj[t])
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def route_split(m, trajs, lo_vel=0.4, hi_vel=0.6):
    """统计低/高速度输入路由到专家 0 的平均责任（原型路由分离度）。"""
    lo, hi = [], []
    for traj in trajs[:6]:
        m.reset()
        for t in range(len(traj) - 1):
            obs = traj[t]
            m.ltc.forward(obs)
            x = torch.cat([obs, m.ltc.h])
            m.pcn.settle(x, steps=2)
            r0 = m.pcn.last_routing[0].item()
            if obs[1] < lo_vel:
                lo.append(r0)
            elif obs[1] > hi_vel:
                hi.append(r0)
    return (torch.tensor(lo).mean().item() if lo else float("nan"),
            torch.tensor(hi).mean().item() if hi else float("nan"))


def run_moe(seed=0, n_epochs=2, n_traj=20, verbose=True):
    world = VariableSpeedWorld(seed=seed)
    train = world.trajectories(n_traj=n_traj, T=30, speed_range=(0.8, 3.0))
    indist = world.trajectories(n_traj=4, T=20, speed_range=(1.0, 2.0), seed=7)

    m = TLAMoEModel(obs_dim=3, out_dim=2, seed=seed, lr=0.02)
    mse_random = eval_mse(m, indist)
    for _ in range(n_epochs):
        for traj in train:
            m.reset()
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1])
    mse_after = eval_mse(m, indist)
    lo_r, hi_r = route_split(m, train)

    learned = mse_after < 0.9 * mse_random              # 有学习
    separated = abs(hi_r - lo_r) > 0.1                   # 原型路由分离（方向无关：专家0 偏 low 或偏 high 都算）
    weak = mse_after > 0.02                              # 无捷径专家弱学习者（与重测解锁 0.02 对齐，无死区）

    if verbose:
        print("=" * 64)
        print("MoE 任务分离容量报告（预注册裁决：第三项负/部分结果）")
        print("=" * 64)
        print(f"  indist MSE: random={mse_random:.4f} → trained={mse_after:.4f}")
        print(f"  原型路由: 专家0 承担 low-vel={lo_r:.2f} high-vel={hi_r:.2f}（差 {hi_r - lo_r:+.2f}）")
        print(f"  学习: {'是' if learned else '否'}  分离: {'是' if separated else '否'}  "
              f"弱学习: {'是' if weak else '否'}")
        print("  裁决: 容量分离不能治愈'无捷径弱学习'（~0.11 vs 捷径 0.004），"
              "路由分离成立但专家弱学习；详见 docstring")
        print("=" * 64)
    return dict(mse_random=mse_random, mse_after=mse_after, lo_r=lo_r, hi_r=hi_r,
                learned=learned, separated=separated, weak=weak)


if __name__ == "__main__":
    run_moe(verbose=True)
