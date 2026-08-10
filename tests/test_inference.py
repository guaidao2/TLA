"""P-COG-1：干净输入内循环少步即停（防 overthinking）；doubtful 标记。"""
import torch
from tla.model import TLAModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


def _train(model, trajs, epochs=2):
    for _ in range(epochs):
        for traj in trajs:
            model.reset()
            for t in range(len(traj) - 1):
                model.train_step(traj[t], traj[t + 1])


def test_pcog1_clean_input_few_steps():
    """P-COG-1/2 行为签名：训练后干净输入 ≤2 步；噪声输入分配更多步（会琢磨的差分特征）。

    不用未训练基线做 < 断言：未训练模型停止条件同样触发（输出收敛停止对垃圾模型也生效），
    真正的签名是训练模型对输入难度的**差分响应**（噪声 > 干净）。
    """
    world = VariableSpeedWorld(seed=3)
    train_trajs = world.trajectories(n_traj=30, T=40, speed_range=(0.8, 3.0))
    test_trajs = world.trajectories(n_traj=5, T=30, speed_range=(1.0, 1.5), seed=7)
    model = TLAModel(obs_dim=3, out_dim=2, seed=3, infer_max_steps=8, lr_inf=0.2)
    noise_gen = torch.Generator().manual_seed(1234)   # 噪声用本地种子，防测试偶发
    _train(model, train_trajs)

    def median_steps(m, noisy=False):
        steps = []
        for traj in test_trajs:
            m.reset()
            for t in range(len(traj) - 1):
                obs = traj[t]
                if noisy:
                    obs = obs + torch.randn(3, generator=noise_gen) * 0.3
                _, info = m.infer(obs)
                if not info["suppressed"]:
                    steps.append(info["steps"])
        return float(torch.tensor(steps).median().item())

    med_clean, med_noisy = median_steps(model), median_steps(model, noisy=True)
    assert med_clean <= 2, f"干净输入应少步即停，实测 median={med_clean}"
    assert med_noisy > med_clean, \
        f"噪声输入应分配更多步（会琢磨）: clean={med_clean} vs noisy={med_noisy}"


def test_doubtful_when_budget_exhausted():
    """步数上限内不收敛 → doubtful 标记（不硬猜）。"""
    model = TLAModel(obs_dim=3, out_dim=2, seed=4, infer_max_steps=1, infer_tol=1e-9)
    obs = torch.tensor([0.1, 0.2, 0.3])
    model.ltc.forward(obs)
    _, info = model.infer(obs)
    assert info["doubtful"], "1 步内达不到极苛刻 tol 应标 doubtful"
