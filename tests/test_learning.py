"""P-LEARN-3 探针：无 BP 局部误差驱动能否学得动；Self_Slot 自监督损失下降；CLS 重放可用。"""
import torch
from tla.model import TLAModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


def _eval_mse(model, trajs):
    mses = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            pred, info = model.infer(traj[t])
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def test_plearn3_error_driven_learning_happens():
    """误差驱动学习（无 BP/autograd）在未见 ω 上显著优于随机初始化基线。"""
    world = VariableSpeedWorld(seed=5)
    train_trajs = world.trajectories(n_traj=20, T=30, speed_range=(0.8, 3.0))
    test_trajs = world.trajectories(n_traj=5, T=30, speed_range=(4.0, 5.0), seed=8)

    model = TLAModel(obs_dim=3, out_dim=2, seed=5)
    base = TLAModel(obs_dim=3, out_dim=2, seed=55)

    mse_before = _eval_mse(base, test_trajs)
    for traj in train_trajs:
        model.reset()
        for t in range(len(traj) - 1):
            model.train_step(traj[t], traj[t + 1])
    mse_after = _eval_mse(model, test_trajs)

    assert mse_after < 0.7 * mse_before, \
        f"误差驱动学习应显著优于随机基线: after={mse_after:.4f} before={mse_before:.4f}"


def test_no_backprop_in_learning_loop():
    """纪律检查：组合体主学习环不依赖 autograd（无 .backward() / .grad）。"""
    import inspect
    from tla.cognitive import learning
    from tla.cognitive import pcn_stack
    src = inspect.getsource(learning) + inspect.getsource(pcn_stack)
    assert ".backward(" not in src and "requires_grad" not in src, \
        "主网络学习环必须纯局部误差驱动（无 BP）"


def test_self_slot_loss_decreases():
    """Self_Slot 自监督损失随训练下降（⑦ 学到'我下一刻会输出什么'）。"""
    world = VariableSpeedWorld(seed=6)
    train_trajs = world.trajectories(n_traj=20, T=30, speed_range=(0.8, 3.0))
    model = TLAModel(obs_dim=3, out_dim=2, seed=6)

    losses = []
    for traj in train_trajs:
        model.reset()
        for t in range(len(traj) - 1):
            _, ss_loss = model.train_step(traj[t], traj[t + 1])
            if ss_loss is not None:
                losses.append(ss_loss)
    head, tail = torch.tensor(losses[:50]).mean(), torch.tensor(losses[-50:]).mean()
    assert tail < head, f"Self_Slot 损失应下降: head={head:.4f} tail={tail:.4f}"


def test_cls_replay_buffer_works():
    """CLS 重放缓冲 push/len 可用，重放路径不炸。"""
    from tla.learning.cls_replay import ReplayBuffer
    world = VariableSpeedWorld(seed=9)
    trajs = world.trajectories(n_traj=2, T=20, speed_range=(0.8, 3.0))
    buf = ReplayBuffer(capacity=64, replay_prob=0.0, batch=4, seed=0)
    for traj in trajs:
        for t in range(len(traj) - 1):
            buf.push(traj[t], traj[t + 1], torch.zeros(16), surprise=1.0)
    assert len(buf) == sum(len(tr) - 1 for tr in trajs)
    assert buf.items[0][0].shape == torch.Size([3])
