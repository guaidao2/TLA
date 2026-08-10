"""P-META-1~4：元层判据（生长门 / 校准期 / 修剪无损 / Self_Slot 行为增益）。"""
import torch
import pytest
from tla.meta.growth import CapacityManager
from tla.model import TLAModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


# ---------- P-META-1：生长门三信号 ----------
def test_pmeta1_grow_requires_all_three_signals():
    """生长 ⟺ (惊奇∧新奇∧能量)，缺一不可。"""
    cm = CapacityManager(n_units=8, grow_interval=1, grow_thresholds=(0.3, 0.3, 0.3))
    cm.mask[0] = False                       # 一个休眠单元可生长
    cm.imp.imp[0] = 0.0                      # 它 importance 最低 → 应被选中
    cm.update(torch.zeros(8), torch.zeros(8), 0.5, 0.6)   # tick=1

    assert cm.maybe_grow(0.1, 0.5, 0.6) is None, "惊奇低 → 不生长"
    assert cm.maybe_grow(0.5, 0.1, 0.6) is None, "新奇低 → 不生长"
    assert cm.maybe_grow(0.5, 0.5, 0.1) is None, "能量低 → 不生长"
    assert cm.maybe_grow(0.5, 0.5, 0.6) == 0, "三信号齐 → 生长休眠单元 0"


# ---------- P-META-2：新神经元校准期 ----------
def test_pmeta2_new_unit_calibration_ramp():
    """新单元低增益(0.3)起步、校准期内 ramp 到 1.0；旧单元增益逐位不变。"""
    cm = CapacityManager(n_units=4, calib_min=0.3, calib_window=100, grow_interval=1)
    cm.mask[1] = False
    cm.update(torch.zeros(4), torch.zeros(4), 0.5, 0.6)
    idx = cm.maybe_grow(0.5, 0.5, 0.6)
    assert idx == 1
    assert cm.gain[idx] == pytest.approx(0.3, abs=1e-6), "新单元应以低增益起步（防噪声污染）"
    assert cm.gain[0] == pytest.approx(1.0, abs=1e-6), "旧单元增益逐位一致（不受生长影响）"

    for _ in range(50):
        cm.update(torch.zeros(4), torch.zeros(4), 0.5, 0.6)
    assert cm.gain[idx] < 1.0, "校准期未完增益应 <1.0"
    for _ in range(60):
        cm.update(torch.zeros(4), torch.zeros(4), 0.5, 0.6)
    assert cm.gain[idx] == pytest.approx(1.0, abs=1e-2), "校准期结束增益应升到 1.0"


# ---------- P-META-3：修剪低 importance 无损 ----------
def _train_model(seed=0, epochs=2):
    world = VariableSpeedWorld(seed=seed)
    train = world.trajectories(n_traj=20, T=30, speed_range=(0.8, 3.0))
    test = world.trajectories(n_traj=4, T=20, speed_range=(1.0, 2.0), seed=7)
    model = TLAModel(obs_dim=3, out_dim=2, seed=seed)
    for _ in range(epochs):
        for traj in train:
            model.reset()
            for t in range(len(traj) - 1):
                model.train_step(traj[t], traj[t + 1])
    return model, test


def _eval(model, trajs):
    mses = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            pred, info = model.infer(traj[t])
            if pred is not None:
                mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def test_pmeta3_prune_low_importance_no_degradation():
    """修剪 empirically 低 importance 的 20% 单元后，测试精度不降（容量释放无代价）。"""
    model, test = _train_model(seed=3)
    mse_before = _eval(model, test)

    n = model.meta.n
    k = max(1, n // 5)
    n_active_before = int(model.meta.mask.sum())
    active = torch.where(model.meta.mask)[0]
    bottom = active[torch.argsort(model.meta.imp.imp[active])[:k]]
    model.meta.mask[bottom] = False
    model.meta.gain[bottom] = 0.0
    assert int(model.meta.mask.sum()) == n_active_before - k, "应恰好释放 k 个单元容量"

    mse_after = _eval(model, test)
    assert mse_after <= mse_before * 1.05, \
        f"修剪低 importance 不应降精度: before={mse_before:.4f} after={mse_after:.4f}"


# ---------- P-META-4：Self_Slot 行为增益 ----------
def test_pmeta4_self_slot_participation_not_worse():
    """Self_Slot 一致性门参与决策后，噪声输入精度不劣化（≤1.05×，'行为更稳'的弱形式）。

    预注册预案：若无行为差异 → 降级为表征能力记录（损失下降已由 test_self_slot_loss_decreases 覆盖）。
    """
    model, test = _train_model(seed=5)
    gen = torch.Generator().manual_seed(11)
    noisy = []
    for traj in test:
        model.reset()
        for t in range(len(traj) - 1):
            obs = traj[t] + torch.randn(3, generator=gen) * 0.3
            pred, _ = model.infer(obs, self_consistency_gate=True)
            if pred is not None:
                noisy.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    mse_gate_on = float(torch.tensor(noisy).mean())
    gen2 = torch.Generator().manual_seed(11)
    noisy_off = []
    for traj in test:
        model.reset()
        for t in range(len(traj) - 1):
            obs = traj[t] + torch.randn(3, generator=gen2) * 0.3
            pred, _ = model.infer(obs, self_consistency_gate=False)
            if pred is not None:
                noisy_off.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    mse_gate_off = float(torch.tensor(noisy_off).mean())
    assert mse_gate_on <= mse_gate_off * 1.05, \
        f"一致性门参与决策不应劣化: on={mse_gate_on:.4f} off={mse_gate_off:.4f}"
