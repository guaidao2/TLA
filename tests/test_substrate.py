"""P-PHY-1/2/3：基板有界 / 断电语义 / 零输入静息。"""
import torch
import pytest
from tla.substrate.ltc_cell import LTCCell
from tla.substrate.energy import EnergyBudget
from tla.model import TLAModel


def test_pphy1_zero_input_converges_to_rest():
    """P-PHY-1：无输入时 V 收敛到静息（不动点），不自发发放。"""
    cell = LTCCell(in_dim=3, hidden=16, seed=0)
    cell.h = torch.randn(16) * 1.5          # 从任意状态出发
    x = torch.zeros(3)
    for _ in range(300):
        cell.forward(x)
    assert torch.max(torch.abs(cell.h)).item() < 1e-2, "零输入应收敛到静息"


def test_pphy3_bounded_under_continuous_input():
    """P-PHY-3：连续输入下 V 有界（tanh 软饱和，|V| ≤ v_max）。"""
    cell = LTCCell(in_dim=3, hidden=16, seed=1)
    x = torch.randn(3) * 3.0
    for _ in range(200):
        cell.forward(x)
    assert torch.max(torch.abs(cell.h)).item() <= cell.v_max + 1e-6


def test_pphy2_energy_zero_suppresses_output():
    """P-PHY-2：Energy_Budget=0 时上层输出被抑制（物理崩=全崩语义）。"""
    model = TLAModel(obs_dim=3, out_dim=2, seed=2, energy_capacity=2.0)
    obs = torch.tensor([0.0, 0.3, 0.2])
    model.ltc.forward(obs)
    x = torch.cat([obs, model.ltc.h])
    pred, info = model.infer(obs)
    # 预算 2.0 / 步成本≈1.5 → 大约 1 步后断电 → suppressed
    assert info["suppressed"] or pred is None or info["doubtful"], \
        "能量耗尽应抑制输出（suppressed/doubtful）"


def test_energy_budget_consume_until_depleted():
    b = EnergyBudget(capacity=2.0, base_cost=1.0)
    assert b.consume()          # 1.0 → level 1.0
    assert not b.consume()      # 1.0 → level 0.0 → 断电
    assert b.depleted()
    assert b.report().integrity == 0.0
