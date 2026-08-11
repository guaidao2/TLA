"""表示坍缩防护：机制 hook 测试（文本世界防预注册；默认关）。"""
import pytest
import torch
from tla.cognitive.pcn_amortized import AmortizedResidualPCN


def test_rep_stats_update_and_collapse_detection():
    """EMA 统计更新 + 坍缩检测（相对）：恒定输入方差 ≪ 随机输入方差（表示坍缩）。"""
    pcn = AmortizedResidualPCN(dims=[4, 8], out_dim=2, seed=0)
    x = torch.tensor([0.5, 0.2, 0.1, 0.0])
    t = torch.tensor([0.4, 0.3])
    pcn.rep_cov_reg = 0.0                       # 先关防护，只验统计
    for _ in range(100):
        pcn.settle(x, t, steps=2)               # 恒定输入 → 表示收敛到同一不动点
    var_constant = float(pcn._rep_var.mean())

    pcn2 = AmortizedResidualPCN(dims=[4, 8], out_dim=2, seed=0)
    pcn2.rep_cov_reg = 0.0
    for _ in range(100):
        pcn2.settle(torch.randn(4), t, steps=2)  # 随机输入 → 表示变化 → 方差大
    var_random = float(pcn2._rep_var.mean())

    assert pcn._rep_mean is not None and pcn._rep_var is not None
    assert var_constant < 0.1 * var_random, \
        f"恒定输入应使表示方差相对坍缩: const={var_constant:.2e} vs random={var_random:.2e}"


def test_anti_collapse_breaks_collapse_when_enabled():
    """防护激活：rep_cov_reg>0 且检测到坍缩时，μ 被推离（方差不再坍缩到 0）。"""
    pcn = AmortizedResidualPCN(dims=[4, 8], out_dim=2, seed=1)
    x = torch.tensor([0.5, 0.2, 0.1, 0.0])
    t = torch.tensor([0.4, 0.3])

    pcn_off = AmortizedResidualPCN(dims=[4, 8], out_dim=2, seed=1)
    pcn_off.rep_cov_reg = 0.0
    for _ in range(60):
        pcn_off.settle(x, t, steps=2)
    var_off = float(pcn_off._rep_var.mean())

    pcn_on = AmortizedResidualPCN(dims=[4, 8], out_dim=2, seed=1)
    pcn_on.rep_cov_reg = 0.1
    for _ in range(60):
        pcn_on.settle(x, t, steps=2)
    var_on = float(pcn_on._rep_var.mean())

    assert var_on > var_off, \
        f"防护应推离坍缩表示: var_on={var_on:.2e} vs var_off={var_off:.2e}"


def test_guard_off_by_default():
    """默认关闭：rep_cov_reg=0，不干扰正常训练。"""
    pcn = AmortizedResidualPCN(dims=[4, 8], out_dim=2, seed=2)
    assert pcn.rep_cov_reg == 0.0
