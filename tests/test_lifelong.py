"""P-LEARN-1/2：终身学习判据（预注册裁决的复现测试）。

run_learn1/run_learn2 各跑一次（模块级 fixture）共享，避免重复训练超时。
"""
import pytest
import torch
from tla.criteria.lifelong import run_learn1, run_learn2


@pytest.fixture(scope="module")
def learn1():
    return run_learn1(seed=0, verbose=False)


@pytest.fixture(scope="module")
def learn2():
    return run_learn2(seed=0, verbose=False)


def test_plearn1_negative_reproducible(learn1):
    """P-LEARN-1 负结果复现：保留率 <95%（锁死预注册负结果，不硬凑）。"""
    assert learn1["retention"] < 0.95, \
        f"预注册负结果应复现: 保留率={learn1['retention'] * 100:.1f}% 应 <95%"


def test_plearn1_replay_helps_absolutely(learn1):
    """负结果的正证据复现：重放确实绝对缓解遗忘（A 遗忘后 mse 低于无重放）。"""
    assert learn1["replay_helps"], \
        f"重放应绝对缓解遗忘: replay={learn1['mse_a1']:.4f} vs noreplay={learn1['mse_a1_nr']:.4f}"


def test_plearn1_catastrophic_forgetting_baseline(learn1):
    """无重放对照应显示灾难性遗忘（保留率 <50%，任务分离是有效的）。"""
    assert learn1["retention_nr"] < 0.50, \
        f"无重放应灾难性遗忘: 保留率={learn1['retention_nr'] * 100:.1f}% 应 <50%"


def test_plearn2_scale_cost_not_collapse(learn2):
    """P-LEARN-2：hidden 32→128 每规则学习成本 <2×（且未触顶，防虚过）。"""
    assert not learn2["capped"], "任一规模触顶(3000)未达标 → 判据无效，不得虚过"
    assert learn2["p_learn2"], f"放大成本不应崩溃: 比率={learn2['ratio']:.2f}"
    assert learn2["cost_h32"] > 0
