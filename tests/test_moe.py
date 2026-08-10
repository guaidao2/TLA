"""MoE 任务分离容量：预注册裁决的复现测试（第三项诚实负/部分结果）。"""
import pytest
from tla.criteria.moe import run_moe


@pytest.fixture(scope="module")
def moe():
    return run_moe(seed=0, n_epochs=2, n_traj=15, verbose=False)


def test_moe_learns_something(moe):
    """MoE（无捷径专家）确实有学习：分布内 MSE 优于随机初始化。"""
    assert moe["learned"], \
        f"MoE 应有学习: trained={moe['mse_after']:.4f} vs random={moe['mse_random']:.4f}"


def test_moe_prototype_routing_partially_separates(moe):
    """原型路由部分分离：高低速输入路由到专家 0 的责任有偏（差 >0.1）。"""
    assert moe["separated"], \
        f"原型路由应部分分离: lo={moe['lo_r']:.2f} hi={moe['hi_r']:.2f}"


def test_moe_weak_learner_negative_recorded(moe):
    """第三项负结果复现：无捷径专家是弱学习者（MSE 显著高于共享捷径基线 0.004）。"""
    assert moe["weak"], f"无捷径专家应弱学习（记录负结果）: mse={moe['mse_after']:.4f}"
