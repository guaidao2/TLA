"""原则一（摊销首猜 + 残差修正）三探针与解锁重测：预注册裁决复现。"""
import pytest
from tla.criteria.pr1 import run_pr1
from tla.criteria.pr1_retest import run_retest


@pytest.fixture(scope="module")
def pr1():
    return run_pr1(seed=0, n_epochs=2, n_traj=20, verbose=False)


@pytest.fixture(scope="module")
def retest():
    return run_retest(seed=0, verbose=False)


def test_pr1_learning_strength_solved(pr1):
    """探针 A：摊销首猜解决无捷径弱学习——分布内 MSE <0.02（解锁阈值达成）。"""
    assert pr1["p_A"], \
        f"学习强度应达标: indist={pr1['mse_indist']:.4f} 应 <0.02（vs 无捷径 0.11 / 捷径 0.004）"


def test_pr1_thinking_still_negative(pr1):
    """探针 B/C：琢磨增益仍未成立（困难输入上 settle 迭代无增益甚至负价值）。"""
    assert not pr1["p_B"], "琢磨增益应仍未成立（预注册负结果）"


def test_pr1_pcog3_retest_guess_best(retest):
    """P-COG-3 解锁重测：负结果确认——纯首猜（guess）在未见 ω 上最好。"""
    assert not retest["p_cog3"], "P-COG-3 重测应仍为负结果"
    assert retest["mse_guess_u"] <= retest["mse_ad_u"], \
        "纯首猜应不劣于自适应（琢磨负价值/过度精化）"


def test_pr1_plearn1_retest_negative(retest):
    """P-LEARN-1 解锁重测：负结果确认——摊销首猜 W_base 任务冲突导致遗忘依旧。"""
    assert not retest["p_learn1"], "P-LEARN-1 重测应仍为负结果（保留率 <95%）"
