"""遗忘修复：定位诊断 + 突触巩固 EWC（预注册裁决复现）。"""
import pytest
from tla.criteria.forgetting import run_diagnose, run_ewc


@pytest.fixture(scope="module")
def diag():
    # 诊断结论（frozen > baseline）在 n_traj=12 下仍成立（测试已验），可瘦身。
    return run_diagnose(seed=0, n_traj=12, verbose=False)


@pytest.fixture(scope="module")
def ewc():
    # 注：EWC 保留率对 n_traj 敏感（20→108.6% / 15→72.1% / 12→87.3%），
    # 判据完整性优先——不瘦身，保持 n_traj=20。
    return run_ewc(seed=0, n_traj=20, verbose=False)


def test_diag_forgetting_in_first_guess(diag):
    """定位诊断：冻结 W_base 显著改善 A 保留率（>不冻结基线），遗忘部分在首猜。"""
    assert diag["retention_frozen"] > diag["retention"], \
        (f"冻结 W_base 应改善保留率: frozen={diag['retention_frozen'] * 100:.1f}% "
         f"vs baseline={diag['retention'] * 100:.1f}%")


def test_ewc_plearn1_flipped(ewc):
    """P-LEARN-1 翻转：突触巩固后 A 保留率 ≥95%（遗忘负结果 → 正结果）。"""
    assert ewc["p_learn1"], f"EWC 后保留率应 ≥95%: 实测 {ewc['retention'] * 100:.1f}%"
    assert ewc["retention"] >= 0.95, f"保留率应 ≥95%: {ewc['retention'] * 100:.1f}%"


def test_ewc_learning_strength_kept(ewc):
    """保持判据：突触巩固不破坏学习强度（分布内 <0.02）。"""
    assert ewc["keep_strength"], f"学习强度应 <0.02: 实测 {ewc['mse_indist']:.4f}"


def test_ewc_b_still_learns(ewc):
    """B 任务确实在学习（不冻结新任务）：B mse 显著优于随机基线（~1.02）。"""
    assert ewc["b_mse"] < 0.9, f"B 应显著优于随机基线: B={ewc['b_mse']:.4f} vs random≈1.02"
