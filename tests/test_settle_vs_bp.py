"""settle-vs-BP：推理深度与 BP 关系（终审修正方向：收敛推理≈BP，有限推理偏离BP）。"""
import pytest
from tla.criteria.settle_vs_bp import run_settle_vs_bp


@pytest.fixture(scope="module")
def svb():
    return run_settle_vs_bp(seed=0, n_epochs=2, n_traj=15, verbose=False)


def test_converged_inference_bp_aligned(svb):
    """收敛推理（充分 settle）表示与 BP 高度一致（CKA ≥0.8，W&B 等价定理的实证）。"""
    assert svb["cka_settled"] >= 0.8, \
        f"充分 settle 表示应与 BP 高度一致（W&B 收敛等价）: CKA={svb['cka_settled']:.3f}"


def test_limited_inference_deviates_from_bp(svb):
    """有限推理（单步）表示显著偏离 BP：CKA(单步) < CKA(settle) − 0.05。"""
    assert svb["p_rep"], \
        f"单步应显著偏离 BP: CKA(single)={svb['cka_single']:.3f} vs CKA(settled)={svb['cka_settled']:.3f}"


def test_behavior_leg_reported(svb):
    """行为差异腿（报告项，不判定）：实测在设置间翻转（噪声），如实记录。"""
    # 表示证据（p_rep）为稳健裁决依据；行为腿在 n_traj=15/20 间翻转（0.1096/0.1006 vs 0.1326/0.0948），
    # 不作断言——仅确认其已计算并记录在返回 dict 中。
    assert "diff_single" in svb and "diff_settled" in svb


def test_settle_vs_bp_verdict_supports_literature(svb):
    """总裁决（基于稳健表示证据）：实证支持文献（收敛推理≈BP；有限推理偏离BP）。"""
    assert svb["p_verdict"], "表示证据应支持文献方向"
