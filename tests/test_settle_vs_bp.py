"""settle-vs-BP：预注册负结果复现（弹簧任务上未观测到离开 W&B 设定的可区分证据）。"""
import pytest
from tla.criteria.settle_vs_bp import run_settle_vs_bp


@pytest.fixture(scope="module")
def svb():
    return run_settle_vs_bp(seed=0, n_epochs=2, n_traj=15, verbose=False)


def test_settle_representation_bp_aligned(svb):
    """负结果复现：充分 settle 的表示与 BP 高度一致（CKA ≥0.8，未更偏离）。"""
    assert svb["cka_settled"] >= 0.8, \
        f"充分 settle 表示应与 BP 高度一致（负结果）: CKA={svb['cka_settled']:.3f}"


def test_settle_not_less_bp_aligned_than_single(svb):
    """判据反证复现：充分 settle 的 CKA 不低于单步（未出现'更偏离 BP'）。"""
    assert svb["cka_settled"] >= svb["cka_single"] - 0.05, \
        f"充分 settle 不应比单步更偏离 BP: settled={svb['cka_settled']:.3f} vs single={svb['cka_single']:.3f}"


def test_behavior_not_distinguishable(svb):
    """行为不可区分复现：|settle−bp| 不显著大于 |单步−bp|。"""
    assert svb["diff_settled"] <= svb["diff_single"] * 1.2 + 1e-6, \
        f"行为差异不应区分: |settle−bp|={svb['diff_settled']:.4f} vs |单步−bp|={svb['diff_single']:.4f}"


def test_settle_vs_bp_verdict_negative(svb):
    """总裁决：判据未通过（负结果如实记录，场地归因见论文 §4.5）。"""
    assert not svb["p_verdict"], "当前任务上应观测到负结果（未离开 W&B 设定）"
