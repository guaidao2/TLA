"""琢磨失败回退首猜（双过程系统2→系统1兜底）：预注册裁决复现。"""
import pytest
from tla.criteria.pr1_fallback import run_fallback


@pytest.fixture(scope="module")
def fb():
    return run_fallback(seed=0, n_epochs=2, n_traj=20, verbose=False)


def test_fallback_never_worse_than_guess_on_competent_axes(fb):
    """有能力轴（分布内+噪声）：琢磨失败回退首猜，永不劣于瞎猜（原则核心）。"""
    assert fb["p_all"], \
        f"有能力轴上 fallback 应≤guess: " \
        f"in-dist {fb['rows']['in-dist']['guess']:.4f}→{fb['rows']['in-dist']['fallback']:.4f}, " \
        f"noise {fb['rows']['noise']['guess']:.4f}→{fb['rows']['noise']['fallback']:.4f}"


def test_fallback_cures_over_refinement(fb):
    """有能力轴上 fallback ≤ reasoned（治过度精化：噪声轴琢磨从负价值翻转为正）。"""
    assert fb["p_all_r"], \
        f"有能力轴上 fallback 应≤reasoned: " \
        f"noise reasoned={fb['rows']['noise']['reasoned']:.4f} vs fallback={fb['rows']['noise']['fallback']:.4f}"


def test_fallback_noise_axis_flips_positive(fb):
    """关键翻转锁定：噪声轴上 fallback 严格优于瞎猜（琢磨从负价值变正价值）。"""
    assert fb["rows"]["noise"]["fallback"] < fb["rows"]["noise"]["guess"], \
        f"噪声轴 fallback 应严格优于 guess: " \
        f"{fb['rows']['noise']['fallback']:.4f} vs {fb['rows']['noise']['guess']:.4f}"


def test_fallback_verdict_passes(fb):
    """总裁决：有能力轴原则成立（P-COG-3 负结果在有能力轴上被翻转）。"""
    assert fb["p_verdict"], "回退原则总裁决应为 PASS（有能力轴）"
