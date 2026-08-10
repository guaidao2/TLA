"""P-COG-3 + P-COG-5：预注册裁决的复现测试（锁死负结果与机制正证据）。

全量消融只跑一次（模块级 fixture），5 个测试共享结果，避免重复训练超时。
"""
import pytest
import torch
from tla.criteria.ablation import run_ablation


@pytest.fixture(scope="module")
def ablation_result():
    return run_ablation(seed=0, verbose=False, n_epochs=2, n_traj=25, T=40)


def test_pcog3_negative_reproducible(ablation_result):
    """P-COG-3 负结果复现：有捷径版未见 ω 泛化上自适应增益 <10%（循环空转）。"""
    r = ablation_result
    assert r["gain_lin_unseen"] < 0.10, \
        f"预注册负结果应复现: 有捷径未见ω gain={r['gain_lin_unseen'] * 100:.1f}% 应 <10%"


def test_pcog3_ood_over_refinement_reproducible(ablation_result):
    """归因②复现：无捷径版未见 ω 上自适应不优于固定1步（过度精化，gain ≤ 5%）。"""
    r = ablation_result
    assert r["gain_nolin_unseen"] <= 0.05, \
        f"无捷径未见ω应无增益(过度精化): gain={r['gain_nolin_unseen'] * 100:.1f}%"


def test_pcog3_mechanism_real_in_competence(ablation_result):
    """机制正证据复现：无捷径版分布内自适应显著优于固定1步（≥20%，'会琢磨'在承重时真实）。"""
    r = ablation_result
    assert r["gain_nolin_indist"] > 0.20, \
        f"无捷径分布内应显著增益: gain={r['gain_nolin_indist'] * 100:.1f}%"


def test_pcog5_see_saw(ablation_result):
    """P-COG-5 跷跷板复现：有捷径版初猜≈自适应（空转镜像）；无捷径版初猜差≥2×（摆烂镜像）。"""
    r = ablation_result
    assert r["guess_ratio_lin_unseen"] <= 1.05, \
        f"有捷径 guess≈adaptive（空转）: ratio={r['guess_ratio_lin_unseen']:.2f}"
    assert r["guess_ratio_nolin_indist"] >= 2.0, \
        f"无捷径 guess 显著差于 adaptive（摆烂）: ratio={r['guess_ratio_nolin_indist']:.2f}"


def test_ablation_numbers_finite(ablation_result):
    """数字健全性：所有 MSE 有限。"""
    for k, v in ablation_result.items():
        if k.startswith(("lin", "nolin")):
            assert torch.isfinite(torch.tensor(v)), f"{k}={v} 应为有限值"
