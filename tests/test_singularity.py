"""奇点神经元 SN-1..6 锁死测试。

实测裁决（2026-08-11，见 tla/criteria/singularity.py 与 docs/奇点神经元_数学推导.md §10）：
- 原文 e^{βI}：SN1/5 PASS、SN2/3/4/6 FAIL——增长项基线恒热（I=0 时 e^0=1 仍有
  α(h+ε)(1−h) 生长，不动点 ~0.902，幽灵态 0.02 非全局吸引子；弱输入终值 ~0.90、
  噪声激活 98.5%）。推导 v0.1 缺陷，判据未改；
- 修复变体 e^{βI}−1（input_gated）：SN-1..6 全 PASS（起爆 14 tick、幽灵态 0.020、
  解码误差 2.59%、弱输入 0.02 终值 < 2h*、噪声激活 0%/强 99.9%）。

判据锁死：跑数前不改判据，只改代码——原文负结果与修复变体正结果同样锁死。
"""
import pytest
from tla.substrate.singularity_cell import SingularityCell
from tla.criteria import singularity


@pytest.fixture(scope="module")
def results():
    return singularity.run_all(verbose=False)


def test_gated_all_criteria_pass(results):
    """修复变体（input_gated）SN-1..6 全 PASS。"""
    g = results["gated e^{βI}−1"]
    for i in range(1, 7):
        assert g[f"sn{i}"], f"gated SN{i} 应 PASS"


def test_gated_sn3_time_decode_accurate(results):
    """SN-3 时间距离解码：误差 <5%（时间戳 claim 的核心，含暗能量修正反演）。"""
    g = results["gated e^{βI}−1"]
    assert len(g["sn3_errors"]) >= 3
    assert max(g["sn3_errors"]) < 0.05


def test_original_hot_baseline_negative_locked(results):
    """原文变体负结果锁死：SN-2/4/6 FAIL（恒热）——增长基线缺陷。
    若未来推导修复使原文也达标，此测试失败 → 强制重新评估。"""
    o = results["original e^{βI}"]
    assert not o["sn2"] and not o["sn4"] and not o["sn6"]
    assert o["h_noinput"] > 0.5            # 无输入仍恒热（对照 h* = 0.02）


def test_sn5_boundedness_both(results):
    """SN-5 有界性：两变体均 PASS（[0, h_max] 不变集数值验证）。"""
    assert results["original e^{βI}"]["sn5"]
    assert results["gated e^{βI}−1"]["sn5"]


def test_semi_implicit_ghost_at_no_input():
    """半隐式（γ=1，线性化）在 I=0 收敛到幽灵态 h*——与欧拉一致。
    注：半隐式把饱和项 (1−h/h_max) 当常数，只在小 h 区域有效（大 I 下平衡点
    与欧拉不同——推导 v0.1 的已知局限，诚实标注）。"""
    c = SingularityCell(input_gated=True)
    for _ in range(2000):
        c.step_semi_implicit(0.0)
    assert abs(c.h - c.ghost()) / c.ghost() < 0.05
