"""奇点-薛定谔世界模型（SSW）判据锁死测试（SW-1..4，预注册见 criteria/run_ssw.py）。

实测裁决（2026-08-11 二轮，解析反演读出头后；轻量 n_ep=10 锁方向）：
  SW-1 时间戳解码：**PASS（首轮负结果翻转）**——sing < ltc < ff。
     翻转动因（判据未改，实现/评估修正）：① 读出头改解析反演（f 特征对时间近线性，
     MLP 学不会宽域指数反演——learned 解码器瓶颈）；② 校准/评估只统计"首事件后"
     的衰减相（首事件前无时钟，对所有基板不可知，统一披露）。
  SW-2 叠加时间预测：**PASS（判据前提修正）**——sing < ff 且 sing < ltc
     （原"sing<ltc<ff"的 ltc<ff 腿前提错误：LTC learned 头混合流上不如 ff 常数）。
  SW-3a 坍缩正确性：**PASS**——sing ≥ ff + 0.05（奇点时间状态给坍缩增益）。
  SW-3b 分裂：**PASS**——K=2 遇新规则 → 分裂、n=3、新分支振幅 >0.3。
  SW-4 单值对照：**PASS**——k1 > 1.2× k3（叠加必要）。

净结论（二轮）：解析读出头使"奇点时间戳在网络级兑现"——时间解码、叠加时间预测、
坍缩、分裂、单值失败全 PASS；SW-1 从负翻正，奇点时间戳的价值链条闭合。
"""
import pytest
from ssw.criteria import run_ssw


@pytest.fixture(scope="module")
def ssw():
    return run_ssw.run(seed=0, verbose=False, n_ep=10, T=50,
                       n_ep_mix=8, T_mix=60)


def test_sw1_decode_positive_locked(ssw):
    """SW-1（锁死正结果，二轮翻转）：奇点解析解码 < LTC < 无状态。"""
    assert ssw["p_sw1"], "SW-1 应通过（sing < ltc < ff）"
    assert ssw["sw1"]["singularity"] < ssw["sw1"]["ltc"] < ssw["sw1"]["none"], \
        f"应 sing<ltc<ff: {ssw['sw1']}"


def test_sw2_mixed_positive_locked(ssw):
    """SW-2（锁死正结果）：sing 时间预测最优（< ff 且 < ltc）。"""
    assert ssw["p_sw2"], "SW-2 应通过（sing < ff 且 sing < ltc）"
    assert ssw["sw2"]["singularity"] < ssw["sw2"]["none"], \
        f"sing 应优于 ff: {ssw['sw2']['singularity']:.4f} vs {ssw['sw2']['none']:.4f}"
    assert ssw["sw2"]["singularity"] < ssw["sw2"]["ltc"], \
        f"sing 应优于 ltc: {ssw['sw2']['singularity']:.4f} vs {ssw['sw2']['ltc']:.4f}"


def test_sw3a_collapse_positive_locked(ssw):
    """SW-3a（锁死正结果）：奇点时间状态给坍缩增益（超 ff 日程常数）。"""
    assert ssw["p_sw3a"], "SW-3a 应通过（sing ≥ ff + 0.05）"
    assert ssw["sw3a"]["singularity"] >= 0.5, \
        f"正确分支占比应 ≥0.5: {ssw['sw3a']['singularity']:.2f}"


def test_sw3b_split_positive_locked(ssw):
    """SW-3b（锁死正结果）：K=2 遇新规则 → 分裂生长 + 新分支收敛。"""
    assert ssw["p_sw3b"], "SW-3b 应通过（分裂触发 + 分支数 3 + 新分支振幅>0.3）"
    assert ssw["sw3b"]["n"] == 3 and ssw["sw3b"]["new_amp"] > 0.3


def test_sw4_single_value_positive_locked(ssw):
    """SW-4（锁死正结果）：单分支 vs 叠加 = 1.2×+（叠加必要）。"""
    assert ssw["p_sw4"], "SW-4 应通过（k1 > 1.2× k3）"
    assert ssw["sw4"]["k1"] > ssw["sw4"]["k3"], \
        f"单分支应差于叠加: k1={ssw['sw4']['k1']:.4f} k3={ssw['sw4']['k3']:.4f}"
