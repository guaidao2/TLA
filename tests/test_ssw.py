"""奇点-薛定谔世界模型（SSW）判据锁死测试（SW-1..4，预注册见 criteria/run_ssw.py）。

实测裁决（2026-08-11，n_ep=10/T=50/n_ep_mix=8/T_mix=60，轻量锁方向）：
  SW-1 时间戳解码：**FAIL（锁死负结果）**——sing 0.0762 vs ltc 0.0274 vs ff 0.0297；
     奇点银行 slow 规则宽域解码（20±5 tick）学不动（MLP 解码器是瓶颈，非基板——
     基板解码能力已由奇点论文 SN-3 用解析反演单独验证 2.59%）。
  SW-2 叠加时间预测：**FAIL（锁死，边缘）**——sing 0.0129 < ltc 0.0166 ✓ 但
     ff 0.0123 边际最优（sing 差 5%）；日程常数已够用，learned 解码未超。
  SW-3a 坍缩正确性：**PASS**——sing 0.86 ≥ ff 0.71 + 0.05（奇点时间状态在日程常数
     之上给坍缩额外准确度；判据前提曾修正：ff 靠常数也能坍缩，见 runner 文档）。
  SW-3b 分裂：**PASS**——K=2 遇 slow → 分裂触发（t≈9）、n=3、新分支振幅 0.72；
     公平设定=专家头冻结（否则在线漂移适应 slow 取代分裂，见 runner 文档）。
  SW-4 单值对照：**PASS**——k1 0.0380 vs k3 0.0129 = 2.95×（叠加必要）。

净结论：叠加机制（坍缩/分裂/单值失败）全验证；奇点时间状态给坍缩真实增益
（SW-3a），但时间解码没超日程常数（SW-1/2 负——learned 解码器瓶颈）。
"""
import pytest
from ssw.criteria import run_ssw


@pytest.fixture(scope="module")
def ssw():
    return run_ssw.run(seed=0, verbose=False, n_ep=10, T=50,
                       n_ep_mix=8, T_mix=60)


def test_sw1_decode_negative_locked(ssw):
    """SW-1（锁死负结果）：learned 时间解码未超 ff 日程常数。若未来解码器改进
    使 sing < ff，此测试失败 → 强制重评。"""
    assert not ssw["p_sw1"], "SW-1 应仍为负结果（sing 解码未超 ff 常数）"
    assert ssw["sw1"]["singularity"] > ssw["sw1"]["none"], \
        f"sing 应仍差于 ff: {ssw['sw1']['singularity']:.4f} vs {ssw['sw1']['none']:.4f}"


def test_sw2_mixed_negative_locked(ssw):
    """SW-2（锁死负结果，边缘）：sing<ltc 方向成立但未超 ff。"""
    assert not ssw["p_sw2"], "SW-2 应仍为负结果"
    assert ssw["sw2"]["singularity"] < ssw["sw2"]["ltc"], \
        f"sing 应仍优于 ltc: {ssw['sw2']['singularity']:.4f} vs {ssw['sw2']['ltc']:.4f}"


def test_sw3a_collapse_positive_locked(ssw):
    """SW-3a（锁死正结果）：奇点时间状态给坍缩增益（超 ff 日程常数）。"""
    assert ssw["p_sw3a"], "SW-3a 应通过（sing ≥ ff + 0.05）"
    assert ssw["sw3a"]["singularity"] >= 0.5, \
        f"正确分支占比应 ≥0.5: {ssw['sw3a']['singularity']:.2f}"


def test_sw3b_split_positive_locked(ssw):
    """SW-3b（锁死正结果）：K=2 遇新规则 → 分裂生长 + 新分支收敛。"""
    assert ssw["p_sw3b"], "SW-3b 应通过（分裂触发 + 分支数 3 + 新分支振幅>0.3）"
    assert ssw["sw3b"]["n"] == 3 and ssw["sw3b"]["new_amp"] > 0.3


def test_sw4_single_value_negative_locked(ssw):
    """SW-4（锁死正结果）：单分支 vs 叠加 = 1.2×+（叠加必要）。"""
    assert ssw["p_sw4"], "SW-4 应通过（k1 > 1.2× k3）"
    assert ssw["sw4"]["k1"] > ssw["sw4"]["k3"], \
        f"单分支应差于叠加: k1={ssw['sw4']['k1']:.4f} k3={ssw['sw4']['k3']:.4f}"
