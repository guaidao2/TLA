"""奇点基板增益实验锁死测试（S-T1/T2/T3，预注册见 criteria/singularity_gain.py）。

实测裁决（2026-08-11，n_traj=20/T=30/n_ep=2）：
  S-T1 硬暴胀标定：PASS——eps=0.01、input_scale=4.0 激活率 10.5%（∈[5%,40%]，
     稀疏硬激活真实发生，首次突破换装首测的 0%）；
  S-T2 双簇：FAIL（锁死负结果）——非激活细胞 h 可达 0.499（贴 0.5 边界），
     不是干净的幽灵态/饱和态双簇，而是连续近边界分布（细胞连续追踪输入驱动的平衡态）；
  S-T3 世界模型增益：**无增益（负向，锁死）**——标准量奇点 0.1394 vs LTC 0.1346（略差
     3.6%，1.1× 容忍内），轻量差 24%；硬暴胀双簇化未带来世界模型净收益，
     "时间戳增益"无迹象（学习成立，换装可用不退化）。

判据锁死：标定参数是测试设置，判据标准未改。
"""
import pytest
from tla.criteria import singularity_gain


@pytest.fixture(scope="module")
def gain():
    return singularity_gain.run_gain(verbose=False, n_traj=16, T=24, n_ep=2)


def test_st1_hard_inflation_calibrated(gain):
    """S-T1：硬暴胀真实发生——标定组合激活率 ∈ [5%, 40%]。"""
    assert gain["p_t1"], \
        f"S-T1 应 PASS: eps={gain['eps']} scale={gain['scale']} 激活率={gain['rate']:.1%}"


def test_st2_bimodal_negative_locked(gain):
    """S-T2（锁死负结果）：双簇不成立——非激活细胞可达 0.499（贴边界），
    状态是连续近边界分布而非幽灵/饱和双簇。若未来修复使双簇成立，此测试失败 → 强制重评。"""
    assert not gain["p_t2"], "S-T2 应仍为负结果（双簇未成立）"
    assert gain["inact_hi"] > 0.1, \
        f"非激活细胞应可达 >0.1（贴边界连续分布）: inact_hi={gain['inact_hi']:.3f}"


def test_st3_no_gain_locked(gain):
    """S-T3（锁死负结果）：奇点基板未显著优于 LTC——标准量略差 3.6%、轻量差 24%，
    '时间戳增益'无迹象（硬暴胀后的双簇化未带来世界模型净收益）。
    若未来标定使其显著优于 LTC（<0.9×），此测试失败 → 强制重评。"""
    assert gain["mse_sing"] >= 0.9 * gain["mse_ltc"], \
        f"奇点未显著优于 LTC 应成立: sing={gain['mse_sing']:.4f} ltc={gain['mse_ltc']:.4f}"


def test_st3_learning_holds(gain):
    """S-T3 学习成立腿：奇点基板显著优于随机基线（换装可用不退化）。"""
    assert gain["p_sb1"], \
        f"学习应成立: sing={gain['mse_sing']:.4f} rand={gain['mse_rand']:.4f}"
