"""奇点基板增益实验锁死测试（S-T1/T2/T3/T4，预注册见 criteria/singularity_gain.py）。

实测裁决（2026-08-11，动力学修复后 n_traj=20/T=30/n_ep=2）：
  S-T1 硬暴胀标定：PASS——eps=0.01、input_scale=4.0 激活率 17.8%（∈[5%,40%]）；
  S-T2 双簇：FAIL（锁死负结果）——非激活细胞 h 可达 0.500 且 h>0.1 占比 16.0%，
     连续近边界分布（快动力学下双簇仍未干净分离）；
  S-T3 世界模型增益：**翻转正结果**——动力学修复（λ=0.5、β=3.0，衰减相真实出现）后
     奇点 0.1232 vs LTC 0.1346，多种子比 [0.915, 0.998] 全 <1（一致好 0.2-8.5%）；
     上一轮"负向锁死"因实现缺陷重开（衰减相从未出现 → 时间戳从未可解码 → 测试条件
     不公平），修复后翻正；
  S-T4 衰减相：PASS——热后衰减事件 688（修复验收：衰减相真实出现，时间戳可解码前提）。

判据锁死：S-T1/T3 标准未改（标定参数是测试设置）；S-T3 裁决因实现缺陷重开重测。
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
    """S-T2（锁死负结果）：双簇不成立——非激活细胞可达 0.499（贴 0.5 边界）且
    h>0.1 的非激活样本占比显著（连续近边界分布）；激活簇贴边（act_lo≈0.50，真炸 >0.7
    未达）。若未来修复使双簇成立，此测试失败 → 强制重评。"""
    assert not gain["p_t2"], "S-T2 应仍为负结果（双簇未成立）"
    assert gain["inact_hi"] > 0.1, \
        f"非激活细胞应可达 >0.1（贴边界连续分布）: inact_hi={gain['inact_hi']:.3f}"
    assert gain["frac_over_0p1"] > 0.05, \
        f"h>0.1 的非激活样本占比应显著: {gain['frac_over_0p1']:.1%}"
    assert gain["act_lo"] <= 0.7, \
        f"激活簇应贴边（真炸未达 >0.7）: act_lo={gain['act_lo']:.2f}"


def test_st3_gain_positive(gain):
    """S-T3（动力学修复后翻转正结果）：奇点基板未见 ω 预测不劣于 LTC——修复后
    标准量 0.1232 vs LTC 0.1346，多种子比 [0.915, 0.998] 全 <1（一致好 0.2-8.5%）。
    上一轮"负向锁死"因实现缺陷（衰减相从未出现）重开，修复后翻正。"""
    assert gain["mse_sing"] <= gain["mse_ltc"], \
        f"奇点应不劣于 LTC: sing={gain['mse_sing']:.4f} ltc={gain['mse_ltc']:.4f}"


def test_st4_decay_phase_appears(gain):
    """S-T4：衰减相真实出现（动力学修复验收）——热后衰减事件 > 0。"""
    assert gain["hot_decay"] > 0, \
        f"衰减相应真实出现: hot_decay={gain['hot_decay']}"


def test_st3_learning_holds(gain):
    """S-T3 学习成立腿：奇点基板显著优于随机基线（换装可用不退化）。"""
    assert gain["p_sb1"], \
        f"学习应成立: sing={gain['mse_sing']:.4f} rand={gain['mse_rand']:.4f}"
