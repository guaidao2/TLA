"""2D 推力体世界模型锁死测试（TC-1/2/3，预注册见 criteria/thrust_cart.py）。

实测裁决（2026-08-11，n_traj=30/T=40/n_ep=2）：
  TC-1 学习成立：PASS——未见重力（g=2.5）MSE 0.0882 < 0.7×随机 0.4630 且 < 恒等 0.1166；
  TC-2 动作条件化：PASS——恒推(+1,+1) 预测 vx'=0.1445 vs 恒推(−1,−1) vx'=−0.0535
     （模型真正使用动作信息，非只看状态——弹簧世界给不了的验证）；
  TC-3 防遗忘：PASS——g1.0→g2.0 重力任务序列 EWC 保留率 101.4%（≥95%）。

判据锁死：跑数前不改判据，只许改代码。
"""
import pytest
from tla.criteria import thrust_cart


@pytest.fixture(scope="module")
def tc():
    return thrust_cart.run(verbose=False, n_traj=16, T=24, n_ep=2)


def test_tc1_learning(tc):
    """TC-1：无 BP 在含动作非线性世界模型上学得动（未见重力泛化）。"""
    assert tc["p_tc1"], \
        f"TC-1 应 PASS: mse={tc['mse_g']:.4f} vs rand={tc['mse_rand']:.4f} " \
        f"vs id={tc['mse_id']:.4f}"


def test_tc2_action_conditioning(tc):
    """TC-2：动作条件化——恒推方向不同，预测 vx' 必须显著不同（动作被真正使用）。"""
    assert tc["p_tc2"], \
        f"TC-2 应 PASS: vx'(+1)={tc['vx_p1']:.4f} vs vx'(−1)={tc['vx_m1']:.4f}"


def test_tc3_forgetting(tc):
    """TC-3：重力任务序列 g1.0→g2.0，EWC 防遗忘保留率 ≥95%。"""
    assert tc["p_tc3"], f"TC-3 应 PASS: EWC 保留率={tc['ret']:.1%}"
