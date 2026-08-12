"""TLA 消融锁死测试（AB-1/2/3/4，预注册见 criteria/ablation_thrust.py）。

实测裁决（2026-08-11，thrust_cart，n_traj=30/T=40/n_ep=2）：
  AB-1 推理环：自适应 0.0882 / 固定1步 0.0878 / 纯首猜 0.0884——无差别（固定1步略好），
     推理环空转（FAIL 预期内，弹簧先例 P-COG-3 跨任务确认）；
  AB-2 双过程回退：默认 off（fallback=False opt-in）——贡献=0 由配置决定；
  AB-3 Self_Slot：只影响默认关闭的一致性门与诊断——不参与输出路径，贡献=0；
  AB-4 摊销捷径：标准量 freeze_base 0.0885 vs 正常 0.0878（1.0× 无差别）；轻量 5.4%
     轻微贡献——W_base 在 thrust_cart 上贡献轻微（不同于弹簧先例 27× 关键性），
     残差通路基本独力学会。
净结论：TLA 赢基线的 2.5-3× 来自残差 PCN + LTC 基板 + 局部误差学习；琢磨/双过程/
  Self_Slot 在平滑任务上全是摆设（"独有机制没起作用"——用户判据的直接回答）。
判据锁死：AB-1 阈值（1.05×）预注册；负结果（空转）同样锁死。
"""
import pytest
from tla.criteria import ablation_thrust


@pytest.fixture(scope="module")
def ab():
    return ablation_thrust.run(verbose=False, n_traj=16, T=24, n_ep=2)


def test_ab1_settle_loop_vacuous_locked(ab):
    """AB-1（锁死负结果）：关掉推理环性能不降（固定1步 ≤ 自适应×1.05）——
    推理环空转；若未来在事件/计时任务上推理环有贡献（>1.05×），此测试失败 → 重评。"""
    assert not ab["p_ab1"], "AB-1 应仍为空转（推理环不贡献）"
    assert ab["mse_f1"] <= ab["mse_ad"] * 1.05, \
        f"关掉推理环不应降: f1={ab['mse_f1']:.4f} ad={ab['mse_ad']:.4f}"


def test_ab1_guess_close_to_settle(ab):
    """AB-1 补充：纯首猜与自适应差距 <5%（连 settle 精化都不如首猜有用）。"""
    assert ab["mse_g0"] <= ab["mse_ad"] * 1.05, \
        f"纯首猜不应差: g0={ab['mse_g0']:.4f} ad={ab['mse_ad']:.4f}"


def test_ab4_shortcut_light_contribution(ab):
    """AB-4：冻结 W_base 仅轻微降（≤1.15×）——残差通路基本独力学会。
    标准量 1.0×（无差别）、轻量 5.4%（轻微贡献）——W_base 在 thrust_cart 上贡献轻微
    （不同于弹簧先例 27× 关键性）；若未来任务上 W_base 关键（>1.15×），此测试失败 → 重评。"""
    assert ab["mse_fb"] <= ab["mse_f1"] * 1.15, \
        f"冻结 W_base 应仅轻微降: fb={ab['mse_fb']:.4f} f1={ab['mse_f1']:.4f}"


def test_ab2_ab3_config_off(ab):
    """AB-2/AB-3：双过程回退与 Self_Slot 默认关（贡献=0 由配置决定）——披露存在。"""
    assert "fallback=False" in ab["ab2"]
    assert "不参与输出路径" in ab["ab3"]
