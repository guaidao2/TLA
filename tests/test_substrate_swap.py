"""奇点基板换装（SingularitySubstrate 代替 LTCCell）锁死测试。

实测裁决（2026-08-11，n_traj=20/T=30/n_ep=2，input_scale=0.4 默认，动力学修复后
λ=0.5/β=3.0 新默认）：
  S-B1 学习（P-LEARN-3 复测）：PASS——未见 ω 0.1330 < 0.7×随机 0.3075 且 < 恒等 0.1778；
  S-B2 步数（P-COG-1/2 复测）：PASS——噪声均值 2.70 > 干净均值 2.11（差分方向成立）；
  S-B3 防遗忘（P-LEARN-1 EWC 复测）：PASS——EWC 保留率 135.3%（≥95%）。
标定发现：input_scale 调大学习与防遗忘变差；默认 scale=0.4 下激活率 0%（I std≈0.09
远低于 ε=1e-4 时的 I_th≈1.0）——近幽灵渐变区，特征行为未触发（硬暴胀需 eps=0.01/
scale=4.0 标定组合，见 singularity_gain.py）。
判据锁死：与 LTC 基线同判据标准，换装只改代码。
"""
import torch
import pytest
from tla.model_pr1 import TLAPR1Model
from tla.substrate.singularity_substrate import SingularitySubstrate
from tla.criteria import substrate_swap


@pytest.fixture(scope="module")
def swap():
    return substrate_swap.run_swap(verbose=False, n_traj=12, T=24, n_ep=1)


def test_swap_learning_sb1(swap):
    """S-B1：奇点基板下无 BP 学习仍成立（P-LEARN-3 判据复测）。"""
    assert swap["p_b1"], \
        f"S-B1 应 PASS: unseen={swap['mse_unseen']:.4f} vs rand={swap['mse_rand']:.4f}"


def test_swap_steps_differential_sb2(swap):
    """S-B2：奇点基板下噪声步数差分方向成立（P-COG-1/2 复测）。"""
    assert swap["p_b2"], \
        f"S-B2 应 PASS: clean mean={swap['mean_c']:.2f} noise mean={swap['mean_n']:.2f}"


def test_swap_forgetting_ewc_sb3(swap):
    """S-B3：奇点基板下 EWC 防遗忘保留率 ≥95%（P-LEARN-1 复测）。"""
    assert swap["p_b3"], f"S-B3 应 PASS: EWC 保留率={swap['ret_ewc']:.1%}"


def test_substrate_dropin_interface():
    """奇点基板 drop-in：与 LTCCell 同接口（forward/reset/h），模型可换装。"""
    m = TLAPR1Model(seed=0, substrate_cls=SingularitySubstrate)
    assert m.ltc.h.shape == (16,)
    h1 = m.ltc.forward(torch.randn(3))
    assert h1.shape == (16,)
    m.reset()
    assert torch.all(m.ltc.h == 0)
    # 输入尺度标定注记：默认 0.4 落在近幽灵渐变区（激活率 ~0%，特征行为未触发）
    assert 0.0 <= m.ltc.h.min() <= m.ltc.h.max() <= 1.0


def test_replay_context_restore():
    """replay h_ctx 还原语义：还原 ltc.h 后 forward 必须从还原态继续演化
    （终审 should-fix：张量与细胞内态需同步，否则换装语义失真）。"""
    a = TLAPR1Model(seed=0, substrate_cls=SingularitySubstrate)
    b = TLAPR1Model(seed=0, substrate_cls=SingularitySubstrate)
    x = torch.randn(3)
    h1 = a.ltc.forward(x)
    h2a = a.ltc.forward(x)               # a 继续走一步
    b.ltc.h = h1.clone()                 # b 还原 a 第一步后的状态（replay 语义）
    h2b = b.ltc.forward(x)
    assert torch.allclose(h2a, h2b, atol=1e-6), "还原后应严格从还原态演化"
