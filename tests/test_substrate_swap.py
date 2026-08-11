"""奇点基板换装（SingularitySubstrate 代替 LTCCell）锁死测试。

实测裁决（2026-08-11，n_traj=20/T=30/n_ep=2，input_scale=0.4 默认）：
  S-B1 学习（P-LEARN-3 复测）：PASS——未见 ω 0.1332 < 0.7×随机 0.3073 且 < 恒等 0.1778；
  S-B2 步数（P-COG-1/2 复测）：PASS——噪声均值 2.40 > 干净均值 2.12（差分方向成立）；
  S-B3 防遗忘（P-LEARN-1 EWC 复测）：PASS——EWC 保留率 139.5%（≥95%）。
标定发现：input_scale 调大（1.0/2.0）学习与防遗忘变差（unseen 0.137/0.203、保留 38.3%/
82.9%）；且所有尺度激活率 0%——ε 起爆慢（1e-4）+ 轨迹短（30 tick），细胞未炸过 0.5，
奇点特征行为（稀疏硬暴胀）未被触发（下一实验的参数标定问题，非判据失败）。
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
