"""可塑性策略判据锁死测试（SW-5，预注册见 criteria/plasticity.py）。

实测裁决（2026-08-11，轻量 n_ep_calib=8/T_phase=60/seed=0 锁方向；终审修复后）：
  uniform：cost 0.1665 / ret 0.0028；gated：cost 0.1452 / ret 0.0014；
  frozen（真 η=0，终审修复）：cost 0.1131 / ret 0.0023；
  SW-5a 回切保留 PASS：gated ret 0.0014 ≤ 0.95×0.0028（保留优势，3/3 seed 方向一致）；
  SW-5b 切换成本 PASS：gated cost 0.1452 ≤ 1.3×0.1665（适应不慢）。

**表述撤回（终审修复）**：初版"uniform≈frozen（振幅加权饿死效应）"是 bug 产物——
plasticity="frozen" 原实现落到 uniform 分支（frozen=uniform 是恒等式）。修复后真实
frozen（η=0）切换成本 0.1131 为最低（不适应+坍缩选最不差分支），uniform 0.1665 最差
（适应反而扰动 min-err）。真实结论：gated 在保留（2×）与切换成本（优于 uniform）上
双向占优——显式门控有效，但"饿死效应"机制表述撤回。

标准量（n_ep_calib=10/T_phase=80）3 seed：gated 回切恒优 1.1-2×、成本恒 ≤1.3×。

净结论：显式可塑性门控（η_i∝1−amp_i）在解析头叠加模型上是有效增量——
"适应 vs 保留"不再二选一。
"""
import pytest
from ssw.criteria import plasticity


@pytest.fixture(scope="module")
def sw5():
    return plasticity.run(seed=0, verbose=False, n_ep_calib=8, T=50, T_phase=60)


def test_sw5_gated_better_tradeoff(sw5):
    """SW-5（锁死正结果）：门控在适应与保留间取得更优权衡。"""
    assert sw5["p_sw5"], "SW-5 应通过（回切保留 + 切换成本均达标）"
    g, u = sw5["res"]["gated"], sw5["res"]["uniform"]
    assert g["restore"] < u["restore"], \
        f"gated 回切应更快: {g['restore']:.4f} vs {u['restore']:.4f}"
    assert g["cost"] <= 1.3 * u["cost"], \
        f"gated 适应不应慢太多: {g['cost']:.4f} vs {u['cost']:.4f}"
