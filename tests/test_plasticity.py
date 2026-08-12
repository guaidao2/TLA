"""可塑性策略判据锁死测试（SW-5，预注册见 criteria/plasticity.py）。

实测裁决（2026-08-11，轻量 n_ep_calib=8/T_phase=60/seed=0 锁方向）：
  uniform：cost 0.1159 / ret 0.0022；gated：cost 0.1136 / ret 0.0015；
  frozen：cost 0.1159 / ret 0.0022（≈uniform——振幅加权损失已隐含提交，
  失败分支被饿死 → uniform 的在线训练≈冻结）；
  SW-5a 回切保留 PASS：gated ret 0.0015 ≤ 0.95×0.0022（保留优势，3/3 seed 方向一致）；
  SW-5b 切换成本 PASS：gated cost 0.1136 ≤ 1.3×0.1159（适应不慢）。

标准量（n_ep_calib=10/T_phase=80）3 seed：gated 回切恒优 1.1-2×、成本恒 ≤1.3×——
门控抵消振幅加权损失的饿死效应，切换期适应更好且保留不损。

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


def test_sw5_uniform_equals_frozen(sw5):
    """机制发现锁死：uniform≈frozen——振幅加权损失饿死失败分支（隐含提交）。"""
    u, f = sw5["res"]["uniform"], sw5["res"]["frozen"]
    assert abs(u["cost"] - f["cost"]) < 1e-6, \
        f"uniform 应≈frozen（饿死效应）: {u['cost']:.6f} vs {f['cost']:.6f}"
