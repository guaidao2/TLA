"""P-COG-4：doubtful 标记校准性（预注册判据）。"""
import torch
from tla.criteria.calibration import run_calibration


def test_pcog4_calibration():
    """doubtful/置信度与实际错误率校准：低置信分位错误显著高于高置信分位。"""
    r = run_calibration(seed=0, verbose=False)
    assert r["mse_lo"] > 2.0 * r["mse_hi"], \
        f"置信度应校准: low-conf mse={r['mse_lo']:.4f} vs high-conf mse={r['mse_hi']:.4f}"


def test_pcog4_doubtful_marks_higher_error():
    """doubtful 标记样本的实际错误率显著高于未标记样本（标记≠虚报）。"""
    r = run_calibration(seed=0, verbose=False)
    assert torch.isfinite(torch.tensor(r["mse_doubt"])), "doubtful 标记应为非空"
    assert r["mse_doubt"] > 2.0 * r["mse_nodoubt"], \
        f"doubtful 应标记高错误样本: doubt={r['mse_doubt']:.4f} vs nodoubt={r['mse_nodoubt']:.4f}"
