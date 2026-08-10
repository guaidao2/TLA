"""P-COG-4：doubtful 标记校准性（预注册判据）。"""
import torch
import pytest
from tla.criteria.calibration import run_calibration


@pytest.fixture(scope="module")
def calib():
    return run_calibration(seed=0, verbose=False, n_epochs=2, n_traj=25, T=40)


def test_pcog4_calibration(calib):
    """doubtful/置信度与实际错误率校准：低置信分位错误显著高于高置信分位。"""
    assert calib["mse_lo"] > 2.0 * calib["mse_hi"], \
        f"置信度应校准: low-conf mse={calib['mse_lo']:.4f} vs high-conf mse={calib['mse_hi']:.4f}"


def test_pcog4_doubtful_marks_higher_error(calib):
    """doubtful 标记样本的实际错误率显著高于未标记样本（标记≠虚报）。"""
    assert torch.isfinite(torch.tensor(calib["mse_doubt"])), "doubtful 标记应为非空"
    assert calib["mse_doubt"] > 2.0 * calib["mse_nodoubt"], \
        f"doubtful 应标记高错误样本: doubt={calib['mse_doubt']:.4f} vs nodoubt={calib['mse_nodoubt']:.4f}"
