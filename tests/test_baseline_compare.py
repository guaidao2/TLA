"""TLA vs 标准基线（LSTM/NeuralODE/SSM）锁死测试（BC-1，预注册见 baseline_compare.py）。

实测裁决（2026-08-11，thrust_cart 同种子同分割，n_traj=30/T=40/n_ep=2）：
  TLA(无BP) 0.0882 vs LSTM(BP) 0.2227（赢 2.5×）、NeuralODE 0.2578（赢 3.0×）、
  SSM(Mamba式) 0.2452（赢 2.8×）——BC-1 PASS。
敏感性（n_traj=20）：基线给 5× 训练（10 epoch）仍在追——LSTM 0.122、SSM 0.275、
  ODE 0.247，全部仍差于 TLA（~0.09）；"没训够"混淆排除。
诚实标注：TLA 含摊销线性捷径（W_base）——基线为标准 MLP/LSTM/ODE/SSM 无捷径，
  赢的方向可能部分来自捷径归纳偏置（架构价值 vs 捷径价值需后续消融）；参数量同量级
  （TLA ~700 vs LSTM 3772/ODE 1088/SSM 320）。
判据锁死：BC-1（TLA ≤1.1×全部基线）标准未改。
"""
import pytest
import torch
from tla.criteria import baseline_compare


@pytest.fixture(scope="module")
def bc():
    return baseline_compare.run(verbose=False, n_traj=16, T=24, n_ep=2)


def test_bc1_tla_not_worse_than_baselines(bc):
    """BC-1：TLA 未见重力 MSE ≤ 每个基线 ×1.1（持平或赢）。"""
    assert bc["p_bc1"], \
        f"BC-1 应 PASS: TLA={bc['mse_tla']:.4f} " \
        + " ".join(f"{k}={v['mse']:.4f}" for k, v in bc["baselines"].items())


def test_bc1_tla_wins_each_baseline(bc):
    """BC-1 强版本：TLA 显著优于每个基线（<0.9×）。"""
    for name, r in bc["baselines"].items():
        rel = bc["mse_tla"] / max(r["mse"], 1e-12)
        assert rel < 0.9, f"TLA 应显著优于 {name}: 比={rel:.2f}"


def test_bc2_disclosure_present(bc):
    """BC-2：公平性披露存在（TLA 无 BP vs 基线 BP + 参数量 + settle 循环不对称）。"""
    assert bc["disclosure"]["tla_bp_free"] and bc["disclosure"]["baselines_bp"]
    assert "settle" in bc["disclosure"]["tla_settle_loop"] or \
        "settle" in str(bc["disclosure"])
    for r in bc["baselines"].values():
        assert r["params"] > 0


def test_sensitivity_reproducible():
    """终审：敏感性可复现——基线给更多 epoch（5）仍差于 TLA（2 epoch）——
    '没训够'排除从仓库可直接验证（不再只存在于文档）。"""
    from tla.criteria import baseline_compare as bc_mod
    from tla.model_pr1 import TLAPR1Model
    from tla.tasks.thrust_cart import ThrustCartWorld
    sens = bc_mod.run_sensitivity(seed=0, n_traj=16, T=24, epochs_list=(2, 5))
    world = ThrustCartWorld(g=1.0, seed=0)
    train = world.task(g=1.0, n=16, T=24, seed_shift=1)
    test = world.task(g=2.5, n=4, T=20, seed_shift=99)
    m = TLAPR1Model(obs_dim=6, out_dim=4, seed=0)
    for _ in range(2):
        for traj in train:
            for s, a, s_next in traj:
                m.train_step(torch.cat([s, a]), s_next)
    mse_tla = bc_mod.eval_tla(m, test)
    # 基线 5 epoch 仍应差于 TLA 2 epoch
    for name, mse in sens[5].items():
        assert mse_tla < mse, f"TLA 应仍优于 {name}@5epoch: TLA={mse_tla:.4f} {name}={mse:.4f}"
