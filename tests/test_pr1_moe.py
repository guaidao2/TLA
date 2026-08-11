"""摊销×MoE：锁死测试（等价性 + 四实验判据实测裁决）。

裁决依据（2026-08-11 实测）：
  E1 学习强度：标准量 0.0066 PASS（首猜承重治愈 MoE 弱学习 ✅）；轻量贴 0.02 线，
     测试锁轻量余量 <0.04；
  E2 防遗忘：未达成（锁死负结果）——路由从未真正分离 A/B（B 域 50/50、usage 0.52/0.48
     均衡），结构性腿轻量 29.3% 不达标，EWC 腿标准量 19.5%（轻量 102.6% 为假阳）。
     共享 LTC 状态使 [s,h] 输入空间的重建路由信号不具任务判别性；
  E3a 步数差分：未达成（clean/noisy/unseen median 全 2）；
  E3b 琢磨：无稳健增益（轻量 0.3126<0.3132 / 标准量 0.3197>0.3196 死平，差异在噪声级）；
  E4 路由分离：PASS（sep 0.71-0.74，A 域 ~71-74% 单专家，B 域 50/50）。
负结果与判据一样锁死。
"""
import torch
import pytest
from tla.model_pr1_moe import TLAPR1MoEModel
from tla.cognitive.pcn_amortized_moe import AmortizedMoEPCN
from tla.tasks.variable_speed_world import VariableSpeedWorld
from tla.criteria import pr1_moe


@pytest.fixture(scope="module")
def experiments():
    return pr1_moe.run_experiments(seed=0, verbose=False, n_traj=12, T=24, n_ep=2)


def test_batch_parity_single():
    """learn_step 与 learn_batch(batch_size=1) 必须给出逐参数一致的更新（局部梯度等价）。"""
    torch.manual_seed(0)
    a = TLAPR1MoEModel(seed=0)
    b = TLAPR1MoEModel(seed=0)
    x = torch.randn(3 + 16)
    t = torch.randn(2)
    a.pcn.learn_step(x, t, lr=0.01, settle_steps=2)
    b.pcn.learn_batch([x], [t], lr=0.01, settle_steps=2)
    pa, pb = a.pcn._params(), b.pcn._params()
    for k in pa:
        assert torch.allclose(pa[k], pb[k], atol=1e-8), f"参数 {k} 批/单步不一致"


def test_moe_stack_no_bp_discipline():
    """无 BP 纪律：堆栈与专家内部均无 requires_grad 张量（与 test_no_backprop 一致）。"""
    pcn = AmortizedMoEPCN(dims=[19, 24], out_dim=2, n_experts=2, seed=0)
    for ex in pcn.expert:
        assert all(not torch.is_tensor(v) or not v.requires_grad
                   for v in vars(ex).values()), "专家参数不应 requires_grad"


def test_e1_learning_strength(experiments):
    """E1：首猜承重治愈 MoE 弱学习（轻量数据锁 <0.04；预注册 0.02 判据以标准量报告为准）。"""
    assert experiments["mse_indist"] < 0.04, \
        f"E1 学习强度应成立: indist={experiments['mse_indist']:.4f}"


def test_e2_forgetting_not_achieved(experiments):
    """E2（锁死负结果）：防遗忘未达成——路由从未真正分离 A/B（B 域 50/50、usage 均衡），
    结构性腿轻量 29.3% 不达标，且 EWC 腿在标准量崩到 19.5%（轻量 102.6% 为假阳）。
    分离的收益必须建立在可判别路由上，而 [s,h] 输入空间 + 共享 LTC 状态使路由无法对齐任务。
    若未来路由修复使其达标，此测试将失败 → 强制重新评估。"""
    assert not experiments["p_e2"], "E2 应仍为负结果（防遗忘未达成）"
    assert experiments["ret_no"] < 0.80, \
        f"结构性腿应 <80%: 实测 {experiments['ret_no']:.1%}"


def test_e3b_thinking_no_measurable_gain(experiments):
    """E3b（锁死负结果）：琢磨无稳健增益——自适应与固定 1 步的差异在噪声级
    （轻量 0.3126<0.3132，标准量 0.3197>0.3196 死平），且步数无差分（E3a）。
    琢磨仍未找到正价值场地（与全部历史琢磨负结果一致）。"""
    rel = abs(experiments["mse_ad"] - experiments["mse_f1"]) / max(experiments["mse_f1"], 1e-9)
    assert rel < 0.05, \
        f"琢磨与固定深度应无可测差异: rel={rel:.3f} " \
        f"ad={experiments['mse_ad']:.4f} f1={experiments['mse_f1']:.4f}"


def test_e3a_steps_differential_not_negative(experiments):
    """E3a（锁死负结果方向）：步数差分未达成（玩具任务过易，clean/noisy/unseen
    median 全 2），但至少不得出现"困难输入琢磨更少"的反向差分。"""
    assert experiments["med_u"] >= experiments["med_c"], \
        f"困难输入不得比干净琢磨更少: clean={experiments['med_c']} unseen={experiments['med_u']}"


def test_e4_routing_separation(experiments):
    """E4：路由分离——A/B 域主导专家不同且分离度 ≥0.60。"""
    assert experiments["p_e4"], \
        f"E4 应 PASS: A域={ {k: round(v, 2) for k, v in experiments['r_a'].items()} } " \
        f"B域={ {k: round(v, 2) for k, v in experiments['r_b'].items()} } " \
        f"sep={experiments['sep']:.2f}"
