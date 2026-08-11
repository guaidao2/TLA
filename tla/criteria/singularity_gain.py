"""奇点基板增益实验（预注册，2026-08-11）：标定 ε/input_scale 使硬暴胀真实发生，
测稀疏激活 + 时间戳保持 + 世界模型增益。

背景：换装首测（substrate_swap.py）激活率 0%——ε 起爆慢（1e-4）+ 轨迹短 + W_h 惰性，
奇点特征行为从未触发。"换装可用"成立，"特征行为带来增益"未验证。

判据（跑数前锁死，标定是参数校准不是改判据）：
  S-T1 硬暴胀标定：存在 (eps, input_scale) 使基板激活率 ∈ [5%, 40%]（稀疏硬激活真实发生）；
  S-T2 双簇成立：激活细胞 h 分布落在 [0.5, 1.0]（真炸），非激活细胞落在 [0, 0.1]（真没炸）
     ——不是渐变噪声（幽灵态/饱和态双不动点结构在基板级成立）；
  S-T3 世界模型增益（核心）：标定后奇点基板未见 ω MSE ≤ LTC 基板 × 1.1（不劣化 10%），
     且学习成立（S-B1：< 0.7×随机 且 < 恒等）、防遗忘保持（S-B3：EWC 保留率 ≥ 0.95）。

实测裁决（2026-08-11，n_traj=20/T=30/n_ep=2）：
  S-T1 PASS：eps=0.01、input_scale=4.0 → 激活率 10.5%（硬暴胀真实发生，首次突破 0%）；
  S-T2 FAIL（负向锁死）：非激活细胞 h 可达 0.499（贴 0.5 边界）——连续近边界分布，
     非干净的幽灵/饱和双簇（细胞连续追踪输入驱动平衡态）；
  S-T3 无增益（负向锁死）：标准量奇点 0.1394 vs LTC 0.1346（略差 3.6%）、轻量差 24%——
     硬暴胀未带来世界模型净收益，"时间戳增益"无迹象（学习成立，换装可用不退化）。

判据锁死：跑数后任何裁决不得篡改；标定参数（eps/input_scale）是测试设置，判据标准不变。
"""
import torch
from tla.model_pr1 import TLAPR1Model
from tla.substrate.ltc_cell import LTCCell
from tla.substrate.singularity_substrate import SingularitySubstrate
from tla.tasks.variable_speed_world import VariableSpeedWorld


def activation_rate(substrate_kw, trajs):
    sub = SingularitySubstrate(in_dim=3, hidden=16, seed=0, **substrate_kw)
    hot, total = 0, 0
    for traj in trajs:
        sub.reset()
        for t in range(len(traj)):
            sub.forward(traj[t])
            hot += int((sub.h > 0.5).sum().item())
            total += sub.hidden
    return hot / max(total, 1)


def cluster_stats(substrate_kw, trajs):
    """双簇统计：激活(h>0.5)与非激活(h<=0.5)细胞各自的 h 范围。"""
    act_lo, act_hi, inact_hi = 1.0, 0.0, 0.0
    sub = SingularitySubstrate(in_dim=3, hidden=16, seed=0, **substrate_kw)
    for traj in trajs:
        sub.reset()
        for t in range(len(traj)):
            sub.forward(traj[t])
            for v in sub.h.tolist():
                if v > 0.5:
                    act_lo = min(act_lo, v)
                    act_hi = max(act_hi, v)
                else:
                    inact_hi = max(inact_hi, v)
    return act_lo, act_hi, inact_hi


def eval_mse(model, trajs):
    mses = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            pred, _ = model.infer(traj[t])
            mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def run_gain(seed=0, verbose=True, n_traj=20, T=30, n_ep=2):
    world = VariableSpeedWorld(seed=seed, mode="spring")
    train = world.trajectories(n_traj=n_traj, T=T, speed_range=(0.8, 3.0))
    test_unseen = world.trajectories(n_traj=3, T=20, speed_range=(3.5, 5.0), seed=999)

    # ---- S-T1 标定：扫 (eps, input_scale) 找激活率 ∈ [5%, 40%] ----
    grid = [(e, s) for e in (1e-4, 1e-3, 1e-2) for s in (0.4, 1.0, 2.0, 4.0)]
    best = None
    for eps, scale in grid:
        rate = activation_rate(dict(eps=eps, input_scale=scale), train)
        if 0.05 <= rate <= 0.40 and (best is None or rate > best[1]):
            best = ((eps, scale), rate)
    if best is None:
        # 无组合达标 → 取激活率最高者并记录 FAIL（标定未达成）
        rates = {f"{e}/{s}": activation_rate(dict(eps=e, input_scale=s), train)
                 for e, s in grid}
        best_kw = max(rates, key=rates.get)
        e, s = (float(best_kw.split("/")[0]), float(best_kw.split("/")[1]))
        best = ((e, s), rates[best_kw])
    (eps_c, scale_c), rate_c = best
    p_t1 = 0.05 <= rate_c <= 0.40

    # ---- S-T2 双簇 ----
    act_lo, act_hi, inact_hi = cluster_stats(dict(eps=eps_c, input_scale=scale_c), train)
    p_t2 = act_hi >= 0.5 and inact_hi <= 0.1 and act_lo >= 0.5

    # ---- S-T3 世界模型增益 ----
    def make_sing():
        return TLAPR1Model(seed=seed, substrate_cls=lambda **kw: SingularitySubstrate(
            eps=eps_c, input_scale=scale_c, **kw))

    def make_ltc():
        return TLAPR1Model(seed=seed, substrate_cls=LTCCell)

    m_sing = make_sing()
    m_rand = make_sing()
    mse_rand = eval_mse(m_rand, test_unseen)
    # 恒等基线（预测 s_t，从测试集实算）
    id_mses = [float(torch.mean((t[0][:2] - t[1][:2]) ** 2).item())
               for tr in test_unseen for t in zip(tr, tr[1:])]
    mse_id = float(torch.tensor(id_mses).mean().item())
    for _ in range(n_ep):
        for traj in train:
            for t in range(len(traj) - 1):
                m_sing.train_step(traj[t], traj[t + 1])
    mse_sing = eval_mse(m_sing, test_unseen)
    m_ltc = make_ltc()
    for _ in range(n_ep):
        for traj in train:
            for t in range(len(traj) - 1):
                m_ltc.train_step(traj[t], traj[t + 1])
    mse_ltc = eval_mse(m_ltc, test_unseen)
    p_sb1 = mse_sing < 0.7 * mse_rand and mse_sing < mse_id
    p_t3 = mse_sing <= mse_ltc * 1.1 and p_sb1

    out = dict(eps=eps_c, scale=scale_c, rate=rate_c, p_t1=p_t1,
               act_lo=act_lo, act_hi=act_hi, inact_hi=inact_hi, p_t2=p_t2,
               mse_sing=mse_sing, mse_ltc=mse_ltc, mse_rand=mse_rand,
               p_sb1=p_sb1, p_t3=p_t3)
    if verbose:
        print("=" * 64)
        print(f"奇点基板增益实验 (n_traj={n_traj}, T={T}, n_ep={n_ep})")
        print("=" * 64)
        print(f"S-T1 标定: eps={eps_c}, input_scale={scale_c}, 激活率={rate_c:.1%} "
              f"→ {'PASS' if p_t1 else 'FAIL'}")
        print(f"S-T2 双簇: 激活 h∈[{act_lo:.2f},{act_hi:.2f}] 非激活 h≤{inact_hi:.3f} "
              f"→ {'PASS' if p_t2 else 'FAIL'}")
        print(f"S-T3 增益: 奇点={mse_sing:.4f} vs LTC={mse_ltc:.4f} vs 随机={mse_rand:.4f} "
              f"→ {'PASS' if p_t3 else 'FAIL'}（学习成立={p_sb1}）")
        print("=" * 64)
    return out


if __name__ == "__main__":
    run_gain(verbose=True)
