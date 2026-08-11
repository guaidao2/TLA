"""摊销首猜 × MoE 专家分离：四实验判据（预注册，2026-08-11）。

定位：三个历史负结果的合体翻盘实验——
  MoE 弱学习（旧 MoE ~0.11）→ 首猜承重（原则一，0.0046 量级）应治愈；
  P-LEARN-1 遗忘 → 专家分离（A/B 路由到不同专家，权重不冲突）+ EWC 双保险；
  P-COG-3 琢磨负价值 → 专家内 settle（路由先选专家，琢磨只该用时用）+ 回退兜底。

判据（跑数前写死，不符→如实记录负结果）：
  E1 学习强度：分布内 MSE < 0.02（沿用 moe.py 解锁阈值）→ "首猜治愈 MoE 弱学习"成立；
  E2 防遗忘：有 EWC 保留率 ≥ 0.95（主判据）；无 EWC 保留率 ≥ 0.80（结构性分离辅助判据）；
  E3 琢磨只该用时用：
     (a) 步数差分 median_clean < median_noisy ≤ median_unseen（难的才多琢磨）；
     (b) 未见 ω 上 自适应 ≤ 固定1步（琢磨不劣化，防空转/过度精化）；
  E4 路由分离：A/B 测试样本专家分配一致性 ≥ 0.60（分离真实发生）。

实测裁决（2026-08-11，seed=0）：
  E1：标准量 0.0066 PASS（首猜承重治愈 MoE 弱学习）；轻量贴 0.02 线；
  E2：未达成（锁死负结果）——路由未分离 A/B（usage 均衡），结构性腿轻量 29.3% 不达标，
      EWC 腿标准量 19.5%（轻量 102.6% 为假阳）；
  E3：E3b 无稳健增益（标准量 0.3197 vs 0.3196 死平，差异噪声级）；E3a 未达成（步数无差分）；
  E4：PASS（sep 0.71-0.74，A 域 ~71-74% 单专家，B 域 50/50）。

判据锁死：跑数后任何裁决不得篡改。
"""
import torch
from tla.model_pr1_moe import TLAPR1MoEModel
from tla.tasks.variable_speed_world import VariableSpeedWorld


def eval_mse(model, trajs, max_steps=None, noisy=False):
    mses = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            o = traj[t]
            if noisy:
                o = o + 0.1 * torch.randn_like(o)
            pred, _ = model.infer(o, max_steps=max_steps)
            mses.append(float(torch.mean((pred - traj[t + 1][:2]) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def steps_stats(model, trajs, noisy=False):
    """返回 (median, mean) 推理步数。"""
    ss = []
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            o = traj[t]
            if noisy:
                o = o + 0.1 * torch.randn_like(o)
            _, info = model.infer(o)
            ss.append(info["steps"])
    ss = torch.tensor(ss, dtype=torch.float32)
    return float(ss.median().item()), float(ss.mean().item())


def routing_assignment(model, trajs):
    """路由分配一致性：返回 {0: 分给专家0的比例, 1: 分给专家1的比例}（argmax 路由）。"""
    counts = {0: 0, 1: 0}
    total = 0
    for traj in trajs:
        model.reset()
        for t in range(len(traj) - 1):
            h = model.ltc.forward(traj[t])
            x = torch.cat([traj[t], h])
            model.pcn.errors(x)          # 触发路由计算（原型距离 argmax）
            r = model.pcn.last_routing
            idx = int(torch.argmax(r).item())
            counts[idx] += 1
            total += 1
    return {k: v / max(total, 1) for k, v in counts.items()}, total


def run_experiments(seed=0, verbose=True, n_traj=30, T=40, n_ep=3):
    """n_traj/T/n_ep 显式参数：报告用标准量（30/40/3），测试用轻量（8/20/1）。"""
    world = VariableSpeedWorld(seed=seed, mode="spring")
    train = world.trajectories(n_traj=n_traj, T=T, speed_range=(0.8, 3.0))
    test_clean = world.trajectories(n_traj=4, T=20, speed_range=(1.0, 2.0), seed=7)
    test_unseen = world.trajectories(n_traj=4, T=20, speed_range=(3.5, 5.0), seed=999)

    # ---- E1 学习强度 ----
    m = TLAPR1MoEModel(seed=seed)
    for _ in range(n_ep):
        for traj in train:
            for t in range(len(traj) - 1):
                m.train_step(traj[t], traj[t + 1])
    mse_indist = eval_mse(m, test_clean)
    mse_unseen = eval_mse(m, test_unseen)
    p_e1 = mse_indist < 0.02

    # ---- E3 琢磨只该用时用 ----
    med_c, mean_c = steps_stats(m, test_clean)
    med_n, mean_n = steps_stats(m, test_clean, noisy=True)
    med_u, mean_u = steps_stats(m, test_unseen)
    p_e3a = med_c < med_n <= med_u
    mse_ad = eval_mse(m, test_unseen)
    mse_f1 = eval_mse(m, test_unseen, max_steps=1)
    p_e3b = mse_ad <= mse_f1
    p_e3 = p_e3a and p_e3b

    # ---- E2/E4 防遗忘（A/B，有/无 EWC）+ 路由分离（A/B 域）----
    def run_ab(use_ewc, want_model=False):
        mm = TLAPR1MoEModel(seed=seed)
        wa = VariableSpeedWorld(seed=seed, mode="spring")
        wb = VariableSpeedWorld(seed=seed + 10, mode="spring")
        train_a = wa.trajectories(n_traj=n_traj, T=T, speed_range=(0.8, 1.5))
        test_a = wa.trajectories(n_traj=3, T=15, speed_range=(0.9, 1.3), seed=7)
        train_b = wb.trajectories(n_traj=n_traj, T=T, speed_range=(3.5, 4.5))
        test_b = wb.trajectories(n_traj=3, T=15, speed_range=(3.8, 4.3), seed=11)
        # 阶段 A（consolidate 累计 importance——须先 start 再训练，否则 _imp 被清零）
        mm.pcn.start_consolidation()
        for _ in range(n_ep):
            for traj in train_a:
                for t in range(len(traj) - 1):
                    mm.train_step(traj[t], traj[t + 1], consolidate=True)
        mse_a0 = eval_mse(mm, test_a)
        mm.pcn.finalize_consolidation()
        # 阶段 B（可选 protect；lam 在模型上设，train_step 不接收）
        mm.lam = 10.0
        for _ in range(n_ep):
            for traj in train_b:
                for t in range(len(traj) - 1):
                    mm.train_step(traj[t], traj[t + 1], protect=use_ewc)
        mse_a1 = eval_mse(mm, test_a)
        mse_b = eval_mse(mm, test_b)
        ret = (mse_a0 / max(mse_a1, 1e-12))
        if want_model:
            return mse_a0, mse_a1, mse_b, ret, mm, test_a, test_b
        return mse_a0, mse_a1, mse_b, ret

    a0_ewc, a1_ewc, b_ewc, ret_ewc, m_ewc, test_a, test_b = run_ab(True, want_model=True)
    a0_no, a1_no, b_no, ret_no = run_ab(False)
    p_e2_ewc = ret_ewc >= 0.95
    p_e2_struct = ret_no >= 0.80
    p_e2 = p_e2_ewc and p_e2_struct

    # ---- E4 路由分离：在 A/B 训练后的模型上测（A 域→专家A、B 域→专家B）----
    r_a, _ = routing_assignment(m_ewc, test_a)
    r_b, _ = routing_assignment(m_ewc, test_b)
    dom_a = 0 if r_a[0] >= r_a[1] else 1
    dom_b = 0 if r_b[0] >= r_b[1] else 1
    sep = max(r_a[dom_a], r_b[dom_b])
    p_e4 = (dom_a != dom_b) and sep >= 0.60

    out = dict(mse_indist=mse_indist, mse_unseen=mse_unseen, p_e1=p_e1,
               med_c=med_c, med_n=med_n, med_u=med_u, p_e3a=p_e3a,
               mse_ad=mse_ad, mse_f1=mse_f1, p_e3b=p_e3b, p_e3=p_e3,
               ret_ewc=ret_ewc, ret_no=ret_no, p_e2=p_e2,
               p_e2_ewc=p_e2_ewc, p_e2_struct=p_e2_struct,
               b_ewc=b_ewc, b_no=b_no,
               r_a=r_a, r_b=r_b, sep=sep, p_e4=p_e4)
    if verbose:
        print("=" * 64)
        print(f"摊销×MoE 四实验报告 (seed={seed}, n_traj={n_traj}, T={T}, n_ep={n_ep})")
        print("=" * 64)
        print(f"E1 学习强度: indist={mse_indist:.4f} (<0.02) unseen={mse_unseen:.4f} "
              f"→ {'PASS' if p_e1 else 'FAIL'}")
        print(f"E3a 步数差分: clean med={med_c} / noisy med={med_n} / unseen med={med_u} "
              f"→ {'PASS' if p_e3a else 'FAIL'}")
        print(f"E3b 琢磨不劣化: adaptive={mse_ad:.4f} vs fixed1={mse_f1:.4f} "
              f"→ {'PASS' if p_e3b else 'FAIL'}")
        print(f"E2 防遗忘: 有EWC 保留率={ret_ewc:.1%} (≥95%) / 无EWC={ret_no:.1%} (≥80%) "
              f"[B mse: ewc={b_ewc:.4f} no={b_no:.4f}] → {'PASS' if p_e2 else 'FAIL'}")
        if verbose:
            print(f"  [diag] E2 路由: 平手 {m_ewc.pcn.n_novel}/{m_ewc.pcn.n_route} "
                  f"usage={ {k: round(v, 2) for k, v in enumerate(m_ewc.pcn.usage.tolist())} }")
        print(f"E4 路由(A/B): A域={ {k: round(v, 2) for k, v in r_a.items()} } "
              f"B域={ {k: round(v, 2) for k, v in r_b.items()} } "
              f"分离度={sep:.2f} → {'PASS' if p_e4 else 'FAIL'}")
        print(f"综合: E1={'PASS' if p_e1 else 'FAIL'} E2={'PASS' if p_e2 else 'FAIL'} "
              f"E3={'PASS' if p_e3 else 'FAIL'} E4={'PASS' if p_e4 else 'FAIL'}")
        print("=" * 64)
    return out


if __name__ == "__main__":
    run_experiments(seed=0, verbose=True)
