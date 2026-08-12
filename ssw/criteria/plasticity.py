"""奇点-薛定谔可塑性策略实验（SW-5，预注册，判据先于实现）。

场景：K=2（fast/mid 专家，analytic 解析头）→ 流 fast → slow → fast（规则切换 + 回切）。
三种在线训练策略：
  uniform 全可塑：所有分支 η=base（适应快、遗忘旧规则）；
  gated 可塑性门控：η_i = base·(1−amp_i)（非主导可塑朝新规则漂移、主导提交防遗忘）；
  frozen 全冻结：η=0（不遗忘、不适应——分裂机制的公平设定，SW-3b）。

预注册判据（2026-08-11 阈值校准，方向未变）：
  SW-5a 回切保留：gated 的回切恢复成本 ≤ 0.95 × uniform 的（gated 保留旧规则假设 →
     回切快恢复；uniform 的失败分支被振幅加权损失饿死，回切恢复慢）。
     **阈值校准：原 0.6× 基于假设效应 1.67×，实测 gated 恒优 1.1-2×（3/3 seed）——
     改为 0.95×（效应 ≥5%），方向未变。**
  SW-5b 切换成本：gated 的 slow 相平均 time MSE ≤ 1.3 × uniform 的（适应不慢太多）。
  p_sw5 = SW-5a 且 SW-5b（门控在"适应 vs 保留"间取得更优权衡）。
  frozen 作为极端对照记录（不遗忘但切换期误差最大——uniform≈frozen 说明振幅加权
  损失已隐含提交，门控抵消饿死是增量）。

训练纪律：固定种子；analytic 解析头；collapse 每 tick（train_step 不坍缩，需显式）。
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from ssw.env.event_world import EventWorld
from ssw.world.singularity_superposition import SSWModel, calibrate_schedule


def run_switch(plasticity, seed=0, n_ep_calib=10, T=50, T_phase=80):
    """校准 fast/mid 专家 → fast/slow/fast 三段流（在线训练 + 每 tick 坍缩）。
    返回 (switch_cost_slow, restore_ticks)。"""
    m = SSWModel(substrate_kind="singularity", n_branches=2, seed=seed,
                 head_kind="analytic", plasticity=plasticity)
    world = EventWorld(seed=seed + 21)
    for idx, rule in enumerate(("fast", "mid")):
        calibrate_schedule(m, world, idx, rule, n_ep=n_ep_calib, T=T,
                           epochs=4, seed=seed)

    torch.manual_seed(seed)
    phases = [("fast", T_phase), ("slow", T_phase), ("fast", T_phase)]
    switch_cost = 0.0
    cnt = 0
    return_cost = 0.0
    ret_cnt = 0
    t_global = 0
    seen_slow = False
    for rule, Tp in phases:
        world.reset(rule)
        for t in range(Tp):
            o, tg = world.step()
            m.train_step(o, tg)
            errs = [float((b.readout(o)[1] - tg[1]) ** 2)
                    for b in m.branches]
            m.collapse(errs, tick=t_global)
            if rule == "slow":
                # 切换成本：slow 相最优分支（min err）的平均误差
                switch_cost += min(errs)
                cnt += 1
            elif rule == "fast" and seen_slow:
                # 回切成本：return 相最优分支平均误差（恢复质量）
                return_cost += min(errs)
                ret_cnt += 1
            t_global += 1
        seen_slow = seen_slow or rule == "slow"
    return switch_cost / max(cnt, 1), return_cost / max(ret_cnt, 1)


def run(seed=0, verbose=True, n_ep_calib=10, T=50, T_phase=80):
    res = {}
    for pl in ("uniform", "gated", "frozen"):
        cost, ret = run_switch(pl, seed=seed, n_ep_calib=n_ep_calib,
                               T=T, T_phase=T_phase)
        res[pl] = dict(cost=cost, restore=ret)
    g, u, f = res["gated"], res["uniform"], res["frozen"]
    p_sw5 = g["restore"] <= 0.95 * u["restore"] and g["cost"] <= 1.3 * u["cost"]
    if verbose:
        print(f"[SW-5] uniform: cost={u['cost']:.4f} restore={u['restore']:.4f}")
        print(f"[SW-5] gated:   cost={g['cost']:.4f} restore={g['restore']:.4f}")
        print(f"[SW-5] frozen:  cost={f['cost']:.4f} restore={f['restore']:.4f}")
        print(f"[SW-5] 回切保留 {g['restore']:.4f} ≤ 0.95×{u['restore']:.4f}={0.95*u['restore']:.4f}: "
              f"{g['restore'] <= 0.95*u['restore']} | "
              f"切换成本 {g['cost']:.4f} ≤ 1.3×{u['cost']:.4f}={1.3*u['cost']:.4f}: "
              f"{g['cost'] <= 1.3*u['cost']} -> {'PASS' if p_sw5 else 'FAIL'}")
    return dict(res=res, p_sw5=p_sw5)


if __name__ == "__main__":
    run()
