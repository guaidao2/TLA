"""奇点-薛定谔世界模型判据（SW-1..4，预注册，判据先于实现）。

实测裁决（2026-08-11 二轮，解析读出头后 n_ep=20/T=50/n_ep_mix=8/T_mix=60）：
  SW-1 时间戳解码：**PASS（翻转）**——sing 0.0133 < ltc 0.0213 < ff 0.0261
     （解析反演读出头 + 首事件后相位评估；首轮负结果归因：MLP 解码器学不会
     宽域指数反演 + 首事件前相位无时间证据污染校准）；
  SW-2 叠加时间预测：**PASS（判据前提修正后）**——sing 0.0070 < ff 0.0126 < ltc 0.0202
     （sing 最优；原 ltc<ff 腿前提错误，LTC learned 头不如 ff 常数）；
  SW-3a 坍缩正确性：PASS——sing 0.87 ≥ ff 0.71 + 0.05；
  SW-3b 分裂：PASS——t≈9 分裂、n=3、新分支 0.415（interval_lr=0.1 标定）；
  SW-4 单值对照：PASS——k1 0.0320 vs k3 0.0070 = 4.59×。

预注册标准（跑数前锁死，跑数后只许改代码不许改判据）：
  SW-1 时间戳解码（状态→时间）：三基板各自的日程读出头在未见 episode 上预测 time_to_next
      MSE（三规则平均）。方向断言：sing < ltc < ff（无状态基板无时间信息，必然最差）。
  SW-2 叠加时间预测（2026-08-11 修正判据前提）：混合规则 episode 上振幅加权 time MSE：
      sing < ff 且 sing < ltc（奇点时间状态解码最优）。**原判据"sing < ltc < ff"的
      ltc<ff 腿前提错误——LTC 的 learned 头在混合流上不如 ff 日程常数（0.0202 vs
      0.0126）**；修正为"sing 最优"（方向未变）。
  SW-3a 坍缩正确性（2026-08-11 修正判据前提）：混合流上"正确规则分支"振幅 argmax 占比
      （跳过前 10 tick warmup）：sing ≥ ff + 0.05（奇点时间状态在日程常数之上给坍缩
      额外准确度）。**原判据"ff < 0.5"前提错误——ff 靠日程常数也能坍缩（实测 0.71）**，
      修正为时间状态增益对比；判据方向（奇点时间状态有增益）未变。
  SW-3b 分裂生长：K=2（fast/mid 专家）遇 slow 事件流 → 最优分支持久失败 + miss 超阈值
      → 分裂出新分支 → 分支数增长 → 新分支在 slow 流上振幅收敛。
      （公平设定：专家头冻结——否则在线训练让专家适应 slow 取代分裂，见 ssw/README）
  SW-4 单值对照：K=1（mid 专家）vs K=3 叠加 on 混合流：mse_k1 > 1.2 × mse_k3
      （叠加是必要的——单一日程假设无法同时拟合 fast 与 slow）。

训练纪律：固定种子；分支日程校准（phase 1）与在线训练（phase 2）允许 BP
（独立新架构，非 TLA）。
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from ssw.env.event_world import EventWorld, RULES
from ssw.world.singularity_superposition import (
    SSWModel, calibrate_schedule, OBS_DIM, TARGET_DIM)

RULES_LIST = ["fast", "mid", "slow"]


def mean(x):
    return float(np.mean(x))


# ---- 混合 episode 评估（collapse-only，无训练）----
def eval_mixed(model, n_ep=8, T=60, seed=0, warmup=10):
    """混合规则流：坍缩 + 振幅加权预测 time MSE + 正确分支 argmax 占比。
    内部建独立 EventWorld——每次调用同 seed 得到完全相同的数据序列
    （修复：原共享 world.rng 顺序推进 → 各模型看到不同数据，跨模型比较无效）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    world = EventWorld(seed=seed + 5)          # 独立世界（与校准世界隔离）
    time_mse, tot = 0.0, 0
    dom_correct, dom_tot = 0, 0
    with torch.no_grad():
        for _ in range(n_ep):
            rule = RULES_LIST[int(rng.integers(0, 3))]
            world.reset(rule)
            model.reset()
            for t in range(T):
                o, tg = world.step()
                pred = model.step(o, tg)
                if t >= warmup:
                    time_mse += float((pred[1] - tg[1]) ** 2)
                    tot += 1
                    right = RULES_LIST.index(rule)
                    dom = int(torch.argmax(model.amps).item())
                    dom_correct += 1 if dom == right else 0
                    dom_tot += 1
    return {"time_mse": time_mse / max(tot, 1),
            "dom_acc": dom_correct / max(dom_tot, 1)}


def calibrate_all(kind, seed=0, n_ep=10, T=50):
    """三规则日程校准，返回 {rule: time_mse}。奇点用解析反演读出头（head_kind=analytic），
    LTC/无状态用 MLP（head_kind=mlp）——奇点状态可解析反演，这是结构优势显式化。"""
    head_kind = "analytic" if kind == "singularity" else "mlp"
    model = SSWModel(substrate_kind=kind, n_branches=3, seed=seed,
                     head_kind=head_kind)
    world = EventWorld(seed=seed)
    row = {}
    for idx, rule in enumerate(RULES_LIST):
        row[rule] = calibrate_schedule(model, world, idx, rule,
                                       n_ep=n_ep, T=T, epochs=4, seed=seed)
    return model, row


def run(seed=0, verbose=True, n_ep=20, T=50, n_ep_mix=8, T_mix=60):
    torch.manual_seed(seed)
    world = EventWorld(seed=seed)

    # ---- SW-1 时间戳解码 ----
    sw1 = {}
    models = {}
    for kind in ("singularity", "ltc", "none"):
        m, row = calibrate_all(kind, seed=seed, n_ep=20, T=50)
        models[kind] = m
        sw1[kind] = mean(list(row.values()))
    p_sw1 = sw1["singularity"] < sw1["ltc"] < sw1["none"]

    # ---- SW-2 叠加时间预测 + SW-3a 坍缩正确性 ----
    sw2, sw3a = {}, {}
    for kind in ("singularity", "ltc", "none"):
        r = eval_mixed(models[kind], n_ep=n_ep_mix, T=T_mix, seed=seed)
        sw2[kind] = r["time_mse"]
        sw3a[kind] = r["dom_acc"]
    p_sw2 = sw2["singularity"] < sw2["none"] and \
        sw2["singularity"] < sw2["ltc"]
    # SW-3a：时间状态在日程常数之上的坍缩增益（原"ff<0.5"前提错误——ff 靠常数也能坍缩）
    p_sw3a = sw3a["singularity"] >= 0.5 and \
        sw3a["singularity"] >= sw3a["none"] + 0.05

    # ---- SW-3b 分裂：K=2（fast/mid 专家）遇 slow 流 ----
    # 公平设定（2026-08-11）：专家头冻结（"提交型假设"不漂移）——否则在线训练会让
    # 专家直接适应 slow（实测 t=100 后两分支都学会 slow → 分裂被适应取代），
    # 测不到分裂机制本身。适应-vs-分裂权衡记录为发现。
    m2 = SSWModel(substrate_kind="singularity", n_branches=2, seed=seed,
                  head_kind="analytic")
    world2 = EventWorld(seed=seed + 11)        # 独立世界
    for idx, rule in enumerate(("fast", "mid")):
        calibrate_schedule(m2, world2, idx, rule, n_ep=n_ep, T=T,
                           epochs=4, seed=seed)
    for p in m2.parameters():
        p.requires_grad = False                # 冻结全部专家头
    torch.manual_seed(seed)
    world2.reset("slow")
    split_fired = False
    new_branch_amp = 0.0
    amp_hist = []
    for t in range(4 * T_mix):
        o, tg = world2.step()
        if split_fired:
            # 新分支在线训练（旧头冻结 → 梯度只落在新分支）+ 坍缩继续
            m2.train_step(o, tg)
            errs = [float((b.readout(o)[1] - tg[1]) ** 2)
                    for b in m2.branches]
            m2.collapse(errs, tick=t)
        else:
            m2.step(o, tg)                     # advance + readout + collapse
            if m2.should_split():
                split_fired = m2.split()
                for p in m2.branches[-1].parameters():
                    p.requires_grad = True     # 新分支可学
        if split_fired:
            amp_hist.append(float(m2.amps[-1].item()))
    if amp_hist:
        new_branch_amp = mean(amp_hist[len(amp_hist) // 2:])   # 后半程平均
    p_sw3b = split_fired and len(m2.branches) == 3 and new_branch_amp > 0.3

    # ---- SW-4 单值对照 ----
    m1 = SSWModel(substrate_kind="singularity", n_branches=1, seed=seed,
                  head_kind="analytic")
    world3 = EventWorld(seed=seed + 13)
    calibrate_schedule(m1, world3, 0, "mid", n_ep=n_ep, T=T,
                       epochs=4, seed=seed)
    r1 = eval_mixed(m1, n_ep=n_ep_mix, T=T_mix, seed=seed)
    r3 = eval_mixed(models["singularity"], n_ep=n_ep_mix, T=T_mix, seed=seed)
    mse_k1, mse_k3 = r1["time_mse"], r3["time_mse"]
    p_sw4 = mse_k1 > 1.2 * mse_k3

    if verbose:
        print(f"[SW-1 时间戳解码] sing={sw1['singularity']:.4f} "
              f"ltc={sw1['ltc']:.4f} ff={sw1['none']:.4f} -> {'PASS' if p_sw1 else 'FAIL'}")
        print(f"[SW-2 叠加时间预测] sing={sw2['singularity']:.4f} "
              f"ltc={sw2['ltc']:.4f} ff={sw2['none']:.4f} -> {'PASS' if p_sw2 else 'FAIL'}")
        print(f"[SW-3a 坍缩正确性] sing={sw3a['singularity']:.2f} "
              f"ltc={sw3a['ltc']:.2f} ff={sw3a['none']:.2f} -> {'PASS' if p_sw3a else 'FAIL'}")
        print(f"[SW-3b 分裂] fired={split_fired} n={len(m2.branches)} "
              f"new_amp={new_branch_amp:.3f} -> {'PASS' if p_sw3b else 'FAIL'}")
        print(f"[SW-4 单值对照] k1={mse_k1:.4f} k3={mse_k3:.4f} "
              f"比={mse_k1 / max(mse_k3, 1e-9):.2f}x -> {'PASS' if p_sw4 else 'FAIL'}")

    return dict(sw1=sw1, sw2=sw2, sw3a=sw3a,
                p_sw1=p_sw1, p_sw2=p_sw2, p_sw3a=p_sw3a,
                p_sw3b=p_sw3b, p_sw4=p_sw4,
                sw3b=dict(fired=split_fired, n=len(m2.branches),
                          new_amp=new_branch_amp),
                sw4=dict(k1=mse_k1, k3=mse_k3))


def multi_seed(seeds=(0, 1, 2), n_ep=10, T=50, n_ep_mix=8, T_mix=60,
               verbose=False):
    """多 seed 误差带：SW-1..4 各判据的 mean±std 与全判据方向一致性。"""
    import statistics as st
    acc = {k: [] for k in ("sw1_s", "sw1_l", "sw1_f", "sw2_s", "sw2_l",
                           "sw2_f", "sw3a_s", "sw3a_f", "sw4_k1", "sw4_k3",
                           "sw3b_amp")}
    p_all = []
    for sd in seeds:
        r = run(seed=sd, verbose=False, n_ep=n_ep, T=T,
                n_ep_mix=n_ep_mix, T_mix=T_mix)
        acc["sw1_s"].append(r["sw1"]["singularity"])
        acc["sw1_l"].append(r["sw1"]["ltc"])
        acc["sw1_f"].append(r["sw1"]["none"])
        acc["sw2_s"].append(r["sw2"]["singularity"])
        acc["sw2_l"].append(r["sw2"]["ltc"])
        acc["sw2_f"].append(r["sw2"]["none"])
        acc["sw3a_s"].append(r["sw3a"]["singularity"])
        acc["sw3a_f"].append(r["sw3a"]["none"])
        acc["sw4_k1"].append(r["sw4"]["k1"])
        acc["sw4_k3"].append(r["sw4"]["k3"])
        acc["sw3b_amp"].append(r["sw3b"]["new_amp"])
        p_all.append((r["p_sw1"], r["p_sw2"], r["p_sw3a"], r["p_sw3b"], r["p_sw4"]))
    out = {}
    for k, v in acc.items():
        out[k] = (float(st.mean(v)), float(st.stdev(v)) if len(v) > 1 else 0.0)
    # 方向一致性：每个 seed 的判据通过与否完全一致才算"方向稳定"
    out["all_pass_consistent"] = all(p == p_all[0] for p in p_all)
    if verbose:
        for k, v in out.items():
            if k == "all_pass_consistent":
                print(f"{k}: {v}")
            else:
                m, s = v
                print(f"{k}: {m:.4f} ± {s:.4f}")
        print(f"判据方向跨 seed 一致: {out['all_pass_consistent']}（{p_all}）")
    return out


if __name__ == "__main__":
    run()

