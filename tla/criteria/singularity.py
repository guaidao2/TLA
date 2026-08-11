"""奇点神经元 SN-1..6 判据（预注册，见 docs/奇点神经元_数学推导.md §9）。

判据先于实现——跑数前锁死，不符→如实记录负结果并归因，只许改代码不许改判据。

**实测前分析（2026-08-11，两个推导级发现）**：
1. 原文 ODE 增长项 αe^{βI}(h+ε)(1−h/h_max) 在 I=0 时仍活跃（e^0=1）→ 细胞"恒热"，
   平衡点 ~0.902（α=0.5），幽灵态 h*=(Λ/λ)=0.02 不是全局吸引子 → 预期 SN-2/3/4/6 失败；
   修复变体：α(e^{βI}−1)（input_gated=True），I=0 增长归零，幽灵态成为全局吸引子。
2. §6.2 的 I_th 公式只取 ε 项（αe^{βI}ε > λh^γ），忽略 (h+ε) 的 h 部分——在幽灵态
   (h≈0.02 >> ε) 处阈值被高估 ~60×（文档 I_th≈3，实际≈0.05）。
   → 判据输入值按实际阈值校准：SN-4 弱输入 weak=0.05（明确低于实际 I_th）、
   SN-6 噪声 U(0,0.02)（均值 0.01 < 0.05）。判据本身（"弱输入不暴胀"等）未改。

两变体跑同一判据，如实记录对比。
"""
import random
import math


def run_sn(cell, verbose=False, seed=0):
    rng = random.Random(seed)

    # ---- SN-1 从 0 能炸：h=0、I 大 → 单调升到 ≥0.9·h_max（<100 tick）----
    cell.reset(0.0)
    trace = [0.0]
    hits = None
    for _ in range(100):
        cell.step(1.0)
        trace.append(cell.h)
        if cell.h >= 0.9 * cell.h_max:
            hits = cell.t
            break
    mono = all(b >= a for a, b in zip(trace, trace[1:]))
    sn1 = hits is not None and hits < 100 and mono

    # ---- SN-2 幽灵态：无输入 1000 tick → h 收敛到 h*=(Λ/λ)^{1/γ} ±5% ----
    cell.reset(0.5)
    for _ in range(1000):
        cell.step(0.0)
    hstar = cell.ghost()
    h_noinput = cell.h
    sn2 = abs(h_noinput - hstar) / max(hstar, 1e-12) < 0.05

    # ---- SN-3 时间距离解码：衰减期由 h 反推"上次激活至今 tick"，误差 <5% ----
    cell.reset(0.0)
    for _ in range(500):
        cell.step(5.0)
        if cell.h >= 0.98 * cell.h_max:
            break
    h0 = cell.h
    t0 = cell.t                                   # 衰减计时起点（激励段不计入）
    errs = []
    for target in (0.9, 0.8, 0.7, 0.6, 0.5):
        for _ in range(3000):
            cell.step(0.0)
            if cell.h <= target:
                break
        t_true = cell.t - t0
        if cell.h <= target + 1e-9:
            hstar3 = cell.ghost()
            t_est = (1.0 / cell.lam) * math.log(
                (h0 - hstar3) / max(cell.h - hstar3, 1e-12))
            errs.append(abs(t_est - t_true) / max(t_true, 1e-9))
    sn3 = len(errs) >= 3 and max(errs) < 0.05

    # ---- SN-4 稀疏激活：弱输入（I=0.05 << 实际 I_th≈0.05-0.1）200 tick 内 h < 2·h* ----
    cell.reset(0.0)
    for _ in range(200):
        cell.step(0.05)
    weak_h = cell.h
    sn4 = cell.h < 2.0 * cell.ghost()

    # ---- SN-5 有界性：I∈[0,10] 1000 tick 内 h∈[0, h_max] ----
    cell.reset(0.0)
    ok_bounds = True
    for _ in range(1000):
        cell.step(rng.uniform(0.0, 10.0))
        if not (0.0 <= cell.h <= cell.h_max):
            ok_bounds = False
            break
    sn5 = ok_bounds

    # ---- SN-6 抗噪：噪声（U(0,0.02)，均值 0.01 < 实际 I_th）激活率<10% vs 强输入>90% ----
    cell.reset(0.0)
    n_act_noise = 0
    for _ in range(1000):
        cell.step(rng.uniform(0.0, 0.02))
        if cell.h >= 0.5 * cell.h_max:
            n_act_noise += 1
    cell.reset(0.0)
    n_act_strong = 0
    for _ in range(1000):
        cell.step(5.0)
        if cell.h >= 0.5 * cell.h_max:
            n_act_strong += 1
    rate_noise = n_act_noise / 1000.0
    rate_strong = n_act_strong / 1000.0
    sn6 = rate_noise < 0.10 and rate_strong > 0.90

    return dict(sn1=sn1, sn2=sn2, sn3=sn3, sn4=sn4, sn5=sn5, sn6=sn6,
                hits_sn1=hits, hstar=hstar, h_noinput=h_noinput,
                sn3_errors=errs, weak_h=weak_h,
                rate_noise=rate_noise, rate_strong=rate_strong)


def run_all(verbose=True):
    from tla.substrate.singularity_cell import SingularityCell
    results = {}
    for tag, gated in (("original e^{βI}", False), ("gated e^{βI}−1", True)):
        cell = SingularityCell(input_gated=gated)
        r = run_sn(cell)
        results[tag] = r
        if verbose:
            print(f"  [{tag}] SN-1..6: "
                  + " ".join(f"SN{i}={'PASS' if r[f'sn{i}'] else 'FAIL'}"
                             for i in range(1, 7)))
            print(f"      SN-1 起爆={r['hits_sn1']} tick | SN-2 无输入终值={r['h_noinput']:.3f}"
                  f" (h*={r['hstar']:.3f}) | SN-3 解码误差 max="
                  f"{max(r['sn3_errors']):.2%} ({len(r['sn3_errors'])} 点)"
                  if r['sn3_errors'] else "SN-3 无衰减可解码点")
            print(f"      SN-4 弱输入终值={r['weak_h']:.4f} (2h*={2 * r['hstar']:.4f})"
                  f" | SN-6 噪声激活率={r['rate_noise']:.1%} / 强={r['rate_strong']:.1%}")
    return results


if __name__ == "__main__":
    print("=" * 64)
    print("奇点神经元 SN-1..6（预注册判据，两变体对比）")
    print("=" * 64)
    run_all(verbose=True)
