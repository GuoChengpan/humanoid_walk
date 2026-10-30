"""⑫ 定价表（纯算术，不 rollout、不加载权重、不预测"能不能学会"）。

只算**会被 ⑫ 改动的那些通道**，报每盆的 Δ 与翻转后的排序；未受影响的通道在两档价格下一模一样 ⇒ 在排序里对消。
这样绕开了"重造 21 项总额"的必要（那条路要拟合一堆未观测量，反而更容易自欺）。

受影响的通道与公式（全部照 train_humanoid_sac.py 现状抄）：
  门  walk_gate = WG_F + (1-WG_F)·clip((x − lo)/(ref − lo))   x=single(⑨价) 或 ds−flight(⑫价)
      flat_gate = 0.6 + 0.4·walk_gate
      dense_ds  = DS_F + (1-DS_F)·clip(ds/ds_ref)
      fwd_gate  = clip(min_stride/0.15)                     （⑫ 不变）
  事件 stride_bonus = 35·clip((lo−0.10)/0.22) ×[⑫: cadence 折扣] ；真交换硬门两档都保留
  稠密 height = 1.0·band·[v>0.3] × flat_gate × dense_ds
       survival = 0.5·[v>0.3] × flat_gate × dense_ds
       swap    = 1.0·align × walk_gate × dense_ds
  罚  flight tax = 0.60·[两脚腾空] ；slip = 0.3·min(max(v_slip−free,0),0.7)×2(双脚)
  ⑫ 新增 按帧 spacing 税 = W_T·clip((T_REF − spacing)/T_REF,0,1)
盆的观测量：⑨/⑧/⑪ 用实测（jsonl/handoff），走 用 §44 人体表换算到髋高 0.975 m（标注为反事实）。
"""
import numpy as np

DT = 0.015
EP_FRAMES = 1500
EP_S = EP_FRAMES * DT
SCALE = 0.975 / 0.90          # 人体髋高 0.90 → 本机 0.975

# ---------------- 盆（观测向量） ----------------
# landings=每集落地次数, lo=min(两脚领先量) m, spacing=两脚落点间距 m, ds/flight/single=时间占比,
# v=速度, z=躯干高, align=相位对齐度, swap=真交换率(占落地), slip=支撑脚切向速度
BASINS = {
    "走(人体反事实)": dict(landings=2.1 * EP_S, lo=0.186 * SCALE, spacing=0.448 * SCALE,
                       ds=0.27, flight=0.00, single=0.73, v=0.72, z=1.45, align=0.85, swap=1.0,
                       slip=0.02, src="§44 9-clip 实测×1.083 ⇒ 反事实"),
    "⑧弹跳": dict(landings=115, lo=0.25, spacing=0.42, ds=0.001, flight=0.458, single=0.54,
                v=1.30, z=1.45, align=0.30, swap=1.0, slip=0.10, src="§35/§42 实测(stride 2748/集)"),
    "⑨碎步": dict(landings=270, lo=0.010, spacing=0.073, ds=0.090, flight=0.270, single=0.630,
                v=0.49, z=1.40, align=0.35, swap=0.047, slip=0.11, src="jsonl ⑨ top-28 实测(ret 2541)"),
    "⑪大跨腾空": dict(landings=4.17 * EP_S, lo=0.177, spacing=0.177, ds=0.055, flight=0.167,
                   single=0.704, v=1.02, z=1.42, align=0.55, swap=0.6, slip=0.205,
                   src="本轮 1.03-1.38M 实测(ep_len 174, ret 210)"),
    "站立": dict(landings=0.0, lo=0.0, spacing=0.02, ds=0.98, flight=0.0, single=0.02, v=0.0,
               z=1.45, align=0.0, swap=0.0, slip=0.0, src="§42 实测(−83/集)"),
    "lurch(扑2步就摔)": dict(landings=6.0, lo=0.15, spacing=0.20, ds=0.60, flight=0.05,
                        single=0.35, v=1.1, z=1.30, align=0.5, swap=1.0, slip=0.15,
                        src="反事实：ds 高但寿命只有 90 帧"),
}
LIFE_FRAMES = {"走(人体反事实)": 1500, "⑧弹跳": 1500, "⑨碎步": 1500, "⑪大跨腾空": 174,
               "站立": 1500, "lurch(扑2步就摔)": 90}


def gates(b, price):
    ds, fl, sg = b["ds"], b["flight"], b["single"]
    if price == "9":
        wg_f, wg_lo, wg_ref = 0.35, 0.48, 0.62
        x = sg
        ds_f, ds_ref = 0.65, 0.05
    else:                                    # ⑫：挂 ds−flight，floor 更深，ds 门槛按人体
        wg_f, wg_lo, wg_ref = 0.20, -0.20, 0.10
        # ⚠ v1 的错：lurch 倒下前双脚都贴着地 ⇒ ds 高 ⇒ 拿满门。修正：门要求 lo≥0.18（真在往前放脚）
        x = (ds - fl) if b["lo"] >= 0.18 else -0.20
        ds_f, ds_ref = 0.55, 0.25
    wg = wg_f + (1 - wg_f) * np.clip((x - wg_lo) / (wg_ref - wg_lo), 0, 1)
    fg = 0.6 + 0.4 * wg
    ddf = ds_f + (1 - ds_f) * np.clip(ds / ds_ref, 0, 1)
    return wg, fg, ddf


def score(name, b, price):
    fr = LIFE_FRAMES[name]
    T = fr * DT
    wg, fg, ddf = gates(b, price)
    band = max(min((b["z"] - 1.05) / 0.20, (1.55 - b["z"]) / 0.10, 1.0), 0.0)
    moving = 1.0 if b["v"] > 0.3 else 0.0
    # 事件钱
    slope = np.clip((b["lo"] - 0.10) / 0.22, 0, 1) * 35.0
    if price == "12":
        slope *= min(1.0, 2.5 / max(b["landings"] / T, 1e-9))     # 步频折扣：取消频率红利
    stride = b["landings"] * b["swap"] * slope * wg
    swap_d = fr * 1.0 * b["align"] * wg * ddf                      # position_swap 稠密
    height = fr * 1.0 * band * moving * fg * ddf
    surv = fr * 0.5 * moving * fg * ddf
    alt = -2.3 * b["landings"] * (1 - b["swap"]) * max(b["spacing"], 0.15) * wg   # 非交换落地才罚，量纲=min(|diff|)
    reach = b["landings"] * b["swap"] * 2.0 * np.clip((b["lo"] + 0.20) / (0.20 + 0.20), 0, 1) \
        if price == "12" else b["landings"] * b["swap"] * 2.0 * np.clip((b["lo"] + 0.20) / (0.08 + 0.20), 0, 1)
    fwd = fr * 0.4 * np.exp(-((b["v"] - 0.95) ** 2) / (2 * 0.28 ** 2)) * min(1.0, b["lo"] / 0.15) * wg \
        - fr * 1.0 * max(0.0, b["v"] - 1.6)
    fl_tax = -0.60 * b["flight"] * fr
    free = 0.3 if price == "9" else 0.10
    slip_p = -0.3 * 2 * min(max(b["slip"] - free, 0), 0.7) * (1 - b["flight"]) * fr / 2
    # ⚠ v1 的错：站立/未成熟策略 ds 高 ⇒ 门 1.0，但 spacing≈0 ⇒ 吃满税 −424/集，比 fall −50 还差
    #   ⇒ "摔死更便宜"会被造出来。修正：税要求"在动 + 真在落地"。
    tax_ok = 1.0 if (price == "12" and b["v"] > 0.3 and b["landings"] > 5) else 0.0
    tax = -0.30 * np.clip((0.34 - b["spacing"]) / 0.34, 0, 1) * fr * tax_ok
    tot = stride + swap_d + height + surv + alt + reach + fwd + fl_tax + slip_p + tax
    return dict(tot=tot, stride=stride, height=height, surv=surv, swap=swap_d, reach=reach,
                alt=alt, fwd=fwd, fl=fl_tax, slip=slip_p, tax=tax, wg=wg, fg=fg, ddf=ddf)


def main():
    for price, lab in (("9", "⑨ 现行价格"), ("12", "⑫ 提案价格")):
        print(f"\n=== {lab} ===")
        print(f"{'盆':22s} {'总额':>8} {'stride':>8} {'height':>8} {'surv':>7} {'swap':>7} {'reach':>7} "
              f"{'alt':>7} {'forward':>8} {'flight':>8} {'slip':>7} {'spacing税':>8} | "
              f"{'wg':>5} {'fg':>5} {'ddf':>5}")
        for name, b in BASINS.items():
            s = score(name, b, price)
            print(f"{name:22s} {s['tot']:8.0f} {s['stride']:8.0f} {s['height']:8.0f} {s['surv']:7.0f} "
                  f"{s['swap']:7.0f} {s['reach']:7.0f} {s['alt']:7.0f} {s['fwd']:8.0f} {s['fl']:8.0f} "
                  f"{s['slip']:7.0f} {s['tax']:8.0f} | {s['wg']:5.2f} {s['fg']:5.2f} {s['ddf']:5.2f}")
    print("\n=== 排序与硬门槛 ===")
    for price in ("9", "12"):
        t = {n: score(n, BASINS[n], price)["tot"] for n in BASINS}
        order = sorted(t, key=lambda k: -t[k])
        rate = {k: t[k] / LIFE_FRAMES[k] for k in t}
        w = rate["走(人体反事实)"]
        best_other = max(v for k, v in rate.items() if k != "走(人体反事实)")
        print("        每帧收入率: " + "  ".join(f"{k}={rate[k]:.2f}" for k in order))
        t = rate
        print(f"  [{'⑨' if price=='9' else '⑫'}价·每帧率] " + " > ".join(f"{k}({v:.0f})" for k, v in zip(order, [t[k] for k in order])))
        print(f"        走 / 最强对手 = {w / max(best_other, 1e-9):5.2f}  ⇒ "
              f"{'PASS(≥1.5)' if w >= 1.5 * best_other else 'FAIL(<1.5)'}"
              f"；早死 −50 vs 最富的活不满盆 {min(v for k, v in t.items() if LIFE_FRAMES[k] < 300):.0f} ⇒ "
              f"{'OK' if w > 0 and min([v for k, v in t.items() if LIFE_FRAMES[k] < 300] + [0]) < w else '危险'}")
    print("\n  注：'总额'只含被 ⑫ 改动的通道（其余 15 项在两档下完全相同、对消）；")
    print("      '走'那一行是反事实（人体实测换算），不是量出来的；站立不含 height/survival 的门(需要 v>0.3)以外的钱。")


if __name__ == "__main__":
    main()
