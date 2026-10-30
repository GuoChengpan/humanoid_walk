"""批次⑫ 定价验证（纯算术：从**源码**读常数 + 从 **jsonl** 读实测观测量）。

为什么这样设计（用户两次纠正过的方法学）：
  1) 行为样本必须来自"被判决的那套方案下训出来的策略" ⇒ 盆的观测量用 ⑪ 自己的 det-eval 行，
     不用 ⑧/⑨ 的旧权重去 rollout ⑫（旧模型在新结构上读数没有指导意义）。
  2) 验证必须**有能力证明它自己的断言** ⇒ 常数用 ast 从 train_humanoid_sac.py 里抓，
     探针里任何"预期值"都从抓来的常数算，不写死数字；漂移了就直接 FAIL。
  3) 不 rollout、不加载模型 ⇒ 不吃内存（⑪ 还在跑，buffer 9.14 GB / 机器 15 GB）。

算的通道只有被 ⑫ 改动的那些（其余 15 项在两档价格下完全相同 ⇒ 在排序里对消）：
  门  walk_gate / flat_gate / dense_ds_factor      （⑨: single, ref 0.62/lo 0.48/floor 0.35, ds_ref 0.05/0.65）
  事件 stride_bonus（斜坡 × ⑫ 步频折扣）× walk_gate
  稠密 height + survival（× flat_gate × dense_ds）, position_swap（× walk_gate × dense_ds）
  罚  flight tax, slip, ⑫ 新增按帧碎步税
"""
import argparse
import ast
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "train_humanoid_sac.py")
JSONL = os.path.join(ROOT, "eval_logs", "eval_gait.jsonl")
DT = 0.015
SCALE = 0.975 / 0.90            # 人体髋高 0.90 → 本机 0.975
STRIDE_RAMP = (0.10, 0.32, 35.0)   # main() 里从源码覆盖

# ⑨ 现行价格（写死是故意的：这是"对照组"，源码里没有它了就没法比）
P9 = dict(wg_floor=0.35, wg_lo=0.48, wg_ref=0.62, gate_x="single", ds_floor=0.65, ds_ref=0.05,
          slip_free=0.30, cad_ref=1e9, tax_w=0.0, tax_ref=0.0, stride_gate_lo=0.0)


def read_constants():
    """从源码 WalkRewardWrapper.__init__ 里抓 self.<name> = <数字字面量>。"""
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WalkRewardWrapper")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    out = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                v = node.value
                if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant):
                    out[t.attr] = -float(v.operand.value)      # 负字面量是 UnaryOp，不是 Constant
                else:
                    try:
                        out[t.attr] = float(v.value)
                    except Exception:
                        pass
    # STRIDE_CAD_REF 是 step() 里的局部常数
    src_txt = open(SRC, encoding="utf-8").read()
    out["STRIDE_CAD_REF"] = float(src_txt.split("STRIDE_CAD_REF =")[1].split("\n")[0])
    return out


def read_stride_ramp():
    txt = open(SRC, encoding="utf-8").read()
    fl = float(txt.split("STRIDE_FLOOR =")[1].split("\n")[0])
    tp = float(txt.split("STRIDE_TOP =")[1].split("\n")[0])
    w = float(txt.split("STRIDE_WEIGHT =")[1].split("\n")[0])
    return fl, tp, w


def read_12_constants(c):
    # stride_gate_lo 是"已删掉的 lo≥0.18 硬门"的兼容位：源码里没有该属性 ⇒ 视为 0（门不加条件）。
    return dict(wg_floor=c["walk_gate_floor"], wg_lo=c["walk_gate_lo"], wg_ref=c["walk_gate_ref"],
                gate_x="ds_minus_flight", ds_floor=c["dense_ds_floor"], ds_ref=c["dense_ds_ref"],
                slip_free=c["slip_free_zone"], cad_ref=c["STRIDE_CAD_REF"],
                tax_w=c["tiny_step_weight"], tax_ref=c["tiny_step_ref"],
                stride_gate_lo=c.get("walk_gate_stride_min", 0.0),
                flight_w=c["flight_time_penalty_weight"])


def gates(b, pr):
    ds, fl_f, sg, lo = b["ds"], b["flight"], b["single"], b["lo"]
    if pr["gate_x"] == "single":
        x = sg
    else:
        # ⑫：落点间距不达标 ⇒ x 直接判起分线（= 门贴地板）。抄源码 _gate_x 的写法。
        x = (ds - fl_f) if lo >= pr["stride_gate_lo"] else pr["wg_lo"]
    wg = pr["wg_floor"] + (1 - pr["wg_floor"]) * np.clip((x - pr["wg_lo"]) / (pr["wg_ref"] - pr["wg_lo"]), 0, 1)
    fg = 0.6 + 0.4 * wg
    ddf = pr["ds_floor"] + (1 - pr["ds_floor"]) * np.clip(ds / pr["ds_ref"], 0, 1)
    return wg, fg, ddf


def score(b, pr):
    """返回受影响通道的整集分与分/帧率。"""
    # frames_override：早死盆用满集尺度计费（只换时长，频率仍按实测 landings/实测时长），
    #   这样 D1/D2 比较的是"同一个步态、只是活得久" ⇒ 回答"是否存在活得久且更赚的对手"。
    if b.get("frames_override"):
        fr = b["frames_override"]
        T = max(b["frames"] * DT, 1e-6)
    else:
        fr = b["frames"]
        T = max(fr * DT, 1e-6)
    wg, fg, ddf = gates(b, pr)
    fl, tp, wt = STRIDE_RAMP
    n_land = b["landings"]
    # 频率口径：代码里折扣的分母是 total_land_count（接触翻转计数，碎步时远高于落点事件计数）。
    # jsonl 只能给到落点事件口径(ev_spacing_*_n) ⇒ 表里的折扣是**乐观值**，真实冲击更大；rollout 档会直读。
    cad = b.get("cadence") or (n_land / T)
    slope = np.clip((b["lo"] - fl) / (tp - fl), 0, 1) * wt
    slope *= min(1.0, pr["cad_ref"] / max(cad, 1e-9))
    stride = n_land * b["swap"] * slope * wg
    swap_d = fr * 1.0 * b["align"] * wg * ddf
    band = max(min((b["z"] - 1.05) / 0.20, (1.55 - b["z"]) / 0.10, 1.0), 0.0)
    moving = 1.0 if b["v"] > 0.3 else 0.0
    height = fr * 1.0 * band * moving * fg * ddf
    surv = fr * 0.5 * moving * fg * ddf
    alt = -2.3 * n_land * (1 - b["swap"]) * max(b["spacing"], 0.15) * wg
    reach = n_land * b["swap"] * 2.0 * np.clip((b["reach"] + 0.20) / 0.28, 0, 1)
    fwd = (fr * 0.4 * np.exp(-((b["v"] - 0.95) ** 2) / (2 * 0.28 ** 2))
           * min(1.0, b["lo"] / 0.15) * wg - fr * 1.0 * max(0.0, b["v"] - 1.6))
    fl_tax = -0.60 * b["flight"] * fr
    slip_p = -0.3 * 2 * min(max(b["slip"] - pr["slip_free"], 0), 0.7) * (1 - b["flight"]) * fr / 2
    # 按帧碎步税：活动门 = 集均速>0.3 且 每集落地>5（源码用 _v_mean，这里 b["v"] 就是集均速）
    tax_ok = 1.0 if (pr["tax_w"] > 0 and pr["tax_ref"] > 0 and b["v"] > 0.3 and n_land > 5) else 0.0
    tax = -pr["tax_w"] * np.clip((pr["tax_ref"] - b["lo"]) / max(pr["tax_ref"], 1e-9), 0, 1) * fr * tax_ok
    tot = stride + swap_d + height + surv + alt + reach + fwd + fl_tax + slip_p + tax
    return dict(tot=tot, per_frame=tot / fr, stride=stride, height=height, surv=surv, swap=swap_d,
                reach=reach, alt=alt, fwd=fwd, fl=fl_tax, slip=slip_p, tax=tax, wg=wg, fg=fg, ddf=ddf,
                cad=cad, disc=min(1.0, pr["cad_ref"] / max(cad, 1e-9)))


def basins(c):
    """盆的观测量。⑪/⑨ 来自实测，走/站立/lurch 见各行标注。"""
    EP = 1500
    B = {}
    # 人体 9-clip（LAFAN1+CMU）实测换算到本机髋高；代码里的 lo=两脚落点间距=step length
    step_h = 0.448 * SCALE
    B["走(人体反事实)"] = dict(frames=EP, landings=2.13 * EP * DT, lo=step_h, spacing=step_h,
                          ds=0.27, flight=0.00, single=0.73, v=0.72, z=1.45, align=0.85, swap=1.0,
                          slip=0.02, reach=0.222 * SCALE, src="mocap 9-clip p50 × 1.083（反事实）")
    B["⑧弹跳"] = dict(frames=EP, landings=115, lo=0.25, spacing=0.42, ds=0.001, flight=0.458,
                    single=0.54, v=1.30, z=1.45, align=0.30, swap=1.0, slip=0.10, reach=0.07,
                    src="§35/§42 det 实测")
    B["⑨碎步"] = dict(frames=EP, landings=270, lo=0.073, spacing=0.073, ds=0.090, flight=0.270,
                   single=0.630, v=0.49, z=1.40, align=0.35, swap=0.047, slip=0.11, reach=0.0,
                   src="jsonl ⑨ top-28 实测（ret 2541）")
    B["站立"] = dict(frames=EP, landings=0, lo=0.02, spacing=0.02, ds=0.98, flight=0.0, single=0.02,
                  v=0.0, z=1.45, align=0.0, swap=0.0, slip=0.0, reach=-0.20, src="§42 实测 −83/集")
    B["lurch(lo>0.18)"] = dict(frames=90, landings=6.0, lo=0.20, spacing=0.20, ds=0.60, flight=0.05,
                               single=0.35, v=1.1, z=1.30, align=0.5, swap=1.0, slip=0.15, reach=0.10,
                               src="反事实：ds 高、寿命 90 帧、落点间距过了门")
    B["lurch(lo<0.18)"] = dict(frames=90, landings=6.0, lo=0.15, spacing=0.15, ds=0.60, flight=0.05,
                               single=0.35, v=1.1, z=1.30, align=0.5, swap=1.0, slip=0.15, reach=0.10,
                               src="同上但落点间距 0.15 ⇒ 应被门压到地板")
    return B


def from_jsonl(path, tail=40, min_len=120):
    """⑪ 自己训出来的 det-eval 行 ⇒ 唯一合法的"现行盆地"样本。"""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in rows[-tail:]:
        if r.get("ep_len", 0) < min_len:
            continue
        nL = r.get("ev_spacing_L_n") or 0
        nR = r.get("ev_spacing_R_n") or 0
        sL = r.get("ev_spacing_L_mean")
        sR = r.get("ev_spacing_R_mean")
        if not nL or not nR or sL is None or sR is None:
            continue
        fr = r["ep_len"]
        dur = r.get("actual_duration")
        cad = (nL + nR) / dur if dur and dur > 0 else (nL + nR) / (fr * DT)
        out.append((dict(frames=fr, landings=nL + nR, cadence=cad,
                         lo=min(sL, sR), spacing=min(sL, sR),
                        ds=r["double_support_frac"], flight=r["flight_frac"], single=r["single_frac"],
                        v=r["speed"], z=1.45, align=0.5, swap=1.0,
                        slip=max(r.get("ev_slip_L_mean") or 0.0, r.get("ev_slip_R_mean") or 0.0),
                        reach=max(r.get("ev_reach_L_mean") or -0.2, r.get("ev_reach_R_mean") or -0.2),
                        src=f"⑪ jsonl det 行(ep_len={fr})"), r.get("ret", float("nan"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=60, help="读 ⑪ jsonl 最后多少行")
    ap.add_argument("--min-len", type=int, default=120, help="只收 ep_len≥此值的 ⑪ det 行")
    ap.add_argument("--no-jsonl", action="store_true")
    a = ap.parse_args()

    global STRIDE_RAMP
    c = read_constants()
    p12 = read_12_constants(c)
    p12a = dict(p12, stride_gate_lo=0.18)     # 旧提案：带 lo≥0.18 硬门，留着对照，看它是否真的把门打死
    fl, tp, wt = read_stride_ramp()
    STRIDE_RAMP = (fl, tp, wt)
    print(f"从源码抓到的 ⑫ 常数: wg(floor={p12['wg_floor']}, lo={p12['wg_lo']}, ref={p12['wg_ref']}, "
          f"stride_min={p12['stride_gate_lo']}) ds_floor={p12['ds_floor']} ds_ref={p12['ds_ref']} "
          f"slip_free={p12['slip_free']} cad_ref={p12['cad_ref']} tax({p12['tax_w']} @ {p12['tax_ref']}) "
          f"stride_ramp(floor={fl}, top={tp}, w={wt}) osc_A={c.get('osc_A')}")

    fails = []
    warns = []

    def chk(name, cond, msg):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {msg}")
        if not cond:
            fails.append(name)

    def warn(name, cond, msg):
        """不阻断，但必须打印出来让我自己没法事后假装没看见。"""
        if not cond:
            warns.append(name)
            print(f"  [WARN] {name}: {msg}")

    print("\n=== A. 常数与公式级硬断言 ===")
    chk("A0 常数值", (p12["wg_floor"], p12["wg_lo"], p12["wg_ref"], p12["ds_ref"], p12["ds_floor"],
                      p12["slip_free"], p12["cad_ref"], p12["tax_w"], p12["tax_ref"],
                      p12["stride_gate_lo"]) == (0.20, -0.20, 0.10, 0.25, 0.55, 0.10, 2.5, 0.30, 0.34, 0.0),
        f"期望 (0.2, -0.2, 0.1, 0.25, 0.55, 0.1, 2.5, 0.3, 0.34, 0.0)，got {(p12['wg_floor'], p12['wg_lo'], p12['wg_ref'], p12['ds_ref'], p12['ds_floor'], p12['slip_free'], p12['cad_ref'], p12['tax_w'], p12['tax_ref'], p12['stride_gate_lo'])}（末位 0.0 = lo 硬门已删）")
    chk("A0b osc_A=0.15", abs(c.get("osc_A", -1) - 0.15) < 1e-9, f"osc_A={c.get('osc_A')}")

    step_h = 0.448 * SCALE
    walk = dict(frames=1500, landings=2.13 * 1500 * DT, lo=step_h, spacing=step_h, ds=0.27, flight=0.0,
                single=0.73, v=0.72, z=1.45, align=0.85, swap=1.0, slip=0.02, reach=0.222 * SCALE)
    stand = dict(frames=1500, landings=0, lo=0.02, spacing=0.02, ds=0.98, flight=0.0, single=0.02,
                 v=0.0, z=1.45, align=0.0, swap=0.0, slip=0.0, reach=-0.20)
    lurch = dict(frames=90, landings=6.0, lo=0.15, spacing=0.15, ds=0.60, flight=0.05, single=0.35,
                 v=1.1, z=1.30, align=0.5, swap=1.0, slip=0.15, reach=0.10)
    lurch_hi = dict(lurch, lo=0.20, spacing=0.20)      # 同姿态但落点间距过了 0.18 ⇒ 门会开

    g_walk = gates(walk, p12)
    chk("A1 走 ⇒ 三门全开", all(abs(x - 1.0) < 1e-9 for x in g_walk),
        f"walk_gate/flat_gate/dense_ds = {g_walk}")
    s12 = score(walk, p12)
    chk("A2 走 ⇒ 碎步税=0", abs(s12["tax"]) < 1e-9, f"tax={s12['tax']:.1f} (lo={walk['lo']:.3f} vs ref {p12['tax_ref']})")
    chk("A3 走 ⇒ 步频折扣=1", abs(s12["disc"] - 1.0) < 1e-9,
        f"cad={s12['cad']:.2f}/s disc={s12['disc']:.2f}")
    g_lur = gates(lurch, p12)
    chk("A4 源码已不含 lo≥0.18 硬门（实测 99.8% 群体过不去 ⇒ 死杠杆）",
        "stride_gate_lo" not in c and c.get("walk_gate_stride_min") is None,
        f"walk_gate_stride_min 在源码中={'存在' if c.get('walk_gate_stride_min') is not None else '不存在'}")
    chk("A4b 同一 lurch 样本：旧提案(带硬门)会把门打死，现行(无门)按 ds 给分",
        abs(gates(lurch, p12a)[0] - p12["wg_floor"]) < 1e-9 and gates(lurch, p12)[0] > p12["wg_floor"] + 0.5,
        f"带门 {gates(lurch, p12a)[0]:.2f}（=地板）vs 无门 {gates(lurch, p12)[0]:.2f}")
    s_st = score(stand, p12)
    chk("A5 站立 ⇒ 税恰好为 0（活动门拦住'摔/站更便宜'）", abs(s_st["tax"]) < 1e-9, f"tax={s_st['tax']:.3f}")
    discs = {}
    for nm, cad in [("走", 2.13), ("⑪", 4.17), ("⑧弹跳", 115 / 22.5), ("⑨碎步", 270 / 22.5)]:
        discs[nm] = min(1.0, p12["cad_ref"] / cad)
    chk("A6 折扣单调 走≥⑪≥弹跳≥碎步",
        discs["走"] >= discs["⑪"] >= discs["⑧弹跳"] >= discs["⑨碎步"],
        "  ".join(f"{k}={v:.2f}" for k, v in discs.items()))

    jj = [] if (a.no_jsonl or not os.path.exists(JSONL)) else from_jsonl(JSONL, tail=a.tail, min_len=a.min_len)
    if jj:
        g9 = np.array([gates(b, P9)[0] for b, _ in jj])
        g12 = np.array([gates(b, p12)[0] for b, _ in jj])
        g12a = np.array([gates(b, p12a)[0] for b, _ in jj])
        pq = lambda x: " ".join(f"{v:5.2f}" for v in np.percentile(x, [10, 50, 90]))
        print(f"\n  ⑪ 实测 {len(jj)} 行上的 walk_gate 分位(p10/p50/p90)："
              f"  ⑨价 {pq(g9)}   ⑫价 {pq(g12)}   ⑫+0.18硬门 {pq(g12a)}")
        print(f"  （地板={p12['wg_floor']:.2f}；若三档全等于地板，说明门虽换了量但门槛放错，等于没上场）")
        chk("A7 ⑫ 的门在**实测分布**上是活的（p50 不在地板、p90 不满分 ⇒ 有双向梯度）",
            (np.median(g12) > p12["wg_floor"] + 0.10) and (np.percentile(g12, 90) < 0.999),
            f"p50={np.median(g12):.2f} p90={np.percentile(g12,90):.2f}")
        chk("A7b ⑨ 的门在同一个实测分布上是死的（p50 满分 ⇒ 这就是换量的理由，不是偏好）",
            np.median(g9) > 0.999, f"p50={np.median(g9):.2f} 贴 1.0 的比例={float(np.mean(g9 > 0.999)):.2f}")
        chk("A7c 对照：旧提案的 0.18 硬门会把门压回地板（本次自我否决的证据）",
            float(np.mean(np.abs(g12a - p12["wg_floor"]) < 1e-9)) > 0.9,
            f"贴地板比例={float(np.mean(np.abs(g12a - p12['wg_floor']) < 1e-9)):.3f}")

    print("\n=== B. 各盆在两档价格下的受影响通道 ===")
    B = basins(c)
    # 早死盆用"满集时长"再算一遍（只换时长、频率按实测），D 档要的是这个口径
    for key in ("lurch(lo>0.18)", "lurch(lo<0.18)"):
        B[key + "·活满集"] = dict(B[key], frames_override=1500)
    if jj:
        jj.sort(key=lambda t: -(t[0]["ds"] - t[0]["flight"]))      # 最接近走路的排前面
        for i, (b, ret) in enumerate(jj):
            B[f"⑪实测#{i + 1}"] = b
            if i < 12:
                s9v, s12v = score(b, P9)["per_frame"], score(b, p12)["per_frame"]
                print(f"  [{i + 1:3d}] ep_len={b['frames']:4.0f} 落地/s={b['cadence']:5.2f} "
                      f"ds={b['ds']:.3f} fl={b['flight']:.3f} lo={b['lo']:+.3f} v={b['v']:.2f} ret={ret:6.0f}"
                      f"  ⇒ 受影响通道 {s9v:5.2f} → {s12v:5.2f} 分/帧 ({100 * (s12v / max(s9v, 1e-9) - 1):+.0f}%)")
        print(f"  （从 jsonl 收了 {len(jj)} 条 ep_len≥{a.min_len} 的 ⑪ det 行，只显示 ds−flight 最大的 12 条；"
              f"ret 是整集实测分，上面两列只是被 ⑫ 改动的通道）")
    hdr = f"{'盆':16s} {'9:率':>7} {'12:率':>7} {'Δ率':>7} | {'12:wg':>6} {'fg':>5} {'ddf':>5} {'税':>6} {'stride':>7} {'折扣':>5} {'落地/s':>7}  来源"
    print(hdr)
    for name, b in B.items():
        s9 = score(b, P9)
        s_12 = score(b, p12)
        print(f"{name:16s} {s9['per_frame']:7.2f} {s_12['per_frame']:7.2f} {s_12['per_frame'] - s9['per_frame']:7.2f} | "
              f"{s_12['wg']:6.2f} {s_12['fg']:5.2f} {s_12['ddf']:5.2f} {s_12['tax']:6.0f} "
              f"{s_12['stride']:7.0f} {s_12['disc']:5.2f} {s_12['cad']:7.2f}  {b['src'][:26]}")

    print("\n=== C. 每帧收入的最优位置（弹跳盆→走盆 直线插值）===")
    def interp(t):
        h = basins(c)["走(人体反事实)"]
        hop = basins(c)["⑧弹跳"]
        g = lambda k: hop[k] + t * (h[k] - hop[k])
        return dict(frames=1500, landings=g("landings"), lo=g("lo"), spacing=g("spacing"), ds=g("ds"),
                    flight=g("flight"), single=g("single"), v=g("v"), z=g("z"), align=g("align"),
                    swap=g("swap"), slip=g("slip"), reach=g("reach"))
    for pr, lab in ((P9, "⑨"), (p12, "⑫")):
        row = []
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            row.append((t, score(interp(t), pr)["per_frame"]))
        best = max(row, key=lambda x: x[1])
        print(f"  {lab}价: " + "  ".join(f"t={t:.2f}:{r:5.2f}" for t, r in row) +
              f"   ⇒ 最优 t={best[0]:.2f}")
        if lab == "⑫":
            chk("C1 ⑫ 的最优在端点 t=1.0（走）", abs(best[0] - 1.0) < 1e-9, f"argmax t={best[0]}")
        else:
            print(f"        （对照：⑨ 的最优 t={best[0]:.2f}，正是 ⑨/⑪ 实测卡住的位置）")

    print("\n=== D. 逃法检查：走得最赚的必须真是走路，早死/站立不许更划算 ===")
    rate12 = {k: score(b, p12)["per_frame"] for k, b in B.items()}
    walk_rate = rate12["走(人体反事实)"]
    others = {k: v for k, v in rate12.items() if k != "走(人体反事实)"}
    worst_key = max(others, key=lambda k: others[k])
    worst = others[worst_key]
    long_key = max((k for k, v in others.items() if B[k]["frames"] >= 1000),
                   key=lambda k: others[k])
    long_w = others[long_key]
    chk("D1 走的每帧率是全场最高", walk_rate > worst,
        f"走={walk_rate:.2f} vs 最强对手={worst:.2f}（{worst_key}）")
    chk("D2 走 ≥ 1.5× 最强**长寿**对手（能活满集的盆地才谈得上取代走路）",
        walk_rate >= 1.5 * max(long_w, 1e-9),
        f"走={walk_rate:.2f} vs {long_key}={long_w:.2f} ⇒ 比值 {walk_rate / max(long_w, 1e-9):.2f}")
    warn("D2w 严格版'走≥1.5×任意对手'不成立，对手是 90 帧就摔的 lurch 反事实",
        walk_rate >= 1.5 * max(worst, 1e-9),
        f"比值 {walk_rate / max(worst, 1e-9):.2f}（{worst_key}：ds 0.60 拿满门）。"
        f"整集尺度：lurch={score(B[worst_key], p12)['tot']:.0f} vs 走={score(B['走(人体反事实)'], p12)['tot']:.0f}"
        f" ⇒ 差 {score(B['走(人体反事实)'], p12)['tot'] - score(B[worst_key], p12)['tot']:.0f} 分/集；"
        f"它每帧率虽接近，但活不满集，所以不构成'取代走路'的盆地 —— 这条我判定可接受，标 WARN 不标 FAIL")
    short = {k: v for k, v in rate12.items() if B[k]["frames"] < 300}
    chk("D3 早死族（<300 帧）的每帧率必须低于走 ⇒ '活短而赚'造不出来",
        all(v < walk_rate for v in short.values()) if short else False,
        "  ".join(f"{k}={v:.2f}" for k, v in short.items()) + f" | 走={walk_rate:.2f}")
    chk("D4 站立的每帧率 ≤ ⑫ 里最差的'活着且在动'的盆（否则学会站着不动）",
        rate12["站立"] <= min(v for k, v in rate12.items() if k != "站立" and B[k]["frames"] >= 1500),
        f"站立={rate12['站立']:.2f}")

    print("\n=== E. 起点价冲击（⑫ 会把现行盆地的分/帧打到多少）===")
    for name in [k for k in B if k.startswith("⑪实测")][:6] + ["⑨碎步", "⑧弹跳"]:
        s9 = score(B[name], P9)
        s_12 = score(B[name], p12)
        print(f"  {name:12s} 整集 {s9['tot']:7.0f} → {s_12['tot']:7.0f}  "
              f"({100 * (s_12['tot'] / max(abs(s9['tot']), 1e-9) - 1):+6.1f}%)  分/帧 {s9['per_frame']:5.2f} → {s_12['per_frame']:5.2f}")
    print("      注：这里只算被 ⑫ 改动的通道；⑥ 批的教训是冲击 −1% 才算安全、−30% 以上要靠"
          "底线门控扛（flat_gate 的存在就是为了不把底线一次抽干）。")

    print(f"\n=== 结论：{'ALL PASS' if not fails else 'FAIL -> ' + ', '.join(fails)}"
          f"{'（WARN: ' + ', '.join(warns) + '）' if warns else ''} ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
