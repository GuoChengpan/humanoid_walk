"""批次⑪·T1 的判决脚本（阈值全部来自 CONTEXT_HANDOFF §46，写在观察之前，事后不挪）。

读 `eval_logs/eval_gait.jsonl`（每条 eval 写 5 个 det episode）+ 最新 TB 事件文件。
行号→env 步数换算：eval_freq=2000 梯度步 × train_freq=512 ÷ 每步 1 次更新 ⇒ **16K env 步/次 eval**，
每次 5 集 ⇒ 1 行 ≈ 3206 env 步（与 ⑨ 的 555 行 / 1.78M 步吻合，作为自检）。

预注册判据（§46）：
  K1' 落地/s ∈ [1.5, 3.0]        K2' min(左右 spacing p50) ≥ 0.20 m   K3' flight ≤ 0.10
  K4' ds ≥ 0.20                  K5' ep_len ≥ 800 且 ret ≥ 1525
  K6' TB gait/osc_active_{L,R} ≈ 0.37±0.05  —— 不达标判"实现坏"，不判"训练坏"
  合格 episode = 同一集里 K1'-K4' 同时成立；窗口内需要 **≥2 个** 合格 episode（沿用 K1 传统）。
  分支（事前登记）：200K 时若 ep_len<300 ⇒ 判"A/介入过强"，退 A=0.15 重跑；
                    1.2M 时 K1'-K4' 全过而 K5' 不过 ⇒ 拖累在钱的结构 ⇒ 转 B2。

对照物（不是判据，只是参照，全部实测）：
  被动基线（零动作+模板）land/s 4.17 spacing 0.134 ds 0.031 flight 0.281 v 0.53
  ⑨ 最优碎步            land/s 12.07 spacing 0.010/0.073 ds 0.090 flight 0.270 v 0.49
  人体（9 clip 合并）    land/s 1.5-2.9 spacing 0.34-0.52 ds 0.23-0.46 flight ≤0.05 v 0.54-1.12
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

STEPS_PER_ROW = 3206.0        # 16K env 步 / eval ÷ 5 episode
DT = 0.015
BARS = dict(k1_lo=1.5, k1_hi=3.0, k2=0.20, k3=0.10, k4=0.20, k5_len=800, k5_ret=1525.0)
REF = {"被动基线": (4.17, 0.134, 0.031, 0.281, 0.53),
       "⑨最优碎步": (12.07, 0.073, 0.090, 0.270, 0.49),
       "人体p50": (2.13, 0.393, 0.270, 0.000, 0.66)}


def num(r, k, default=np.nan):
    v = r.get(k, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def load(path, since):
    rows = [json.loads(l) for l in open(path)][since:]
    out = []
    for i, r in enumerate(rows):
        ep = num(r, "ep_len")
        nL, nR = num(r, "ev_spacing_L_n"), num(r, "ev_spacing_R_n")
        if not np.isfinite(ep) or ep <= 0:
            continue
        land = (0.0 if not np.isfinite(nL) else nL) + (0.0 if not np.isfinite(nR) else nR)
        spL = num(r, "ev_spacing_L_p50")
        spR = num(r, "ev_spacing_R_p50")
        out.append(dict(row=since + i, step=int(i * STEPS_PER_ROW), ep_len=ep,
                        ret=num(r, "ret"), v=num(r, "speed"),
                        land_s=land / (ep * DT),
                        spacing_min=min(x for x in (spL, spR) if np.isfinite(x))
                        if np.isfinite(spL) or np.isfinite(spR) else np.nan,
                        ds=num(r, "double_support_frac"), flight=num(r, "flight_frac"),
                        single=num(r, "single_frac"),
                        rom_L=num(r, "ev_rom_L_mean"), rom_R=num(r, "ev_rom_R_mean"),
                        lift=max(num(r, "ev_lift_L_mean", 0), num(r, "ev_lift_R_mean", 0)),
                        walk_gate=num(r, "walk_gate"), flat_gate=num(r, "flat_gate"),
                        slip=max(num(r, "ev_slip_L_mean", 0), num(r, "ev_slip_R_mean", 0))))
    return out


def find_tb(root):
    """只认带 ⑪ 新键（osc_active_L）的事件文件，避免误读 ⑦/⑩' 的旧日志。"""
    cands = []
    for f in glob.glob(os.path.join(root, "**", "events.out.tfevents*"), recursive=True):
        if os.path.getsize(f) < 1000:
            continue
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(f, size_guidance={"scalars": 1})
            ea.Reload()
            tags = ea.Tags()["scalars"]
        except Exception as e:
            print(f"  [tb] 跳过 {f}: {e}")
            continue
        if "gait/osc_active_L" in tags:
            cands.append((os.path.getmtime(f), f, ea))
    if not cands:
        return None, None, None
    cands.sort()
    return cands[-1][1], cands[-1][2], [c[1] for c in cands]


def window(rows, upto_step):
    return [r for r in rows if r["step"] <= upto_step]


def report(rows, upto, label):
    sub = window(rows, upto)
    B = BARS
    print(f"\n### {label}（≤{upto/1e6:.1f}M env 步，{len(sub)} 集）")
    if not sub:
        print("  还没有这个区间的数据")
        return None
    med = lambda k: float(np.nanmedian([r[k] for r in sub])) if sub else np.nan
    q = [r for r in sub if B["k1_lo"] <= r["land_s"] <= B["k1_hi"]
         and (np.isfinite(r["spacing_min"]) and r["spacing_min"] >= B["k2"])
         and (np.isfinite(r["flight"]) and r["flight"] <= B["k3"])
         and (np.isfinite(r["ds"]) and r["ds"] >= B["k4"])]
    print(f"  中位：land/s={med('land_s'):5.2f} spacing_min={med('spacing_min'):6.3f} "
          f"ds={med('ds'):5.3f} flight={med('flight'):5.3f} single={med('single'):5.3f} "
          f"ep_len={med('ep_len'):6.0f} ret={med('ret'):7.1f} v={med('v'):5.2f} "
          f"lift={med('lift'):5.3f} slip={med('slip'):5.3f} rom={med('rom_L'):5.1f}/{med('rom_R'):5.1f}")
    print(f"  门控实况：walk_gate={med('walk_gate'):.3f} flat_gate={med('flat_gate'):.3f} "
          f"（人体 single 0.72 / ds 0.27 ⇒ 若仍为 1.0 说明门没关上）")
    verdict = {}
    verdict["K1' 落地/s∈[1.5,3.0]"] = (B["k1_lo"] <= med("land_s") <= B["k1_hi"], med("land_s"))
    verdict["K2' min步长≥0.20"] = (med("spacing_min") >= B["k2"], med("spacing_min"))
    verdict["K3' flight≤0.10"] = (med("flight") <= B["k3"], med("flight"))
    verdict["K4' ds≥0.20"] = (med("ds") >= B["k4"], med("ds"))
    verdict["K5' ep_len≥800"] = (med("ep_len") >= B["k5_len"], med("ep_len"))
    verdict["K5' ret≥1525"] = (med("ret") >= B["k5_ret"], med("ret"))
    verdict[f"合格episode≥2（K1'-K4'同集同时成立）"] = (len(q) >= 2, len(q))
    for k, (ok, val) in verdict.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}  实测={val:.3f}" if isinstance(val, float)
              else f"  {'PASS' if ok else 'FAIL'}  {k}  实测={val}")
    if med("ep_len") < 300:
        print("  ⚠ 分支触发（事前登记）：ep_len<300 ⇒ 判 A/介入过强 ⇒ 退 A=0.15 重跑，不加新机制")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/home/guo/workspace/humanoid_walk/eval_logs/eval_gait.jsonl")
    ap.add_argument("--since", type=int, default=2460, help="⑪ 开跑时的 jsonl 行数（新数据从这行之后）")
    ap.add_argument("--tb-root", default="/home/guo/workspace/humanoid_walk/sac_humanoid_walk")
    ap.add_argument("--upto", type=float, default=1.2e6, help="判到多少 env 步为止")
    a = ap.parse_args()
    rows = load(a.jsonl, a.since)
    print(f"jsonl={a.jsonl}  总行数={len([1 for _ in open(a.jsonl)])}  --since={a.since} ⇒ 取到 {len(rows)} 集")
    if not rows:
        print("新数据 0 集：⑪ 还没产出 eval，或 --since 给错了（开跑前的行数才是 since）。")
    else:
        last = max(r["step"] for r in rows)
        print(f"最新一集 ≈ {last/1e6:.2f}M env 步")
    for upto, lab in ((0.2e6, "200K 首判"), (0.6e6, "600K 复判"), (1.2e6, "1.2M 终判")):
        if upto <= a.upto * 1.05:
            report(rows, upto, lab)
    print("\n=== K6' 实现自检（TB）===")
    path, ea, allc = find_tb(a.tb_root)
    if ea is None:
        print(f"  在 {a.tb_root} 下没找到含 gait/osc_active_L 的事件文件 ⇒ "
              f"要么 ⑪ 还没跑，要么 TB 键没写进去（**这本身就是 FAIL：模板可能在空转**）")
    else:
        for t in ("gait/osc_active_L", "gait/osc_active_R", "gait/osc_k_now"):
            if t in ea.Tags()["scalars"]:
                s = ea.Scalars(t)
                v = np.array([x.value for x in s])
                print(f"  {t}: 均值={v.mean():.3f}（应≈0.367）末段={v[-50:].mean():.3f} 采样={len(v)}")
        print(f"  取用的事件文件：{path}")
    print("\n=== 参照（实测，非判据）===")
    for k, (ls, sp, ds, fl, v) in REF.items():
        print(f"  {k:10s} land/s={ls:6.2f} spacing={sp:5.3f} ds={ds:5.3f} flight={fl:5.3f} v={v:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
