"""⑮ 的三个改价项，在**实测观测量**上做前后对比（纯算术，可和训练并存）。

它只回答三件事，每件都可核查、都不预测训练结果：
  N1 reach 带 [−0.20, 0.08] → [0.00, 0.20]：原地钱少发多少 / 每厘米的梯度变成多少
  N2 stride_bonus 去掉 ×walk_gate、STRIDE_TOP 0.32→0.20：在实测 min 上仍是 0（要如实说出它是死杠杆），
     以及 min 到达 0.10/0.15/0.20 时各值多少钱（这才是"付钱阶梯"的高度表）
  N3 height/survival/碎步税 的 v>0.3 二值门 → clip(v/0.3)：**断点跳变**必须为负

自检（读表之前先跑）：用 jsonl(det 中位) × 实测 land/s × 实测 swap_gate 反推 reach 的每帧发钱量，
与 TB(随机 rollout) 的 gait/reach_reward_gated 同量级才算接线对。两者本就不同尺（噪声+摔段，见
记忆 telemetry #5），所以这里只要求 |比| ∈ [0.6, 1.8]，超界 ⇒ 本脚本不可信，停止读数。
"""
import ast
import collections
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = "/home/guo/workspace/humanoid_walk"
SRC = os.path.join(ROOT, "train_humanoid_sac.py")
JSONL = os.path.join(ROOT, "eval_logs", "eval_gait.jsonl")
DT, TAIL = 0.015, 60


def consts():
    txt = open(SRC, encoding="utf-8").read()
    tree = ast.parse(txt)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WalkRewardWrapper")
    C = {}
    for node in ast.walk(cls.body[next(i for i, n in enumerate(cls.body)
                                       if isinstance(n, ast.FunctionDef) and n.name == "__init__")]):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                try:
                    v = node.value
                    C[t.attr] = -float(v.operand.value) if (isinstance(v, ast.UnaryOp)
                                                            and isinstance(v.operand, ast.Constant)) else float(v.value)
                except Exception:
                    pass
    for nm in ("STRIDE_FLOOR", "STRIDE_TOP", "STRIDE_WEIGHT", "SWING_DRIVE_WEIGHT"):
        m = re.search(r"\b" + nm + r"\s*=\s*([0-9.]+)", txt)
        if m:
            C[nm] = float(m.group(1))
    m = re.search(r"STRIDE_CAD_REF\s*=\s*([0-9.]+)", txt)
    C["STRIDE_CAD_REF"] = float(m.group(1)) if m else 2.5
    return C


def tb_latest(tags):
    best = None
    for f in glob.glob(os.path.join(ROOT, "sac_humanoid_walk", "**", "events.out.tfevents*"), recursive=True):
        if os.path.getsize(f) > 5000 and os.path.getmtime(f) > (os.stat(best).st_mtime if best else 0):
            best = f
    from tensorboard.backend.event_processing import event_file_loader
    out = collections.defaultdict(lambda: collections.deque(maxlen=3000))
    for ev in event_file_loader.EventFileLoader(best).Load():
        for v in (ev.summary.value or []):
            if v.tag in tags:
                x = (v.tensor.float_val[0] if v.tensor.float_val else
                     v.tensor.double_val[0] if v.tensor.double_val else v.value)
                out[v.tag].append(float(x))
    return {k: float(np.mean(v)) for k, v in out.items()}, best


C = consts()
rows = [json.loads(l) for l in open(JSONL)]
rows = [r for r in rows if "swap_rate" in r][-TAIL:]
g = lambda k: float(np.nanmedian([float(r.get(k, np.nan)) for r in rows]))
M, tf = tb_latest({"gait/reach_reward_gated", "gait/swap_gate", "gait/walk_gate", "gait/forward_gate"})

REACH_W = C["SWING_DRIVE_WEIGHT"]
min_sp = min(g("ev_spacing_L_p50"), g("ev_spacing_R_p50"))
reach = {"L": g("ev_reach_L_mean"), "R": g("ev_reach_R_mean")}
land_f = g("land_per_s") * DT
sc = lambda r, f, t: REACH_W * np.clip((r - f) / (t - f), 0, 1)
score_now = {k: sc(v, C["reach_floor"], C["reach_top"]) for k, v in reach.items()}
score_n1 = {k: sc(v, 0.0, 0.20) for k, v in reach.items()}
est = lambda s: sum(s.values()) * land_f / 2.0 * M["gait/swap_gate"]

print(f"常数: reach[{C['reach_floor']},{C['reach_top']}] w={REACH_W} stride[{C['STRIDE_FLOOR']},{C['STRIDE_TOP']}] w={C['STRIDE_WEIGHT']} cad={C['STRIDE_CAD_REF']}")
print(f"实测(det 末{len(rows)}集): reach L={reach['L']*100:.1f}cm R={reach['R']*100:.1f}cm  "
      f"min落点间距={min_sp:.3f}  落地/s={g('land_per_s'):.2f}  ｜TB: swap_gate={M['gait/swap_gate']:.3f} "
      f"walk_gate={M['gait/walk_gate']:.3f} forward_gate={M['gait/forward_gate']:.3f}")
ratio = est(score_now) / M["gait/reach_reward_gated"]
print(f"自检 反推 reach 发钱={est(score_now):.4f}/帧 vs TB={M['gait/reach_reward_gated']:.4f}/帧 比={ratio:.2f} "
      f"⇒ {'接线可接受(两尺不同，允许 0.6-1.8)' if 0.6 <= ratio <= 1.8 else '**不可信，停读**'}")
if not (0.6 <= ratio <= 1.8):
    sys.exit(1)

print("\n=== N1 reach 带重挂 ===")
print(f"  {'腿':3s} {'实测reach':>10} {'现行分/落地':>12} {'N1分/落地':>10} {'每cm现行':>10} {'每cm N1':>9}")
for k in ("L", "R"):
    print(f"  {k:3s} {reach[k]*100:9.1f}cm {score_now[k]:12.3f} {score_n1[k]:10.3f} "
          f"{REACH_W*100/(C['reach_top']-C['reach_floor']):10.3f} {REACH_W*100/0.20:9.3f}")
print(f"  ⇒ 这一项从原地钱里拿走 {est(score_now)-est(score_n1):+.4f}/帧（TB 口径换算后 ≈ "
      f"{(est(score_now)-est(score_n1))/ratio:+.4f}），并将每厘米梯度从 {REACH_W*100/(C['reach_top']-C['reach_floor']):.3f} "
      f"提到 {REACH_W*100/0.20:.3f} 分/落地/cm")
print(f"  ⚠ 现行带下 R 的 reach=0 仍值 {sc(0.0, C['reach_floor'], C['reach_top']):.2f}/2.0 = "
      f"{100*sc(0.0, C['reach_floor'], C['reach_top'])/REACH_W:.0f}% 满分 ⇒ 这是补贴不是定价")

print("\n=== N2 stride_bonus 付钱阶梯（含/不含 walk_gate 两档）===")
wg = M["gait/walk_gate"]
disc = min(1.0, C["STRIDE_CAD_REF"] / max(g("land_per_s"), 1e-9))
for lab, top in (("现 TOP 0.32", C["STRIDE_TOP"]), ("N2 TOP 0.20", 0.20)):
    print(f"  {lab}（floor {C['STRIDE_FLOOR']}）:")
    for lo in (min_sp, 0.09, 0.10, 0.15, 0.20):
        v = np.clip((lo - C["STRIDE_FLOOR"]) / (top - C["STRIDE_FLOOR"]), 0, 1) * C["STRIDE_WEIGHT"] * disc
        print(f"    min={lo:.3f} ⇒ 每落地 {v:6.2f} 分 = 每帧 {v*land_f:6.4f}（乘 walk_gate {wg:.3f} 后 {v*land_f*wg:6.4f}）")
print(f"  ⇒ 实测 min={min_sp:.3f} < floor={C['STRIDE_FLOOR']} ⇒ **N2 在今天的策略上是死杠杆**，"
      f"它只在 min≥0.10 后付钱；这是设计意图（不给盆地发钱），不是梯度。")

print("\n=== N3 三道 v>0.3 二值门 → clip(v/0.3) 的断点跳变 ===")
for nm, v1, v2 in (("跨越 0.3 向下", 0.301, 0.299), ("当前均速附近 ±0.1", 0.527, 0.327)):
    for lab, gate in (("二值(现行)", lambda v: 1.0 if v > 0.3 else 0.0),
                      ("线性(⑮)", lambda v: np.clip(v / 0.3, 0, 1))):
        d = gate(v2) - gate(v1)
        print(f"  {lab}: {nm} gate(v={v1:.3f})={gate(v1):.3f} → gate(v={v2:.3f})={gate(v2):.3f} 跳变={d:+.3f}"
              f"{'  ⇒ 断点已抹平' if abs(d) < 0.01 else ''}")
print("  注：三项必须同批改（height/survival/税共用同一个 0.3，只改其中两个会造出『跌破 0.3 免稅』的坑，"
      "已实测 +0.119/帧）。")
