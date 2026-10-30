"""批次⑰ 判决脚本（**在 ⑰ 产生任何数据之前写好**，判据不许事后改）。

用法：
  python3 debug_probes/judge_batch17.py                 # 读 eval_logs/eval_gait.jsonl
  python3 debug_probes/judge_batch17.py --rows 1300     # 只读某个行数做分段趋势
  python3 debug_probes/judge_batch17.py --path <jsonl>  # 用户换/归档了文件时指定

行定位：只取带 `b17` 标记的行。⚠ 已知陷阱：⑰ 的行**同时带 b16=1**（我在 b16 那行之前插入），
所以 judge_batch16.py 会把 ⑰ 的行当成 ⑯ —— 本脚本反过来不会错（b17 只在 ⑰ 写入）。
若同批续跑（⑰ 的延续）追加到同一文件，本脚本无法区分，需要归档或按行数切。

起点基线 = **要加载的那份权重**（⑮ best，num_timesteps=960,000）在 8 seeds det rollout 上的实测，
不是我先前误用的"⑮ 末态 250 集"。基线值：
  adv_L 0.1137  adv_R 0.0906  asym 0.203  place_operand 0.1743  abd_near 0.0771
  land_per_s 3.8025  flight 0.1175  ds 0.0100  ep_len 1499
判据（观测前钉死）：
  J0  200K 只读趋势：若 adv_L 与 adv_R **双双 < 0.05** 且 place_operand 未升 ⇒ 判付款没指对方向，停。
  J1  800K 首判 adv_L ≥ 0.13 且 adv_R ≥ 0.11；2M 终判 ≥ 0.14 / ≥ 0.12；任一侧 < 0.063 = 硬 FAIL。
  J2  对称 |adv_L−adv_R|/max ≤ 0.25
  J3  guard：land_per_s ≤ 5.0、flight ≤ 0.15、double_support_frac ≥ 0.02
  J4  姿势：abd_y_near_limit_frac ≤ 0.15
  J5  存活：ep_len ≥ 1200（不许用摔换 J1）
  J6  付款对象真的在被满足：place_operand_mean ≥ 0.24（@800K）/ ≥ 0.28（@2M，人形满额的 93%）
      —— 这条原本我写成"place 占毛收入 ≥0.35"，但起点就已 784/662>1，那条判据恒真是我的错，
         在任何 ⑰ 数据存在之前换成上面这条可被满足、也可被否证的量。
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="eval_logs/eval_gait.jsonl")
ap.add_argument("--rows", type=int, default=0, help="只看前 N 行（分段趋势用），0=全部")
A = ap.parse_args()

if not os.path.exists(A.path):
    sys.exit(f"找不到 {A.path}（用户可能已归档；用 --path 指定）")
allr = [json.loads(l) for l in open(A.path, encoding="utf-8") if l.strip()]
rows = [r for r in allr if r.get("b17")]
if A.rows:
    rows = rows[:A.rows]
print(f"{A.path}：全文件 {len(allr)} 行，其中 b17 行 {len(rows)} 行"
      f"（1 行 = 1 集 = 5 集/eval 块 ⇒ 每块 16K 步 ⇒ 已训约 {len(rows)/5*16000/1e6:.2f}M 步）")
if len(rows) < 6:
    sys.exit("b17 行不足 6 行（<0.2M 步），现在读什么都不是结论。等 200K（≈62 行）再来。")

START = dict(adv_L=0.1137, adv_R=0.0906, asym=0.203, place=0.1743, abd=0.0771,
             land=3.8025, flight=0.1175, ds=0.0100, ep=1499.0)


def med(key):
    v = [r.get(key, np.nan) for r in rows]
    v = [float(x) for x in v if x is not None and x == x]
    return float(np.median(v)) if v else float("nan")


def seg(key, nb=4):
    """分 nb 段的中位数（噪声地板：每段 ≥30 行才可信）。"""
    out = []
    for i in range(nb):
        s, e = int(len(rows) * i / nb), int(len(rows) * (i + 1) / nb)
        blk = rows[s:e]
        v = [float(r[key]) for r in blk if key in r and r[key] == r[key]]
        out.append(float(np.median(v)) if len(v) else np.nan)
    return out


aL, aR = med("ev_step_adv_L_mean"), med("ev_step_adv_R_mean")
asym = abs(aL - aR) / max(aL, aR, 1e-9)
pl, abd = med("place_operand_mean"), med("abd_y_near_limit_frac")
land, fl, ds, ep = med("land_per_s"), med("flight_frac"), med("double_support_frac"), med("ep_len")
prog = len(rows) / 5 * 16000
stage = "200K趋势" if prog < 400_000 else ("800K首判" if prog < 1_400_000 else "2M终判")
th1 = (0.13, 0.11) if prog < 1_400_000 else (0.14, 0.12)
th6 = 0.24 if prog < 1_400_000 else 0.28

print(f"\n=== ⑰ 判决（{len(rows)} 集 ≈ {prog/1e6:.2f}M 步；判定点={stage}；窗内中位）===")
print(f"{'量':26s} {'起点':>9s} {'当前':>9s} {'Δ':>9s}   判据")
def row(nm, k, cur, base, need):
    print(f"  {nm:24s} {base:9.4f} {cur:9.4f} {cur-base:+9.4f}   {need}")
row("J1 adv_L (m)", "ev_step_adv_L_mean", aL, START["adv_L"], f"≥{th1[0]:.2f} 且 ≥0.063")
row("J1 adv_R (m)", "ev_step_adv_R_mean", aR, START["adv_R"], f"≥{th1[1]:.2f} 且 ≥0.063")
row("J2 不对称度", "", asym, START["asym"], "≤0.25")
row("J6 place_operand/帧", "", pl, START["place"], f"≥{th6:.2f}")
row("J4 腰贴限位帧占比", "", abd, START["abd"], "≤0.15")
row("J3 land_per_s", "", land, START["land"], "≤5.0")
row("J3 flight_frac", "", fl, START["flight"], "≤0.15")
row("J3 double_support", "", ds, START["ds"], "≥0.02")
row("J5 ep_len", "", ep, START["ep"], "≥1200")

verdicts = []
if stage == "200K趋势":
    J0 = (aL < 0.05 and aR < 0.05 and pl <= START["place"])
    verdicts.append(("J0 付款方向没指对（该停）", J0))
else:
    verdicts.append(("J1 前行达标", aL >= th1[0] and aR >= th1[1]))
    verdicts.append(("J1 硬失败（跌破起点 0.7×）", aL < 0.063 or aR < 0.063))
verdicts.append(("J2 对称达标", asym <= 0.25))
verdicts.append(("J5 存活达标", ep >= 1200))
verdicts.append(("J4 姿势未退化", abd <= 0.15))
verdicts.append(("J6 新通道被满足", pl >= th6))
print("\n判定：")
for nm, hit in verdicts:
    print(f"  {'✔' if hit else '✘'} {nm}")

if len(rows) >= 120:
    print("\n分段趋势（每段 ≥30 行，早→晚）：")
    for nm, k in (("adv_L", "ev_step_adv_L_mean"), ("adv_R", "ev_step_adv_R_mean"),
                  ("place", "place_operand_mean"), ("land/s", "land_per_s"),
                  ("abd_limit", "abd_y_near_limit_frac"), ("ep_len", "ep_len")):
        print(f"  {nm:10s} " + "  ".join(f"{x:+.3f}" if x == x else "   —  " for x in seg(k)))
print("\n提醒：`swap_rate`/`ev_spacing_*` **不在判据里**——它们是被证明可以通胀的量"
      "（⑯：swap 0.33→0.97 而 adv ÷3.3）。")
