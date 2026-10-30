"""批次⑯ 的判决（D1-D4 在观察之前写在 verify_batch16_T2.py 与本文件头，事后不挪）。

⑯ = T2 驱动定律：髋前摆由**时钟窗口**改成**该脚离地起、真接触止**，窗口内恒值 −A（原 −A·sin(πu)），
     `osc_wmax=28` 帧硬上限，双腾空时谁都不驱动（无左右偏好 ⇒ 镜像等变），`osc_A=0.15` 不变。
     ⑮ 的价格（N1 reach 带 [0,0.20] / N2 stride 摘 walk_gate+TOP0.20 / N3 三扇 v>0.3 门→同一把斜坡）
     一行未动 —— 已用真·⑮ 归档源码 `19_4/train_humanoid_sac.py` 逐帧对奖励核过：max|Δr|=0.00e+00。

事前登记的判据（2026-10-15 21:0x 写，早于任何 ⑯ 数据）：
  D1 触地被驱动覆盖率 osc_land_cover ≥ **0.80**        （⑮ 同尺 T1 = 0.086；双腾空使上限 <1，故同时报上限）
  D2 落地大腿前倾 ev_hip_at_land_{L,R}_p50 ≥ **25°**   （人体 28.8，p10 25.4；⑮ det 为 18.2 / 10.4）
  D3 ev_reach_R_mean ≥ **0.10 m**                      （人体这些片段 0.233 属快步态 ⇒ 取保守中间值）
  D4 不退步：det ep_len ≥ **1200**、swap_rate ≥ 0.18、land_per_s ≤ 6.3
  红线：ret < −600 且 exploration_std ∉ [0.15,0.30] ⇒ 记在 LR 冲击（暖机把 LR 重置回 2e-4），不动价格
分支（也事前写）：
  · D2 只有一半（一条腿到 25°、另一条不到）⇒ 问题不在驱动时机，在左右仲裁/单侧偏好 ⇒ ⑰ 开镜像重放
  · D2 两腿都过而 D3 不过 ⇒ 到位了但小腿往后甩 ⇒ ⑰ 动膝（屈膝现在无任何奖励项；实测人体触地 25°，
    我们两腿各 40/41° ⇒ 是"别弯太多"，不是"多伸"）
  · flight 不随步长下降 ⇒ ⑰ 先动腾空，别动价格
尺子规矩：det（eval jsonl 每集中位）与 TB（随机 rollout 末段均值）是两把尺，只在同尺内比；
  ds/flight/stance 是 200 帧滚动窗口在集末的取值，不是整集占比。
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/guo/workspace/humanoid_walk"
JSONL = os.path.join(ROOT, "eval_logs", "eval_gait.jsonl")
MARK = ("b16",)
BARS = dict(d1=0.80, d2=25.0, d3=0.10, d4_len=1200, d4_swap=0.18, d4_land=6.3)
WIN = 270          # 与 ⑮ 的 C1 同一窗口长度，跨批次可比
TAGS = {"gait/stride_bonus_gated", "gait/reach_reward_gated", "gait/phase_reward_gated",
        "gait/height_reward_gated", "gait/forward_gauss_gated", "gait/position_swap_gated",
        "gait/walk_gate", "gait/swap_gate", "gait/speed_ramp", "gait/tiny_step_frame_frac",
        "gait/flight_frac", "gait/double_support_frac", "gait/speed", "gait/osc_active_L",
        "gait/osc_active_R", "gait/osc_land_cover", "train/exploration_std",
        "rollout/ep_len_mean", "rollout/ep_rew_mean", "train/learning_rate"}


def rows_of(path):
    if not os.path.exists(path):
        return []
    return [r for r in map(json.loads, open(path)) if all(k in r for k in MARK)]


def g(w, k):
    v = [float(r.get(k, np.nan)) for r in w]
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def tb():
    cands = []
    for f in glob.glob(os.path.join(ROOT, "sac_humanoid_walk", "**", "events.out.tfevents*"), recursive=True):
        _ = f
        if os.path.getsize(f) < 5000:
            continue
        with open(f, "rb") as fh:
            # 认文件用 ⑮ 就有的 tag —— `osc_land_cover` 只进了 jsonl、没进 TB（那笔欠账等停训补），
            # 用它当定位键会让整个 TB 段静默跳过（第一版就栽在这，std 读成 nan）
            if b"gait/speed_ramp" in fh.read(2_000_000):
                cands.append((os.path.getmtime(f), f))
    if not cands:
        return {}, None, 0
    path = max(cands)[1]
    from tensorboard.backend.event_processing import event_file_loader
    out = collections.defaultdict(list)
    last = 0
    for ev in event_file_loader.EventFileLoader(path).Load():
        for v in (ev.summary.value or []):
            if v.tag in TAGS:
                x = (v.tensor.float_val[0] if v.tensor.float_val else
                     (v.tensor.double_val[0] if v.tensor.double_val else v.value))
                out[v.tag].append(float(x))
        last = max(last, ev.step)
    return {k: np.array(v) for k, v in out.items()}, path, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="起点/恢复判读窗（行）")
    a = ap.parse_args()
    rows = rows_of(JSONL)
    print(f"⑮ 血统标记 {MARK} 命中 {len(rows)} 行（文件总行数 {sum(1 for _ in open(JSONL)) if os.path.exists(JSONL) else 0}）")
    if not rows:
        print("找不到带 b16 的行 ⇒ ⑯ 还没产出 eval。**不回退去看别的行**（那是别的血统）。")
        return 1
    M, tf, last = tb()
    spr = last / len(rows) if last else float("nan")
    print(f"TB={os.path.relpath(tf, ROOT) if tf else '缺'}  末步={last/1e6:.2f}M ⇒ 实测 {spr:.0f} env 步/行")
    if tf is None:
        print("⚠ TB 里没有 gait/osc_land_cover ⇒ ⑯ 的仪表没写进 TB（欠账，不是故障；D1 仍可从 jsonl 读）")

    print("\n=== 恢复轨迹（每窗 ~%d 行）===" % max(len(rows) // 6, 25))
    B = max(len(rows) // 6, 25)
    print(f"{'窗':12s}{'ep_len':>7}{'ret':>8}{'min间距':>9}{'髋L°':>7}{'髋R°':>7}{'reachL':>8}{'reachR':>8}"
          f"{'swap':>7}{'land/s':>7}{'cover':>7}{'dair':>6}{'cap':>5}")
    for i in range(0, len(rows), B):
        w = rows[i:i + B]
        if len(w) < 12:
            continue
        hl = g(w, "ev_hip_at_land_L_p50"); hr = g(w, "ev_hip_at_land_R_p50")
        print(f"行{i}-{i+len(w):<5d}{g(w,'ep_len'):7.0f}{g(w,'ret'):8.1f}"
              f"{min(g(w,'ev_spacing_L_p50'),g(w,'ev_spacing_R_p50')):9.3f}{hl:7.1f}{hr:7.1f}"
              f"{g(w,'ev_reach_L_mean'):8.3f}{g(w,'ev_reach_R_mean'):8.3f}{g(w,'swap_rate'):7.3f}"
              f"{g(w,'land_per_s'):7.2f}{g(w,'osc_land_cover'):7.3f}{g(w,'osc_double_air_frac'):6.3f}{g(w,'osc_cap_cut'):5.0f}")

    W = rows[-WIN:]
    d1, d2l, d2r = g(W, "osc_land_cover"), g(W, "ev_hip_at_land_L_p50"), g(W, "ev_hip_at_land_R_p50")
    d3, dl = g(W, "ev_reach_R_mean"), g(W, "ev_reach_L_mean")
    swap, land, ep = g(W, "swap_rate"), g(W, "land_per_s"), g(W, "ep_len")
    dair, cap = g(W, "osc_double_air_frac"), g(W, "osc_cap_cut")
    ceil = 1.0 - dair
    print(f"\n=== 判决（末 {len(W)} 行；⑮ 同尺基线：cover 0.086 / 髋 18.2,10.4 / reach_R 0.002）===")
    V = [("D1 cover ≥0.80", d1 >= BARS["d1"], d1),
         ("D1b 结构上限 1−double_air", None, ceil),
         ("D2 落地大腿 L ≥25°", d2l >= BARS["d2"], d2l),
         ("D2 落地大腿 R ≥25°", d2r >= BARS["d2"], d2r),
         ("D3 reach_R ≥0.10 m", d3 >= BARS["d3"], d3),
         ("D4a det ep_len ≥1200", ep >= BARS["d4_len"], ep),
         ("D4b swap ≥0.18", swap >= BARS["d4_swap"], swap),
         ("D4c land/s ≤6.3", land <= BARS["d4_land"], land)]
    for lab, ok, val in V:
        mark = "—" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {mark:4s} {lab:26s} 实测={val:8.3f}")
    print(f"  （cap 截断 {cap:.0f} 次/集 ⇒ 有腿悬空超过 osc_wmax=28 帧；D1 的可达上限≈{ceil:.2f}，"
          f"cover 离它还有 {max(ceil-d1,0):.2f} ⇒ 差额才是『驱动真的没盖住』的部分）")
    if d2l >= 25 and d2r < 25:
        print("  ⇒ 事前登记的分支：**只有 L 到位** ⇒ 问题在左右仲裁/单侧偏好，不在驱动时机 ⇒ ⑰ 考虑镜像重放入 buffer")
    elif d2l >= 25 and d2r >= 25 and d3 < 0.10:
        print("  ⇒ 分支：髋到位但脚仍不往前 ⇒ ⑰ 动膝（人体触地屈膝 25°，⑯ 前实测两腿 40/41° ⇒ 是『别弯那么多』）")
    elif g(W, "flight_frac") > 0.15:
        print(f"  ⇒ 分支：flight 仍 {g(W,'flight_frac'):.3f}（人体 0.000）⇒ ⑰ 先治腾空，别动价格")

    if M:
        adv = sum(M[t][-150:].mean() for t in ("gait/stride_bonus_gated", "gait/forward_gauss_gated",
                                              "gait/position_swap_gated") if t in M)
        ip = sum(M[t][-150:].mean() for t in ("gait/phase_reward_gated", "gait/height_reward_gated",
                                             "gait/reach_reward_gated") if t in M)
        print(f"\n=== TB 尺（末150点，随机）===")
        print(f"  前进 {adv:.4f} ｜phase+height+reach {ip:.4f} ⇒ **{ip/max(adv,1e-9):.1f}:1**"
              f"（⑮ 停训时 27:1，⑯ 0.63M 时 16.8:1；A/B 同轨迹上 ⑮ 起点 10.6:1）")
        for t in sorted(M):
            print(f"   {t.replace('gait/','').replace('rollout/','').replace('train/',''):24s}{M[t][-150:].mean():9.4f}")
    std = float(M["train/exploration_std"][-150:].mean()) if "train/exploration_std" in M else float("nan")
    print(f"\n红线：ret={g(W,'ret'):.1f} std={std:.3f} ⇒ "
          f"{'**LR 冲击触发，按规则不动价格**' if (g(W,'ret') < -600 and not (0.15 <= std <= 0.30)) else '未触发'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
