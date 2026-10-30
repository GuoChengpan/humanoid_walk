"""批次⑮ 的判决（阈值全部在观察之前写在 CONTEXT 与本文件头，事后不挪）。

⑮ = N1 reach 带 [−0.20,0.08]→[0.00,0.20] ／ N2 stride_bonus 摘 walk_gate + TOP 0.32→0.20
     ／ N3 height+survival+碎步税 的 `v>0.3` 二值门 → 同一把 `_speed_ramp=clip((v−0.10)/0.20)`。

**事前登记的判据**（2026-10-14 21:4x 写，早于任何 ⑮ 数据）：
  C1 末 270 行 min(左,右) 落地间距中位 ≥ **0.07**        （⑬⑭ 停训时 0.040）
  C2 末 270 行 ev_reach_R_mean ≥ **0.03 m**              （⑬⑭ 0.004）
  C3 gait/stride_bonus_gated ≥ **0.05 分/帧** 且 speed_ramp_mean < **1.0**
  C4 不退步：swap_rate ≥0.18、land_per_s ≤6.3、ep_len 中位 ≥1400
  分支（也事前写）：C1 与 C2 **同时**不过而 C4 过 ⇒ 幅度既不响应价格也不响应 reach 带
                    ⇒ ⑯ 动膝（膝盖目前无任何奖励项）或动模板振幅，不再动价格。
  红线：起点塌方判据 `ret < −600 且 exploration_std ∉ [0.15,0.30]` ⇒ 记在 LR 冲击头上
        （暖机把 LR 重置到 2e-4），不判价格；⑮ 在同一条轨迹上的 A/B 只有 −2.7 分/集，
        所以若真出现塌方，第一嫌疑人是接线 ⇒ 先重跑 verify_batch15_ab.py 排除。

血统：⑮ 的行带 `b15`/`speed_ramp_mean` 键；缺这两个键的行**一律不看**（找不到就报 0，不回退到全文件）。
步数标定：不假设 3206 步/行，用 TB `time/total_timesteps` 末步 ÷ 当前行数 实测反推并打印。
对照：`--ref <⑬⑭ 归档 jsonl>` 时做**同窗行号**对照，并明说它不是干净的价格 A/B（起点权重不同）。
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
MARK = ("b15", "speed_ramp_mean")
BARS = dict(c1=0.07, c2=0.03, c3=0.05, c3b=1.0, c4_swap=0.18, c4_land=6.3, c4_len=1400)
TB_KEYS = ["gait/stride_bonus_gated", "gait/reach_reward_gated", "gait/phase_reward_gated",
           "gait/height_reward_gated", "gait/survival_bonus_gated", "gait/position_swap_gated",
           "gait/forward_gauss_gated", "gait/speed_ramp", "gait/walk_gate", "gait/swap_gate",
           "gait/tiny_step_frame_frac", "gait/speed", "train/exploration_std",
           "time/total_timesteps", "rollout/ep_rew_mean"]


def rows_of(path):
    try:
        all_rows = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        return []
    return [r for r in all_rows if all(k in r for k in MARK)]


def g(w, k):
    v = [float(r.get(k, np.nan)) for r in w]
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def minsp(w):
    a, b = g(w, "ev_spacing_L_p50"), g(w, "ev_spacing_R_p50")
    return min(x for x in (a, b) if np.isfinite(x)) if (np.isfinite(a) or np.isfinite(b)) else float("nan")


def tb_read(path=None):
    if path is None:
        cands = []
        for f in glob.glob(os.path.join(ROOT, "sac_humanoid_walk", "**", "events.out.tfevents*"), recursive=True):
            if os.path.getsize(f) < 5000:
                continue
            with open(f, "rb") as fh:
                head = fh.read(2_000_000)
            if b"gait/speed_ramp" in head:
                cands.append((os.path.getmtime(f), f))
        if not cands:
            return {}, None
        path = max(cands)[1]
    from tensorboard.backend.event_processing import event_file_loader
    out = collections.defaultdict(lambda: collections.deque(maxlen=6000))
    for ev in event_file_loader.EventFileLoader(path).Load():
        for v in (ev.summary.value or []):
            if v.tag in TB_KEYS:
                val = (v.tensor.float_val[0] if v.tensor.float_val else
                       v.tensor.double_val[0] if v.tensor.double_val else v.value)
                out[v.tag].append((ev.step, float(val)))
    return {k: np.array(v) for k, v in out.items()}, path


def money_ratio(M, tail=120):
    """同一把尺（TB 随机 rollout）、同一个分桶定义的 原地:前进。分桶与 verify_batch15_ab 一致。"""
    adv = sum(M[t][-tail:, 1].mean() for t in ("gait/stride_bonus_gated", "gait/forward_gauss_gated",
                                               "gait/position_swap_gated") if t in M and len(M[t]) >= tail)
    ip = sum(M[t][-tail:, 1].mean() for t in ("gait/phase_reward_gated", "gait/height_reward_gated",
                                              "gait/survival_bonus_gated", "gait/reach_reward_gated")
             if t in M and len(M[t]) >= tail)
    return ip, adv


def last_step(M):
    s = [arr[:, 0].max() for arr in M.values() if len(arr)]
    return float(max(s)) if s else float("nan")


def line(w, lab):
    print(f"  {lab:16s} ep_len={g(w,'ep_len'):6.0f} ret={g(w,'ret'):8.1f} ret/帧={g(w,'ret')/max(g(w,'ep_len'),1):+.4f} "
          f"min_sp={minsp(w):+.3f} swap={g(w,'swap_rate'):.3f} land/s={g(w,'land_per_s'):5.2f} "
          f"ds={g(w,'double_support_frac'):.3f} fl={g(w,'flight_frac'):.3f} "
          f"reach_R={g(w,'ev_reach_R_mean'):+.3f} tax占空={g(w,'tiny_step_frame_frac'):.3f} ramp={g(w,'speed_ramp_mean'):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(ROOT, "best_model/tmp_model_data/SAC_19/19_3_实现原地踏步交替/eval_gait.jsonl"),
                    help="⑬⑭ 归档 jsonl，做同窗行号对照（传空串则跳过）")
    ap.add_argument("--n", type=int, default=90, help="起点判读窗的行数")
    ap.add_argument("--ref-tb", default=None, help="⑬⑭ 的 TB 事件文件；缺省自动取同目录下最新的那个非本轮文件")
    a = ap.parse_args()
    rows = rows_of(JSONL)
    print(f"⑮ 行数 = {len(rows)}（按 {MARK} 标记筛选；文件总行数 "
          f"{sum(1 for _ in open(JSONL)) if os.path.exists(JSONL) else 0}）")
    if not rows:
        print("找不到带 b15 标记的行 ⇒ ⑮ 还没产出 eval。**不回退去分析其它行**（那是别的血统）。")
        return 1
    M, tpath = tb_read()
    step = last_step(M)
    spr = step / len(rows) if np.isfinite(step) else np.nan
    print(f"TB={os.path.relpath(tpath, ROOT) if tpath else '（没找到含 gait/speed_ramp 的事件文件）'}")
    print(f"步数标定：TB 末步 {step/1e6:.3f}M ÷ {len(rows)} 行 = **{spr:.0f} env 步/行**（实测反推，不假设 3206）")

    print("\n=== 起点判读（前 %d 行）===" % a.n)
    for k in range(0, min(len(rows), a.n), 30):
        line(rows[k:k + 30], f"行{k}-{k+30}")
    if a.ref and os.path.exists(a.ref):
        ref = [json.loads(l) for l in open(a.ref)]
        print(f"\n  同窗对照（⑬⑭ 归档 {len(ref)} 行；⚠ 起点权重不同，这不是干净的价格 A/B）")
        for k in range(0, min(len(rows), a.n, len(ref)), 30):
            w, rw = rows[k:k + 30], ref[k:k + 30]
            if not w or not rw:
                continue
            print(f"  行{k}-{k+30}: ret/帧 ⑬⑭={g(rw,'ret')/max(g(rw,'ep_len'),1):+.4f} ⑮={g(w,'ret')/max(g(w,'ep_len'),1):+.4f}"
                  f" ｜reach_R {g(rw,'ev_reach_R_mean'):+.3f}→{g(w,'ev_reach_R_mean'):+.3f}"
                  f" ｜land/s {g(rw,'land_per_s'):.2f}→{g(w,'land_per_s'):.2f}"
                  f" ｜ds {g(rw,'double_support_frac'):.3f}→{g(w,'double_support_frac'):.3f}")
    red = (g(rows[-a.n:], "ret") < -600)
    std = float(M["train/exploration_std"][-120:, 1].mean()) if "train/exploration_std" in M else np.nan
    print(f"\n  红线（eval<−600 且 std∉[0.15,0.30] ⇒ 判 LR 冲击）：eval_ret={g(rows[-a.n:],'ret'):.1f} "
          f"std={std:.3f} ⇒ {'**触发，按规则记在 LR 头上，不动价格**' if red and not (0.15 <= std <= 0.30) else '未触发'}")

    print("\n=== C3（TB 仪表，与 det 不同尺）===")
    for t in ("gait/stride_bonus_gated", "gait/reach_reward_gated", "gait/speed_ramp",
              "gait/phase_reward_gated", "gait/height_reward_gated", "gait/forward_gauss_gated",
              "gait/position_swap_gated", "gait/tiny_step_frame_frac", "gait/walk_gate"):
        if t in M:
            v = M[t][:, 1]
            print(f"  {t.replace('gait/',''):24s} 末段均值={v[-120:].mean():+.4f} 全程={v.mean():+.4f} n={len(v)}")
    ip, adv = money_ratio(M)
    print(f"  ⇒ ⑮ 原地:前进（TB 尺，末 120 点）= {ip:.3f}/{adv:.3f} = **{ip/max(adv,1e-9):.1f}:1**")
    ref_tb = a.ref_tb
    if ref_tb is None:
        cands = [f for f in glob.glob(os.path.join(ROOT, "sac_humanoid_walk", "**", "events.out.tfevents*"),
                                      recursive=True)
                 if os.path.getsize(f) > 100000 and os.path.realpath(f) != os.path.realpath(tpath or "")]
        cands = [f for f in cands if (lambda p: (b"gait/swap_gate" in open(p, "rb").read(4_000_000)))(f)]
        ref_tb = max(cands, key=os.path.getmtime) if cands else ""
    if ref_tb and os.path.exists(ref_tb):
        try:
            Mr, _ = tb_read(ref_tb)
            ip2, adv2 = money_ratio(Mr)
            print(f"  同定义同尺的 ⑬⑭ 基线（{os.path.basename(ref_tb)[:28]}…）= {ip2:.3f}/{adv2:.3f} = "
                  f"**{ip2/max(adv2,1e-9):.1f}:1** ⇒ 比值变化 {ip/max(adv,1e-9) - ip2/max(adv2,1e-9):+.1f}"
                  f"（负=钱更往走路那侧挪）")
        except Exception as e:
            print(f"  ⑬⑭ 基线读取失败：{e}")
    print(f"  （参考：A/B 在同一条 det 轨迹上量到 ⑮ 开跑时 10.6:1、⑬⑭ 停训时 16.8:1；"
          f"TB 尺含摔倒的 rollout，绝对值本就更高，只能同尺互比）")

    print("\n=== C1/C2/C4 正式判决（需要末 270 行，现有 %d 行 ⇒ 覆盖 %.0f%%）===" %
          (len(rows), 100 * min(len(rows), 270) / 270))
    w = rows[-270:]
    n1, n2, n4 = minsp(w), g(w, "ev_reach_R_mean"), (g(w, "swap_rate"), g(w, "land_per_s"), g(w, "ep_len"))
    s3 = M["gait/stride_bonus_gated"][-120:, 1].mean() if "gait/stride_bonus_gated" in M else np.nan
    r3 = g(w, "speed_ramp_mean")
    V = [("C1 min_spacing ≥0.07", n1 >= BARS["c1"], n1),
         ("C2 reach_R ≥0.03", n2 >= BARS["c2"], n2),
         ("C3a stride_bonus_gated ≥0.05/帧", s3 >= BARS["c3"], s3),
         ("C3b speed_ramp_mean <1.0", r3 < BARS["c3b"], r3),
         ("C4a swap ≥0.18", n4[0] >= BARS["c4_swap"], n4[0]),
         ("C4b land/s ≤6.3", n4[1] <= BARS["c4_land"], n4[1]),
         ("C4c ep_len ≥1400", n4[2] >= BARS["c4_len"], n4[2])]
    for lab, ok, val in V:
        print(f"  {'PASS' if ok else 'FAIL'}  {lab:32s} 实测={val:.4f}")
    if len(rows) < 270:
        print("  （样本不足 270 行 ⇒ 上面只是预告，不作数；到 row 280 ≈ +0.9M 步再判）")
    if all(len(rows) >= 270 and lab.startswith(("C1", "C2")) and not ok for lab, ok, _ in V[:2]) and \
       all(ok for lab, ok, _ in V[4:]):
        print("  ⇒ 事前登记的分支触发：**价格与 reach 带都不动幅度** ⇒ ⑯ 动膝/动模板，别再动价格。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
