"""分析当前 run（SAC_18, 摆动驱动修复版）的训练趋势。

按 timesteps 分桶统计：左/右摆动驱动 dL,dR、左/右步幅、负步幅率、
exploration_std、eval 曲线，并与上一轮 SAC_18_3 对照。
"""
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "18(cur)": "sac_humanoid_walk/SAC_18/events.out.tfevents.1791531914.guo-MYG-XX.23033.0",
}
FLOOR, TOP, WEIGHT = 0.006, 0.03, 2.0


def load(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for t in ea.Tags()["scalars"]:
        s = ea.Scalars(t)
        out[t] = (np.array([x.step for x in s]), np.array([x.value for x in s]))
    return out


def series(d, tag):
    return d.get(tag, (np.array([]), np.array([])))


def bucket(d, tag, edges):
    steps, vals = series(d, tag)
    if len(vals) == 0:
        return None
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (steps >= lo) & (steps < hi)
        rows.append((lo, vals[m]))
    return rows


def fmt(tag, rows, agg):
    parts = []
    for lo, v in rows:
        parts.append(f"{lo/1000:.0f}K:{fmt_val(tag, agg(v))}")
    return "  ".join(parts)


def fmt_val(tag, x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  --  "
    if tag in ("gait/last_drive_L", "gait/last_drive_R"):
        return f"{x:6.4f}"
    if tag in ("gait/last_left_stride", "gait/last_right_stride"):
        return f"{x:6.3f}"
    return f"{x:6.3f}"


def main():
    import glob
    import os
    os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    p = sorted(glob.glob("sac_humanoid_walk/SAC_18/events.out.tfevents.*"), key=os.path.getmtime)[-1]
    d = load(p)
    print(f"RUN FILE: {p}")
    steps, _ = series(d, "gait/last_drive_L")
    print(f"last logged step = {steps.max():,}   points = {len(steps)}")
    _, es = series(d, "train/exploration_std")
    print(f"exploration_std  first={es[0]:.3f} last={es[-1]:.3f} min={es.min():.3f} max={es.max():.3f}")

    edges = list(range(0, int(steps.max()) + 50000, 50000))
    for tag in ("gait/last_drive_L", "gait/last_drive_R",
                "gait/last_left_stride", "gait/last_right_stride",
                "gait/negative_stride_rate", "gait/swing_drive_reward",
                "rollout/ep_len_mean", "train/action_std", "gait/speed",
                "gait/forward_gate", "gait/alternate_penalty_rate"):
        rows = bucket(d, tag, edges)
        if rows is None:
            continue
        med = [None if len(v) == 0 else float(np.median(v)) for _, v in rows]
        p90 = [None if len(v) == 0 else float(np.percentile(v, 90)) for _, v in rows]
        mean = [None if len(v) == 0 else float(np.mean(v)) for _, v in rows]
        lab = "K".join(f"{lo/1000:.0f}K" for lo, _ in rows)
        print(f"\n{tag}")
        print("  med :", "  ".join("--" if m is None else f"{m:7.4f}" for m in med))
        print("  p90 :", "  ".join("--" if m is None else f"{m:7.4f}" for m in p90))
        print("  mean:", "  ".join("--" if m is None else f"{m:7.4f}" for m in mean))
        print("  step:", "  ".join(f"{lo/1000:>7.0f}K" for lo, _ in rows))

    # drive_L 的分布细节（贴地板 vs 尖峰），分前/后半段
    _, dl = series(d, "gait/last_drive_L")
    half = len(dl) // 2
    for name, seg in (("前半", dl[:half]), ("后半", dl[half:])):
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue
        print(f"\ndrive_L {name}: n={len(seg)} med={np.median(seg):.4f} "
              f"<floor={np.mean(seg < FLOOR)*100:.1f}% "
              f"in[floor,top)={np.mean((seg >= FLOOR) & (seg < TOP))*100:.1f}% "
              f">=top={np.mean(seg >= TOP)*100:.1f}% "
              f">0.1={np.mean(seg > 0.1)*100:.1f}% p95={np.percentile(seg,95):.4f}")

    # eval
    ev, evr = series(d, "eval/mean_reward")
    _, evl = series(d, "eval/mean_ep_length")
    if len(evr):
        print("\nEVAL (step / reward / ep_len)")
        for i in range(0, len(evr), max(1, len(evr) // 20)):
            print(f"  {ev[i]:>9,}  {evr[i]:8.1f}  {evl[i]:7.1f}")
        best = int(np.argmax(evr))
        print(f"  BEST: step={ev[best]:,} reward={evr[best]:.1f} ep_len={evl[best]:.1f}")
        print(f"  last5 reward: {np.round(evr[-5:],1).tolist()}  last5 len: {np.round(evl[-5:],1).tolist()}")


if __name__ == "__main__":
    main()
