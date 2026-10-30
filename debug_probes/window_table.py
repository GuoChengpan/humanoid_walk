"""跨 100K 窗均值表——§21/§22 钉的判决口径：**只用跨窗均值**，窗内斜率 R² 只有 0.01-0.05。

用法：python3 window_table.py [TB目录或文件] [--prev TB目录]
默认读当前活着的 run（按 PID 认，避免误读归档 run），并和上一个 run 的同窗对比。
"""
import glob
import os
import subprocess
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/home/guo/workspace/humanoid_walk"

GAUGES = [
    "gait/ev_rom_L_mean", "gait/ev_rom_R_mean",          # ⑤ 的主判据
    "gait/swing_min_reward_gated", "gait/reach_reward",
    "gait/flight_frac", "gait/single_frac", "gait/duty_sum", "gait/double_support_frac",
    "gait/speed", "gait/ev_reach_L_mean",
    "rollout/ep_len_mean", "train/exploration_std", "train/critic_loss",
    "eval/mean_reward",
]
STEP = 100_000


def live_file():
    pid = subprocess.run(["pgrep", "-f", "train_humanoid_sac.py"], capture_output=True,
                         text=True).stdout.split()
    cands = sorted(glob.glob(os.path.join(ROOT, "sac_humanoid_walk/*/events.out.tfevents*")),
                   key=os.path.getmtime, reverse=True)
    for p in cands:
        if os.path.basename(p).split(".")[-2] in pid:
            return p, "活进程 PID=" + "/".join(pid)
    return (cands[0], "无活进程，用最近写入的") if cands else (None, "没有 TB")


def load(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    return ea, set(ea.Tags()["scalars"])


def windows(ea, tag, n=STEP):
    if tag not in ea.Tags()["scalars"]:
        return None
    s = ea.Scalars(tag)
    if not s:
        return None
    last = s[-1].step
    out = []
    for lo in range(0, last + 1, n):
        v = [x.value for x in s if lo <= x.step < lo + n]
        if len(v) >= max(3, n // 4000):        # 窗内至少 25% 的点，否则不算这个窗
            out.append((lo, float(np.mean(v)), len(v)))
    return (out, last) if out else None


def main():
    arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = arg[0] if arg else None
    if path and os.path.isdir(path):
        path = sorted(glob.glob(os.path.join(path, "events.out.tfevents*")),
                      key=os.path.getmtime)[-1]
    src = None
    if not path:
        path, src = live_file()
    ea, tags = load(path)
    print("TB: %s  (%s)" % (path.replace(ROOT + "/", ""), src or "指定"))
    print("每格=该 100K 窗均值（括号内窗末 step，K）；判决只看这一列的走向\n")
    hdr = "%-30s" % "gauge"
    rows = {}
    for g in GAUGES:
        w = windows(ea, g)
        if w is None:
            continue
        ws, last = w
        rows[g] = ws
    nmax = max(len(v) for v in rows.values())
    print("%-30s%s" % ("gauge", "".join("%13s" % ("W%d" % i) for i in range(nmax))))
    for g, ws in rows.items():
        cells = []
        for i in range(nmax):
            if i < len(ws):
                v = ws[i][1]
                cells.append("%13.4f" % v if abs(v) < 100 else "%13.0f" % v)
            else:
                cells.append("%13s" % "—")
        print("%-30s%s" % (g.replace("gait/", "").replace("train/", "t/").replace("rollout/", "r/"),
                           "".join(cells)))
    last = max(v[-1][0] for v in rows.values()) + STEP
    print("\n当前约 %dK；W0=前100K，窗内点数不足 25%% 的窗已跳过" % (last // 1000))


if __name__ == "__main__":
    main()
