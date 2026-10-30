"""确定性 eval 轨迹：把 eval_logs/eval_gait.jsonl 按"每次 eval 5 集"折成行，看走向。

为什么需要它（2026-10-11 的血泪）：训练 TB 的 `gait/ev_rom_L_mean` 是**随机策略** rollout 的量，
⑥ 把 ent_coef 提到 0.10 之后它被探索噪声抬高（W4=7.3°），而 det eval 同量只有 4.7°。
**判决键 ev_rom_L 必须用 det eval 这一份**，否则 ⑥ 会把噪声冒充成进步。

两个坑：
 · jsonl 没有 step/run 字段，路径写死 → 新 run 是**追加**在旧 run 的行后面的。
   本脚本默认取"末尾 5×N 行"并把第 1 次 eval 当作 warm-start 基线（16K 时策略≈起点权重）。
   要跨 run 比较必须先数行数（本 run 的行数 = TB 里 eval/mean_reward 的点数 × 5）。
 · `_n` 结尾的键是滚动窗的**样本计数**，不是物理量（用户 2026-10-11 问的"一个>10 一个<10"就是它）。
"""
import json
import os
import sys

import numpy as np

ROOT = "/home/guo/workspace/humanoid_walk"
PATH = os.environ.get("EVAL_JSONL", os.path.join(ROOT, "eval_logs/eval_gait.jsonl"))
COLS = [("romL", "ev_rom_L_mean", "%6.2f"), ("romR", "ev_rom_R_mean", "%6.1f"),
        ("single", "single_frac", "%6.3f"), ("flight", "flight_frac", "%6.3f"),
        ("stL", "stance_frac_L", "%6.3f"), ("stR", "stance_frac_R", "%6.3f"),
        ("reachL", "ev_reach_L_mean", "%+6.3f"), ("liftL", "ev_lift_L_mean", "%6.3f"),
        ("kneeL", "ev_knee_ext_L_mean", "%6.1f"), ("sp_L", "ev_spacing_L_mean", "%6.3f"),
        ("sp_R", "ev_spacing_R_mean", "%6.3f"), ("v", "speed", "%6.2f"),
        ("len", "ep_len", "%5.0f"), ("ret", "ret", "%6.0f")]


def main():
    tail = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    if tail:
        rows = rows[-tail * 5:]
    n = len(rows) // 5
    print("%s：共 %d 行 = %d 次 eval（若本文件被多个 run 追加过，用 'python3 本脚本 29' 只取末尾 29 次）\n"
          % (os.path.basename(PATH), len(rows), n))
    print("%-4s %7s | %s" % ("ev", "step", " | ".join("%6s" % c[0] for c in COLS)))
    for i in range(0, len(rows) - 4, 5):
        g = rows[i:i + 5]
        ev = i // 5 + 1
        cells = []
        for _, key, fmt in COLS:
            v = [r[key] for r in g if key in r]
            cells.append(fmt % float(np.mean(v)) if v else "     —")
        print("%-4d %7d | %s" % (ev, ev * 16000, " | ".join(cells)))
    # 首尾对比（判决用：det 的 romL 有没有离开 warm-start 的值）
    first, last = rows[:5], rows[-5:]
    print("\n判决键 det ev_rom_L_mean：首次 eval %.2f° → 末次 %.2f°（Δ %+.2f°）；"
          "同期 flight %.3f→%.3f、single %.3f→%.3f、stance_L %.3f→%.3f" % (
              np.mean([r["ev_rom_L_mean"] for r in first]), np.mean([r["ev_rom_L_mean"] for r in last]),
              np.mean([r["ev_rom_L_mean"] for r in last]) - np.mean([r["ev_rom_L_mean"] for r in first]),
              np.mean([r["flight_frac"] for r in first]), np.mean([r["flight_frac"] for r in last]),
              np.mean([r["single_frac"] for r in first]), np.mean([r["single_frac"] for r in last]),
              np.mean([r["stance_frac_L"] for r in first]), np.mean([r["stance_frac_L"] for r in last])))


if __name__ == "__main__":
    main()
