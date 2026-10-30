"""当前 run 的奖励收支表（只读 TB，不动训练进程/模型）。

把每个 per-step 计价的键换算成"每 1500 步集"的钱，按收入/税分列，
用来回答：平地收入(height/survival)是否大到让策略没有换步态的动力。
"""
import glob
import os
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/home/guo/workspace/humanoid_walk"
EP = 1500  # 换算口径：满寿命集


def newest():
    c = [p for p in glob.glob(os.path.join(ROOT, "sac_humanoid_walk/*/events.out.tfevents*"))
         if os.path.getsize(p) > 1000]
    return max(c, key=os.path.getmtime)


path = sys.argv[1] if len(sys.argv) > 1 else newest()
print(f"TB: {path}")
ea = EventAccumulator(path, size_guidance={"scalar": 0})
ea.Reload()
tags = ea.Tags().get("scalar", [])

rows = []
for t in tags:
    if not (t.startswith("gait/") or t.startswith("rollout/") or t.startswith("train/")):
        continue
    s = ea.Scalars(t)
    v = np.array([x.value for x in s], dtype=np.float64)
    if len(v) < 16 or not np.isfinite(v).any():
        continue
    k = max(1, len(v) // 8)
    head = float(np.nanmean(v[:k]))
    tail = float(np.nanmean(v[-k:]))
    rows.append((t, head, tail, s[-1].step, v))

KEYS = ["height_reward", "survival_bonus", "forward_reward", "forward_gauss", "speed_penalty",
        "forward_gate", "walk_gate", "single_frac", "duty_sum", "ds_landing_rate",
        "reach_reward", "stride_bonus", "stride_bonus_gated", "position_swap_gated",
        "position_swap_reward", "alternate_penalty", "last_alternate_penalty",
        "support_sym_penalty", "hip_xz_angle_penalty", "flight_phase_penalty",
        "slip_penalty", "lateral_penalty", "pitch_penalty", "roll_penalty",
        "ctrl_cost", "fall_penalty", "stride_balance_penalty", "negative_stride_penalty",
        "short_stride_penalty", "last_left_stride", "last_right_stride",
        "double_support_frac", "flight_frac", "stance_frac_L", "stance_frac_R",
        "duty_sum", "speed", "ev_reach_L_mean", "ev_reach_R_mean", "ev_slip_L_mean",
        "ev_step_adv_L_mean", "ev_step_adv_R_mean", "ev_lift_L_mean", "ev_lift_R_mean",
        "ev_rom_L_mean", "ev_rom_R_mean", "last_reach_L", "last_reach_R",
        "left_swing_steps", "right_swing_steps", "swing_steps_L", "swing_steps_R"]

by = {r[0]: r for r in rows}
print("\n%-34s %10s %10s %12s %12s" % ("gait 键", "首1/8", "末1/8", "末1/8×1500", "备注"))
print("-" * 92)
for k in KEYS:
    t = "gait/" + k
    if t not in by:
        continue
    _, h, tl, st, _ = by[t]
    print("%-34s %10.4f %10.4f %12.1f" % (k, h, tl, tl * EP))
for k in ["rollout/ep_len_mean", "rollout/ep_rew_mean", "train/exploration_std",
          "eval/mean_reward", "train/sync_ratio"]:
    if k in by:
        _, h, tl, st, _ = by[k]
        print("%-34s %10.1f %10.1f" % (k, h, tl))

print("\n=== 末段可用的其它 gait 键（全列出，便于发现漏项）===")
shown = {"gait/" + k for k in KEYS}
extra = [r for r in rows if r[0].startswith("gait/") and r[0] not in shown]
for t, h, tl, st, _ in sorted(extra, key=lambda r: -abs(r[2])):
    print("%-40s %10.4f %10.4f %12.1f" % (t[5:], h, tl, tl * EP))
