"""解析当前 SAC_17 run：暖启动 560K + stride_balance_penalty 2.0。

重点不是 reward，是新加的三个 tag 有没有真的把左脚抬起来：
  gait/last_left_stride   期望从 ~0.03 朝 0.38 涨
  gait/last_right_stride  期望稳定在 ~0.38
  gait/stride_balance_penalty  期望 ~110/集 起步，若左脚抬上来应该往下走
"""
import os, sys
os.chdir("/home/guo/workspace/humanoid_walk")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("sac_humanoid_walk/SAC_17")
ea.Reload()
S = {k: ea.Scalars(k) for k in ea.Tags()["scalars"]}

last = S["rollout/ep_rew_mean"][-1]
print("=" * 92)
print("最新 step: %d   ep_rew_mean=%6.1f   %d 条 rollout / %d 条 eval" % (
    last.step, last.value, len(S["rollout/ep_rew_mean"]), len(S["eval/mean_reward"])))
print("=" * 92)

print("\n【eval/mean_reward 全部 %d 个】" % len(S["eval/mean_reward"]))
ev = S["eval/mean_reward"]
for i, s in enumerate(ev):
    mark = "  ← 最新" if i == len(ev) - 1 else ("  ← 峰值" if s.value == max(x.value for x in ev) else "")
    print("   %9d  %8.1f%s" % (s.step, s.value, mark))

print("\n【eval/mean_ep_length 全部 %d 个】" % len(S["eval/mean_ep_length"]))
for s in S["eval/mean_ep_length"]:
    print("   %9d  %6.1f" % (s.step, s.value))

# 按 step 分桶的轨迹
def bucket(tag, n=10):
    xs = S[tag]
    out = []
    for i, s in enumerate(xs):
        b = min(n - 1, i * n // len(xs))
        if b >= len(out):
            out.append([])
        out[b].append(s)
    return out

KEY = [
    ("rollout/ep_rew_mean", "ep_rew/集", 6),
    ("gait/stride_balance_penalty", "balance罚", 7),
    ("gait/last_left_stride", "左脚距(m)", 5),
    ("gait/last_right_stride", "右脚距(m)", 5),
    ("gait/activity_difference", "活动差", 4),
    ("gait/speed", "速度", 4),
    ("gait/forward_reward", "前进奖", 7),
    ("gait/hip_xz_angle_penalty", "hipxz罚", 7),
    ("gait/height_reward", "高度奖", 6),
    ("gait/ctrl_cost", "动作成本", 7),
    ("gait/stride_bonus", "步幅奖", 7),
    ("gait/stall_max", "stall_max", 4),
]

print("\n【关键项按 10 等分桶（每桶取区间末 25%% 均值）】")
hdr = "%-26s %s" % ("step区间", "  ".join("%9s" % h for h, _, _ in KEY))
print(hdr)
print("-" * len(hdr))
for b in bucket("rollout/ep_rew_mean", 10):
    if not b:
        continue
    lo, hi = b[0].step, b[-1].step
    vals = []
    for tag, _, _ in KEY:
        seg = [x.value for x in S[tag] if lo - 1 <= x.step <= hi]
        tail = seg[-max(1, len(seg) // 4):] if seg else [float("nan")]
        vals.append(sum(tail) / len(tail))
    row = "  ".join("%9.2f" % v for v in vals)
    print("%5.2fM-%5.2fM %s" % (lo / 1e6, hi / 1e6, row))

print("\n【最近 8 个 eval 附近的关键项（逐条，非分桶）】")
xs = S["gait/last_left_stride"]
for tag, lab, w in [("gait/last_left_stride", "左脚距", 5),
                    ("gait/last_right_stride", "右脚距", 5),
                    ("gait/stride_balance_penalty", "balance罚", 7),
                    ("rollout/ep_rew_mean", "ep_rew", 6)]:
    vals = S[tag][-8:]
    print("  %-12s %s" % (lab, "  ".join("%8.*f" % (w, v.value) for v in vals)))

print("\n【训练侧】")
for tag in ["train/ent_coef", "train/learning_rate", "train/actor_loss",
            "train/critic_loss", "train/exploration_std", "train/action_std"]:
    xs = S[tag]
    print("  %-24s %d 条   首 %.4g  →  末 %.4g   min %.4g / max %.4g" % (
        tag, len(xs), xs[0].value, xs[-1].value, min(x.value for x in xs), max(x.value for x in xs)))

print("\n【新惩罚的覆盖率：balance 罚不为 0 的采样占比】")
bp = S["gait/stride_balance_penalty"]
nz = [x for x in bp if abs(x.value) > 1e-9]
print("  %d/%d = %.1f%%  非零；非零时均值 %.2f，最大 %.2f" % (
    len(nz), len(bp), 100.0 * len(nz) / len(bp),
    sum(x.value for x in nz) / max(len(nz), 1), max((x.value for x in nz), default=0.0)))
