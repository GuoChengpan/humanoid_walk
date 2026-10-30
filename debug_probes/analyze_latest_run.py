"""分析当前 run 的 TB 指标：摆动驱动 / 左右步幅 / 训练健康。"""
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = sorted(glob.glob("sac_humanoid_walk/SAC_18/events.out.tfevents.*.23033.0"))[0]
ea = EventAccumulator(path, size_guidance={"scalars": 0})
ea.Reload()

def series(tag):
    s = ea.Scalars(tag)
    return np.array([x.step for x in s]), np.array([x.value for x in s])

def q(v):
    if len(v) == 0:
        return "n/a"
    return (f"n={len(v)} med={np.median(v):.4f} mean={v.mean():.4f} "
            f"p10={np.percentile(v,10):.4f} p90={np.percentile(v,90):.4f} max={v.max():.4f}")

step, driveL = series("gait/last_drive_L")
_, driveR = series("gait/last_drive_R")
_, lstr = series("gait/last_left_stride")
_, rstr = series("gait/last_right_stride")
_, std = series("train/action_std")
_, exstd = series("train/exploration_std")
_, negrate = series("gait/negative_stride_rate")
_, shorate = series("gait/short_stride_event_rate")
_, sdrew = series("gait/swing_drive_reward")
_, fwd = series("gait/forward_reward")
_, height = series("gait/height_reward")
_, actdiff = series("gait/activity_difference")
_, swap = series("gait/position_swap_rate")
_, speed = series("gait/speed")
_, epmlen = series("rollout/ep_len_mean")
_, eprew = series("rollout/ep_rew_mean")
_, lr = series("train/learning_rate")
_, crit = series("train/critic_loss")
_, act = series("train/actor_loss")
_, ent = series("train/ent_coef")
_, stall = series("gait/stall_penalty")
_, xz = series("gait/hip_xz_angle_penalty")

print(f"event file steps: {int(step.min())} .. {int(step.max())}  ({len(step)} pts)")
print("\n=== 摆动驱动 (floor=0.006, top=0.03, weight=2.0) ===")
print(f"driveL  all : {q(driveL)}")
print(f"driveR  all : {q(driveR)}")
for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
    a, b = int(len(driveL) * lo), int(len(driveL) * hi)
    wL, wR = driveL[a:b], driveR[a:b]
    if len(wL) == 0:
        continue
    print(f"  win {int(step[a])//1000:>4}K-{int(step[b-1])//1000:>4}K  "
          f"L med={np.median(wL):.4f} p75={np.percentile(wL,75):.4f} p90={np.percentile(wL,90):.4f} "
          f"| >floor {100*np.mean(wL>0.006):.0f}% >top {100*np.mean(wL>0.03):.0f}% >0.1 {100*np.mean(wL>0.1):.1f}%"
          f"  R med={np.median(wR):.4f}")

print("\n=== 左右步幅 ===")
for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
    a, b = int(len(lstr) * lo), int(len(lstr) * hi)
    if b - a == 0:
        continue
    L, R = lstr[a:b], rstr[a:b]
    m = np.mean(np.minimum(L, R))
    print(f"  win {int(step[a])//1000:>4}K-{int(step[b-1])//1000:>4}K  "
          f"L med={np.median(L):.3f} R med={np.median(R):.3f} "
          f"L/R={np.median(L)/max(np.median(R),1e-6):.2f} min(L,R) mean={m:.3f}")

print("\n=== 训练健康 ===")
print(f"action_std     : {q(std)}")
print(f"exploration_std: {q(exstd)}")
print(f"learning_rate  : last={lr[-1]:.3e}")
print(f"critic_loss    : {q(crit)}")
print(f"actor_loss     : {q(act)}")
print(f"ent_coef       : {q(ent)}")
print(f"ep_len_mean    : {q(epmlen)}")
print(f"ep_rew_mean    : {q(eprew)}")

print("\n=== 奖励项 / 行为率 ===")
for name, v in [("swing_drive_reward", sdrew), ("forward_reward", fwd), ("height", height),
                ("neg_stride_rate", negrate), ("short_stride_rate", shorate),
                ("activity_diff", actdiff), ("swap_rate", swap), ("speed", speed),
                ("stall_penalty", stall), ("hip_xz_penalty", xz)]:
    print(f"{name:>20}: {q(v)}")

if "eval/mean_reward" in ea.Tags()["scalars"]:
    es, ev = series("eval/mean_reward")
    _, el = series("eval/mean_ep_length")
    print("\n=== eval ===")
    for i in range(len(es)):
        print(f"  {int(es[i]):>8}K  rew={ev[i]:8.1f}  len={el[i]:7.1f}")
