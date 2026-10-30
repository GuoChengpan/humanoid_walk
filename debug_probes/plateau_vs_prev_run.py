"""平台期判定 + 与上一轮 SAC_18_4 的同尺对照（只读 TB）。

两件事：
 1) 按 200K 分箱，把两轮 run 的**力学量**（flight/ds/stance/reach/rom/lift）并排打出来
    ——奖励定义变了所以 ep_rew 不可比，但接触判据两轮相同，力学量可比。
 2) 对本轮末段做线性拟合，判平台：斜率/拟合优度 + eval 最高纪录陈旧度。
"""
import glob
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/home/guo/workspace/humanoid_walk"
RUNS = {
    "18_4(旧奖励,跑满2M)": "sac_humanoid_walk/SAC_18_4_降速+改接触判定+日志逻辑等",
    "18_5(R''+ds门控,788K停)": "sac_humanoid_walk/SAC_18_5_加双支撑收入门控",
    "本轮(批次①+② single门控)": "sac_humanoid_walk/SAC_18",
}
KEY_OLD = "18_4(旧奖励,跑满2M)"
KEY_NOW = "本轮(批次①+② single门控)"
KEY_PREV = "18_5(R''+ds门控,788K停)"
BIN = 200000

MECH = ["gait/flight_frac", "gait/double_support_frac", "gait/stance_frac_L", "gait/stance_frac_R",
        "gait/ev_reach_L_mean", "gait/ev_reach_R_mean", "gait/ev_rom_L_mean", "gait/ev_rom_R_mean",
        "gait/ev_lift_L_mean", "gait/ev_lift_R_mean", "gait/ev_step_adv_L_mean", "gait/ev_step_adv_R_mean",
        "gait/speed", "gait/pitch_angle", "gait/support_sym_penalty", "gait/hip_xz_angle_penalty",
        "gait/height_reward", "gait/ds_gate", "gait/forward_gate"]
PERF = ["rollout/ep_len_mean", "rollout/ep_rew_mean", "eval/mean_reward",
        "train/exploration_std", "train/critic_loss"]

data = {}
for name, d in RUNS.items():
    p = sorted(glob.glob(os.path.join(ROOT, d, "events.out.tfevents*")))[-1]
    ea = EventAccumulator(p, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    ser = {}
    for t in tags:
        s = ea.Scalars(t)
        ser[t] = (np.array([x.step for x in s], float), np.array([x.value for x in s], float))
    data[name] = ser
    print(f"{name:22s} {os.path.basename(p)[:40]}  点数(ep_rew)={len(ser.get('rollout/ep_rew_mean',(0,0))[1])}  末step={ser['rollout/ep_rew_mean'][0][-1]:,.0f}")

for name, ser in data.items():
    print(f"\n===== {name} 力学量 200K 分箱 =====")
    maxstep = ser["rollout/ep_rew_mean"][0][-1]
    nb = int(maxstep // BIN) + 1
    print("%-26s" % "gauge" + "".join("%8s" % f"{k*BIN//1000}K" for k in range(nb)))
    for t in MECH + PERF:
        if t not in ser:
            continue
        st, v = ser[t]
        row = []
        for k in range(nb):
            m = (st >= k * BIN) & (st < (k + 1) * BIN)
            row.append(np.nanmean(v[m]) if m.sum() >= 3 else np.nan)
        print("%-26s" % t.split("/", 1)[1] + "".join("%8.3f" % x for x in row))

# 平台判定：本轮末段趋势
ser = data[KEY_NOW]
print("\n===== 本轮平台判定（末 300K 线性拟合）=====")
print("%-30s %10s %12s %10s %8s" % ("gauge", "末均值", "斜率/100K", "R²", "首末差"))
for t in ["rollout/ep_len_mean", "rollout/ep_rew_mean", "eval/mean_reward",
          "gait/double_support_frac", "gait/flight_frac", "gait/ds_gate",
          "gait/stance_frac_L", "gait/stance_frac_R", "gait/ev_reach_L_mean",
          "gait/ev_reach_R_mean", "gait/speed", "gait/pitch_angle", "train/exploration_std",
          "train/critic_loss", "train/entropy_loss", "train/actor_loss"]:
    if t not in ser:
        continue
    st, v = ser[t]
    m = st >= st[-1] - 300000
    s2, v2 = st[m], v[m]
    if len(s2) < 8:
        continue
    a, b = np.polyfit(s2 / 1e5, v2, 1)
    pred = a * s2 / 1e5 + b
    ss_res = float(np.sum((v2 - pred) ** 2))
    ss_tot = float(np.sum((v2 - v2.mean()) ** 2)) + 1e-12
    r2 = 1 - ss_res / ss_tot
    print("%-30s %10.3f %12.3f %10.2f %8.3f" % (t.split("/", 1)[1], v2[-1], a, r2, v2[-1] - v2[0]))

# eval 陈旧度
st, v = ser["eval/mean_reward"]
i = int(np.argmax(v))
last20 = v[-20:]
print(f"\neval: 最高 {v[i]:.0f} @ {st[i]:,.0f} | 全程均值 {v.mean():.0f} | 末20次均值 {last20.mean():.0f}"
      f" | 纪录已陈旧 {(st[-1]-st[i])/1000:.0f}K 步 | 末20次最高 {last20.max():.0f}")
ok = np.abs(v) < 1e5
print(f"eval 点数={len(v)} 采样间隔≈{np.median(np.diff(st)):,.0f} 步")

# 上一轮末段（1.4M-2M）的力学量，看它当年是否也停在同一点
ser4 = data["18_4(旧奖励,跑满2M)"]
print("\n===== 上一轮末段 1.4M-2.0M 力学量（它Plateau的终点在哪）=====")
for t in MECH:
    if t not in ser4:
        continue
    s4, v4 = ser4[t]
    m = s4 >= 1400000
    if m.sum() < 5:
        continue
    m2 = (s4 >= 1400000) & (s4 < 1700000)
    m3 = s4 >= 1700000
    print("%-26s 1.4-1.7M=%8.3f  1.7-2.0M=%8.3f" % (t.split("/", 1)[1], np.nanmean(v4[m2]), np.nanmean(v4[m3])))
s4, v4 = ser4["eval/mean_reward"]
print(f"上一轮 eval: 最高 {v4.max():.0f} @ {s4[int(np.argmax(v4))]:,.0f} | 末20次均值 {v4[-20:].mean():.0f}")
