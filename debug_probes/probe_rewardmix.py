"""把 WalkRewardWrapper 第 934-954 行的 17 项奖励逐项拆解到每集，看占比。
info 没暴露的三项自己按公式补算：
  activity_diff_penalty = 0.5 * activity_difference^2  （单脚支撑时）
  survival_bonus        = 0.2 if speed > 0.5 else 0.0
  fall_penalty          = -50 （terminated 那一步）
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

POS = ["forward_reward", "swing_forward_reward", "stride_bonus", "net_balance",
       "height_reward", "phase_reward", "survival_bonus", "contact_phase_reward",
       "lead_credit", "fall_penalty"]
NEG = ["activity_diff_penalty", "support_sym_penalty", "alternate_penalty",
       "lateral_penalty", "hip_xz_angle_penalty", "flight_phase_penalty",
       "ctrl_cost", "short_stride_penalty", "negative_stride_penalty"]
ALT = ["lead_credit", "stride_bonus", "alternate_penalty"]  # 与交替直接相关
CZ = 0.08
SEEDS = [3, 5, 8, 13, 17, 19, 23, 29, 31, 37, 42, 44, 53, 55, 61, 71]


def probe(path, tag):
    m = SAC.load(path, buffer_size=20000)
    tot = {k: 0.0 for k in POS + NEG}
    tot["reward"] = 0.0
    lengths, n_fall = [], 0
    for sd in SEEDS:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        data = env.unwrapped.data
        n = 0
        while True:
            a, _ = m.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            n += 1
            tot["reward"] += float(r)
            for k in ["forward_reward", "swing_forward_reward", "stride_bonus", "net_balance",
                      "height_reward", "phase_reward", "contact_phase_reward", "lead_credit",
                      "support_sym_penalty", "alternate_penalty", "lateral_penalty",
                      "hip_xz_angle_penalty", "flight_phase_penalty", "ctrl_cost",
                      "short_stride_penalty", "negative_stride_penalty"]:
                tot[k] += float(info.get(k, 0.0))
            # 补算 info 未暴露的三项
            spd = float(info.get("speed", 0.0))
            tot["survival_bonus"] += 0.2 if spd > 0.5 else 0.0
            if te:
                tot["fall_penalty"] += -50.0
                n_fall += 1
            lx, rx = data.body("left_foot").xpos, data.body("right_foot").xpos
            lc_ = lx[2] < CZ
            rc_ = rx[2] < CZ
            if not lc_ or not rc_:
                tot["activity_diff_penalty"] += 0.5 * float(info.get("activity_difference", 0.0)) ** 2
            if te or tr:
                break
        lengths.append(n)
        env.close()
    m.replay_buffer = None
    del m
    gc.collect()

    n_ep = len(SEEDS)
    L = np.mean(lengths)
    P = sum(tot[k] for k in POS)
    N = sum(tot[k] for k in NEG)
    print("\n### %s  (%d 集, 平均 %d 步, 摔倒 %d 集)" % (tag, n_ep, L, n_fall))
    print("  总 reward/集 %9.1f   正项合计 %9.1f   负项合计 %9.1f (校验 正-负 = %9.1f)"
          % (tot["reward"] / n_ep, P / n_ep, -N / n_ep, (P - N) / n_ep))
    print()
    print("  %-24s %11s %8s %8s" % ("项", "每集", "占正项", "占负项"))
    for k in POS:
        print("  %-24s %+11.1f %7.1f%%" % (k, tot[k] / n_ep, 100 * tot[k] / P if P else 0))
    print("  " + "-" * 58)
    for k in NEG:
        v = -tot[k] / n_ep
        print("  %-24s %+11.1f %7.1f%%" % (k, v, 100 * v / (-N) if N else 0))
    print()
    a = sum(tot[k] for k in ALT) / n_ep
    print("  交替相关三项合计 %+.1f/集  (lead_credit %+.1f, stride_bonus %+.1f, alternate_penalty %+.1f)"
          % (a, tot["lead_credit"] / n_ep, tot["stride_bonus"] / n_ep, -tot["alternate_penalty"] / n_ep))
    print("  占总 reward %.1f%%   占正项 %.1f%%" % (100 * a / (tot["reward"] / n_ep), 100 * sum(tot[k] for k in ALT if k != "alternate_penalty") / P))
    print("  对照：稠密每步项合计 %+.1f/集" % ((P - tot["lead_credit"] - tot["stride_bonus"] - tot["fall_penalty"]) / n_ep))
    return {k: tot[k] / n_ep for k in POS + NEG + ["reward"]}


b = probe("best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip", "576K 基线")
c = probe("best_model/best_model.zip", "当前 best 560K")
print("\n=== 变化 (560K 减 576K) ===")
for k in ["reward"] + POS + NEG:
    d = c[k] - b[k]
    if abs(d) < 0.5 and k != "reward":
        continue
    print("  %-24s %+8.1f -> %+8.1f  (%+8.1f)" % (k, b[k], c[k], d))
