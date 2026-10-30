"""零动作站立在当前 reward 下的收益 —— 躺平账的分母"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk"); os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np, gymnasium as gym
from train_humanoid_sac import WalkRewardWrapper

terms = ["forward_reward","survival_bonus","height_reward","stride_bonus","ctrl_cost",
         "phase_reward","contact_phase_reward","swing_forward_reward","net_balance",
         "stall_penalty","activity_difference","support_sym_penalty","lateral_penalty",
         "hip_xz_angle_penalty","flight_phase_penalty","short_stride_penalty","negative_stride_penalty"]
sums = {k:0.0 for k in terms}
tot_rew, tot_len, n_ep = 0.0, 0, 0
for ep in range(3):
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=1000+ep)
    ep_r = 0.0; n = 0
    while True:
        obs, r, te, tr, info = env.step(np.zeros(17, dtype=np.float32))
        ep_r += float(r); n += 1
        for k in terms:
            if k in info: sums[k] += float(info[k])
        if te or tr: break
    env.close()
    tot_rew += ep_r; tot_len += n; n_ep += 1
    print("集%d 站立 reward=%8.1f  len=%d" % (ep, ep_r, n))

print()
print("站立 3 集均值: reward %.1f / 集, ep_len %.0f" % (tot_rew/n_ep, tot_len/n_ep))
print("--- 分项（每集均值，正为奖励）---")
for k in terms:
    v = sums[k]/n_ep
    if abs(v) > 0.5: print("  %-24s %9.1f" % (k, v))
