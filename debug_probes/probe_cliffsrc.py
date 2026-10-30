"""定位噪声悬崖的来源：对比 det vs σ=0.08 时各奖励项的均值和"速度门"失败率。
height_reward 和 survival_bonus 都有硬门 advance/0.015 > 0.5（= info["speed"] > 0.5），
噪声一抖就掉 0.7/步。量化它占了多少。
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

CURR = "best_model/best_model.zip"
SEEDS = [7, 11, 23, 31, 42, 55]
TERMS = ["height_reward", "forward_reward", "ctrl_cost", "hip_xz_angle_penalty", "lateral_penalty"]
m = SAC.load(CURR, buffer_size=20000)


def run(sigma, tag):
    tot = dict(rew=0.0, n=0, gate_fail=0, **{k: 0.0 for k in TERMS})
    spd, hz = [], []
    for sd in SEEDS:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        k = 0
        while True:
            a, _ = m.predict(obs, deterministic=True)
            a = np.asarray(a, dtype=np.float32)
            if sigma > 0:
                a = a + np.float32(np.random.default_rng(sd * 7 + k).normal(0, sigma, a.shape))
            obs, r, te, tr, info = env.step(a)
            tot["rew"] += float(r)
            tot["n"] += 1
            s = float(info.get("speed", 0.0))
            spd.append(s)
            hz.append(float(info.get("height_reward", 0.0)))
            if s <= 0.5:
                tot["gate_fail"] += 1
            for key in TERMS:
                tot[key] += float(info.get(key, 0.0))
            k += 1
            if te or tr:
                break
        env.close()
    n = tot["n"]
    print("\n### %s (%d 步, %d 集)" % (tag, n, len(SEEDS)))
    print("  总 reward        %10.1f  (每步 %+.4f)" % (tot["rew"], tot["rew"] / n))
    print("  速度门失败占比   %7.2f%%   (%d / %d 步)" % (100 * tot["gate_fail"] / n, tot["gate_fail"], n))
    print("  speed            均值 %.4f  最小 %.4f" % (np.mean(spd), np.min(spd)))
    for key in TERMS:
        print("  %-22s %+.5f/步   每集 %+.1f" % (key, tot[key] / n, tot[key] / n * 1500))
    return tot


a = run(0.0, "det (σ=0)")
b = run(0.08, "σ=0.08")
print("\n=== 每步差值 (σ=0.08 减 det) ===")
na, nb = a["n"], b["n"]
for key in TERMS:
    print("  %-22s %+.5f/步   每集 %+.1f" % (key, b[key] / nb - a[key] / na, (b[key] / nb - a[key] / na) * 1500))
print("  总 reward/步             %+.5f   每集 %+.1f" % (b["rew"] / nb - a["rew"] / na, (b["rew"] / nb - a["rew"] / na) * 1500))
print("  速度门失败率             %+.2f 个百分点" % (100 * (b["gate_fail"] / nb - a["gate_fail"] / na)))
loss_a = a["gate_fail"] * 0.7
loss_b = b["gate_fail"] * 0.7
print("\n速度门失败若每步损失 height 0.5 + survival 0.2 = 0.7：")
print("  det 损失估算 %7.1f/集   σ=0.08 %7.1f/集   差 %+.1f/集" % (loss_a, loss_b, loss_b - loss_a))
m.replay_buffer = None
del m
gc.collect()
