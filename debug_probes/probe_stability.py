"""评估两个模型的稳定性：满集率 / 摔倒率 / 摔倒前步数 / 摔倒时躯干高度。
用来判断最近的 eval 下滑到底是"走得更差"还是"摔得更频繁"。
buffer_size=20000 —— zip 内不含 replay_buffer，避免默认 9GB 分配挤爆训练进程的内存。
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

BASE = "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"
CURR = "best_model/best_model.zip"
N, SEEDS = 12, [3, 5, 8, 13, 17, 19, 29, 37, 44, 53, 61, 71]


def probe(path, tag):
    m = SAC.load(path, buffer_size=20000)
    res = []
    for sd in SEEDS:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        data = env.unwrapped.data
        rew = 0.0
        n = 0
        h_min = 1e9
        while True:
            a, _ = m.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            rew += float(r)
            n += 1
            h_min = min(h_min, float(data.qpos[2]))
            if te or tr:
                break
        fallen = bool(te)          # terminated = 摔倒；truncated = 走满 1500
        env.close()
        res.append((n, fallen, rew, h_min))
    m.replay_buffer = None
    del m
    gc.collect()

    arr = np.array([(r[0], r[1]) for r in res])
    falls = [r for r in res if r[1]]
    fulls = [r for r in res if not r[1]]
    print("\n### %s (%d 集)" % (tag, N))
    for sd, r in zip(SEEDS, res):
        print("  seed=%-3d 步=%4d %s reward=%8.1f 躯干最低=%.2f m"
              % (sd, r[0], "摔倒" if r[1] else "走满", r[2], r[3]))
    print("  ---- 汇总")
    print("  走满 1500 集: %d/%d (%.0f%%)   摔倒: %d/%d (%.0f%%)"
          % (len(fulls), N, 100 * len(fulls) / N, len(falls), N, 100 * len(falls) / N))
    if fulls:
        print("  走满集 reward 均值 %7.1f" % np.mean([r[2] for r in fulls]))
    if falls:
        print("  摔倒集 步数均值 %6.0f (范围 %d-%d)   reward 均值 %7.1f"
              % (np.mean([r[0] for r in falls]), min(r[0] for r in falls),
                 max(r[0] for r in falls), np.mean([r[2] for r in falls])))
    print("  全部集 reward 均值 %7.1f" % np.mean([r[2] for r in res]))
    return len(fulls) / N, np.mean([r[2] for r in res])


bf, br = probe(BASE, "576K 基线")
cf, cr = probe(CURR, "当前 best 560K")
print("\n=== 对比（deterministic, 各 %d 集, 同 seed）===" % N)
print("走满率:  576K %5.0f%%  →  560K %5.0f%%" % (100 * bf, 100 * cf))
print("reward:  576K %7.1f  →  560K %7.1f" % (br, cr))
