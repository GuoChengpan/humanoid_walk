"""核心实验：给冻结的 560K 确定性策略注入不同强度的高斯噪声，
看 reward 随噪声强度怎么塌。
如果噪声一上来 reward 就崩，说明奖励地形本身排斥这个量级的探索 ——
那 ent_coef 减半只是减慢漂移，救不了自反馈。
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
NAMES = ["det (σ=0)", "σ=0.02", "σ=0.04", "σ=0.06", "σ=0.08", "σ=0.12", "原生随机"]
SIGMAS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12]

m = SAC.load(CURR, buffer_size=20000)


def run(sigma, native=False):
    res = []
    for sd in SEEDS:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        rew = 0.0
        n = 0
        while True:
            a, _ = m.predict(obs, deterministic=not native)
            a = np.asarray(a, dtype=np.float32)
            if sigma > 0:
                a = a + np.float32(np.random.default_rng(sd * 1000 + n).normal(0, sigma, a.shape))
            obs, r, te, tr, _ = env.step(a)
            rew += float(r)
            n += 1
            if te or tr:
                break
        env.close()
        res.append((rew, n))
    return res


print("模型: %s   每个条件 %d 集, 同 seed" % (CURR, len(SEEDS)))
print()
print("%-12s %10s %10s %8s %10s" % ("噪声", "reward均值", "reward范围", "走满率", "步数均值"))
for name, sg in zip(NAMES[:6], SIGMAS):
    res = run(sg)
    r = np.array([x[0] for x in res])
    ln = np.array([x[1] for x in res])
    print("%-12s %10.1f  %6.0f~%5.0f %7.0f%% %10.0f"
          % (name, r.mean(), r.min(), r.max(), 100 * (ln >= 1490).mean(), ln.mean()))
res = run(0.0, native=True)
r = np.array([x[0] for x in res])
ln = np.array([x[1] for x in res])
print("%-12s %10.1f  %6.0f~%5.0f %7.0f%% %10.0f"
      % ("原生随机", r.mean(), r.min(), r.max(), 100 * (ln >= 1490).mean(), ln.mean()))

m.replay_buffer = None
del m
gc.collect()
