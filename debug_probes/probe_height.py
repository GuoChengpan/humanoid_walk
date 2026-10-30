"""躯干 z 分布：初始化自然站高、576K 步态的高度分布与起伏幅度"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk"); os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

env0 = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
obs, _ = env0.reset(seed=1)
z0 = env0.unwrapped.data.qpos[2]
print("reset 后自然站高 torso z = %.3f" % z0)

seed_env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
model = SAC.load("best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip", env=seed_env)
del model.replay_buffer; gc.collect()

allz, rngs = [], []
for ep in range(1, 4):
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=1000+ep)
    zs = []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        zs.append(float(env.unwrapped.data.qpos[2]))
        if te or tr: break
    env.close()
    zs = np.array(zs); allz.append(zs)
    # 周期起伏幅度：滑动 25 帧(≈1步)窗口的极差
    k = 25
    rr = [zs[i:i+k].max()-zs[i:i+k].min() for i in range(0, len(zs)-k, k)]
    rngs.append(np.mean(rr))
    print("集%d: z 均值 %.3f  p5 %.3f  p95 %.3f  步内起伏均值 %.3f m" % (ep, zs.mean(), np.percentile(zs,5), np.percentile(zs,95), np.mean(rr)))

z = np.concatenate(allz)
print()
print("3 集汇总: z 均值 %.3f  中位 %.3f  p5 %.3f  p95 %.3f" % (z.mean(), np.median(z), np.percentile(z,5), np.percentile(z,95)))
print("在 1.1-1.3 窗口内帧占比 %.1f%%   在 1.3-1.45 占比 %.1f%%" % (100*((z>=1.1)&(z<=1.3)).mean(), 100*((z>1.3)&(z<=1.45)).mean()))
print("每步(25帧)起伏幅度均值 %.1f cm" % (100*np.mean(rngs)))
