"""576K 全落地序列：alternate_penalty 实际净收益 vs 真交替的理论收益"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk"); os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

PATH = "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"
W = 2.3; SCALE = 0.18; CZ = 0.08
seed_env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
model = SAC.load(PATH, env=seed_env); del model.replay_buffer; gc.collect()

neg_sum = pos_sum = 0.0; n_flip = n_noflip = 0; alldiff = []
for ep in range(1, 4):
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=1000 + ep); data = env.unwrapped.data
    prev = None; pl = pr = True
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, _ = env.step(np.asarray(a, dtype=np.float32))
        lx, rx = data.body("left_foot").xpos[0], data.body("right_foot").xpos[0]
        lc, rc = data.body("left_foot").xpos[2] < CZ, data.body("right_foot").xpos[2] < CZ
        if ((lc and not pl) or (rc and not pr)):
            d = (rx - lx) / SCALE
            if abs(d) >= 0.01:
                if prev is not None:
                    p = prev * d
                    if p < 0: neg_sum += p; n_flip += 1
                    else:     pos_sum += p; n_noflip += 1
                prev = d; alldiff.append(d)
        pl, pr = lc, rc
        if te or tr: break
    env.close()

n = n_flip + n_noflip
print("3 集落地 %d 次  (flip %d / noflip %d)" % (n, n_flip, n_noflip))
print("flip 率 %.1f%%   (wrapper 口径 alt_r ≈ 同)" % (100*n_flip/n))
print()
print("=== alternate_penalty 实际账（×%.1f）===" % W)
print("  flip 奖励   Σproduct=%.2f → %+.1f 分/3集 = %+.1f 分/集" % (neg_sum, -neg_sum*W, -neg_sum*W/3))
print("  noflip 惩罚 Σproduct=%.2f → %+.1f 分/3集 = %+.1f 分/集" % (pos_sum, -pos_sum*W, -pos_sum*W/3))
print("  净计 %+.1f 分/集" % (-(neg_sum+pos_sum)*W/3))
print()
print("落地 diff 分布: 均值 %+.2f  中位 %+.2f  负值占比 %.1f%%  |diff|<0.56(=10cm) 占比 %.1f%%"
      % (np.mean(alldiff), np.median(alldiff), 100*np.mean(np.array(alldiff)<0),
         100*np.mean(np.abs(alldiff)<0.556)))
flipmag = [abs(p) for p in []]
print()
print("=== 如果真交替（R 落 +2.0 / L 落 -1.5，即 0.36m/0.27m 步幅）===")
print("  每落地 product = -3.0 → 奖励 %+.1f × %d 落地/集 = %+.0f 分/集" % (3.0*W, n//3, 3.0*W*(n//3)))
print("  vs 当前 %+.1f → 差距 %.0f 分/集 (占总分 %.0f%%)" % (-(neg_sum+pos_sum)*W/3,
      (3.0*W*(n//3)) - (-(neg_sum+pos_sum)*W/3), 100*((3.0*W*(n//3))-( -(neg_sum+pos_sum)*W/3))/1295))
