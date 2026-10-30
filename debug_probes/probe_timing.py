"""左脚落地瞬间右脚的高度与离地时刻：区分'差一点双支撑' vs '结构性腾空'"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk"); os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

PATH = "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"
CZ = 0.08
seed_env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
model = SAC.load(PATH, env=seed_env); del model.replay_buffer; gc.collect()

rz_atL, lz_atR, gapR_L, gapL_R = [], [], [], []
for ep in range(1, 4):
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=1000 + ep); data = env.unwrapped.data
    pl = pr = True
    t_R_lift = t_L_lift = None   # 右/左脚离地的帧号
    t = 0
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, _ = env.step(np.asarray(a, dtype=np.float32))
        t += 1
        lz = data.body("left_foot").xpos[2]; rz = data.body("right_foot").xpos[2]
        lc, rc = lz < CZ, rz < CZ
        if lc and not pl:                      # 左脚落地
            rz_atL.append(rz)
            if t_R_lift is not None: gapR_L.append(t - t_R_lift)
        if rc and not pr:                      # 右脚落地
            lz_atR.append(lz)
            if t_L_lift is not None: gapL_R.append(t - t_L_lift)
        if rc and not pr is False: pass
        if (not rc) and pr: t_R_lift = t       # 右脚离地
        if (not lc) and pl: t_L_lift = t       # 左脚离地
        pl, pr = lc, rc
        if te or tr: break
    env.close()

rz_atL = np.array(rz_atL); lz_atR = np.array(lz_atR)
gapR_L = np.array(gapR_L); gapL_R = np.array(gapL_R)
print("左脚落地时右脚高度 rz 分布（3 集 n=%d）：" % len(rz_atL))
print("  <0.08(双支撑) %.1f%%   0.08-0.15 %.1f%%   0.15-0.25 %.1f%%   >0.25 %.1f%%   中位 %.3f m" % (
    100*(rz_atL<0.08).mean(), 100*((rz_atL>=0.08)&(rz_atL<0.15)).mean(),
    100*((rz_atL>=0.15)&(rz_atL<0.25)).mean(), 100*(rz_atL>=0.25).mean(), np.median(rz_atL)))
print()
print("右脚落地时左脚高度：中位 %.3f m, <0.08 占 %.1f%%" % (np.median(lz_atR), 100*(lz_atR<0.08).mean()))
print()
print("右脚离地 → 左脚落地的间隔帧数：中位 %.0f  均值 %.1f  ≤2帧占 %.1f%%" % (
    np.median(gapR_L), gapR_L.mean(), 100*(gapR_L<=2).mean()))
print("左脚离地 → 右脚落地的间隔帧数：中位 %.0f  均值 %.1f  ≤2帧占 %.1f%%" % (
    np.median(gapL_R), gapL_R.mean(), 100*(gapL_R<=2).mean()))
