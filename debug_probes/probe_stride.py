"""复刻 wrapper 的 stride 结算，看 shuffle 实际从 stride_bonus 拿多少"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk"); os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

PATH = "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"
CZ, SCALE = 0.08, 0.18
MIN_SPACING, STRIDE_WEIGHT, STRIDE_MAX = 0.2, 20.0, 0.12
seed_env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
model = SAC.load(PATH, env=seed_env); del model.replay_buffer; gc.collect()

tot = dict(n_ep=0, L_ev=0, R_ev=0, L_bonus=0.0, R_bonus=0.0, L_half=0, R_half=0,
           L_short=0, R_short=0, L_neg=0, R_neg=0, ds_frames=0, L_land=0, R_land=0)
for ep in range(1, 4):
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=1000 + ep); data = env.unwrapped.data
    pl = pr = True; alt_prev = None
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        lx, rx = data.body("left_foot").xpos[0], data.body("right_foot").xpos[0]
        lc, rc = data.body("left_foot").xpos[2] < CZ, data.body("right_foot").xpos[2] < CZ
        if lc and rc: tot['ds_frames'] += 1
        # 复刻 alternate_result（只在落地沿更新）
        d = (rx - lx) / SCALE
        alt = 0.0
        if abs(d) >= 0.01:
            if (lc and not pl) or (rc and not pr):
                if alt_prev is not None: alt = alt_prev * d
                alt_prev = d
        if lc and not pl: tot['L_land'] += 1
        if rc and not pr: tot['R_land'] += 1
        if (not pl) and lc and rc:            # 左脚落地且双支撑
            s = lx - rx; tot['L_ev'] += 1
            if s > MIN_SPACING:
                b = np.clip((s-MIN_SPACING)/STRIDE_MAX,0,1)*STRIDE_WEIGHT
                if alt > 0: b *= 0.5; tot['L_half'] += 1
                tot['L_bonus'] += b
            elif s > 0: tot['L_short'] += 1
            else: tot['L_neg'] += 1
        if (not pr) and rc and lc:            # 右脚落地且双支撑
            s = rx - lx; tot['R_ev'] += 1
            if s > MIN_SPACING:
                b = np.clip((s-MIN_SPACING)/STRIDE_MAX,0,1)*STRIDE_WEIGHT
                if alt > 0: b *= 0.5; tot['R_half'] += 1
                tot['R_bonus'] += b
            elif s > 0: tot['R_short'] += 1
            else: tot['R_neg'] += 1
        pl, pr = lc, rc
        if te or tr: break
    env.close(); tot['n_ep'] += 1

n = tot['n_ep']
print("3 集, 落地 L=%d R=%d, 双支撑帧=%d, 步幅事件 L=%d R=%d" % (
    tot['L_land'], tot['R_land'], tot['ds_frames'], tot['L_ev'], tot['R_ev']))
print()
print("左脚事件: bonus %.1f 分/3集 | 打折 %d | short %d | neg %d" % (
    tot['L_bonus'], tot['L_half'], tot['L_short'], tot['L_neg']))
print("右脚事件: bonus %.1f 分/3集 | 打折 %d | short %d | neg %d" % (
    tot['R_bonus'], tot['R_half'], tot['R_short'], tot['R_neg']))
print()
print("stride_bonus 总计 %.1f 分/3集 = %.1f 分/集" % (tot['L_bonus']+tot['R_bonus'], (tot['L_bonus']+tot['R_bonus'])/3))
print("右脚贡献占 stride_bonus 的 %.1f%%" % (100*tot['R_bonus']/max(tot['L_bonus']+tot['R_bonus'],1e-9)))
