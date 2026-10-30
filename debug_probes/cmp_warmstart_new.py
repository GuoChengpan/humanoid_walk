#!/usr/bin/env python3
# 新奖励（phase min）下的暖启动候选对比
# 旧对比在旧奖励下测的；这次改动对不同模型的代价不同（左腿越僵掉得越多），必须重测
import os, sys, gc
sys.path.insert(0, '/home/guo/workspace/humanoid_walk')
os.chdir('/home/guo/workspace/humanoid_walk')
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

CANDS = [
    ('best_model.zip (944K, 当前run峰值)', 'best_model/best_model.zip'),
    ('tmp best (560K, 旧对照)',            'best_model/tmp_model_data/best_model.zip'),
    ('snapshot 848K',                      'best_model/best_model_snapshots/best_model_848000.zip'),
    ('snapshot 128K',                      'best_model/best_model_snapshots/best_model_128000.zip'),
]
SEEDS = (7, 11, 13, 17)

rows = []
for label, path in CANDS:
    model = SAC.load(path, buffer_size=20000)
    nt = model.num_timesteps
    rewards, phases, sl_all, sr_all, swaps, lens = [], [], [], [], [], []
    for s in SEEDS:
        env = WalkRewardWrapper(gym.make('Humanoid-v4', max_episode_steps=1500))
        obs, _ = env.reset(seed=s)
        tot = ph = 0.0
        nswap = 0; nstep = 0
        ls = rs = 0.0; cnt = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            tot += r; nstep += 1
            ph += info['phase_reward']
            if env.last_landing_swapped:
                nswap += 1
            L = np.array(env.left_hip_angle_buffer); R = np.array(env.right_hip_angle_buffer)
            if len(L) >= 50:
                ls += np.std(L - L.mean()); rs += np.std(R - R.mean()); cnt += 1
            if te or tr:
                break
        env.close()
        rewards.append(tot); phases.append(ph); lens.append(nstep)
        sl_all.append(ls/cnt if cnt else 0.0); sr_all.append(rs/cnt if cnt else 0.0)
        swaps.append(nswap)
        print('  %s seed %2d  步数 %4d  reward %8.1f  phase %6.1f  swap %2d'
              % (label.split('(')[0].strip()[:12], s, nstep, tot, ph, nswap))
    model.replay_buffer = None; del model; gc.collect()
    ml = max(np.mean(sl_all), 1e-9)
    rows.append((label, nt, np.mean(rewards), np.std(rewards), np.mean(phases),
                 np.mean(sl_all), np.mean(sr_all), np.mean(sr_all)/ml, np.mean(swaps)))

print('%-34s %8s %8s %8s %8s %7s %7s %6s %6s' %
      ('候选', '步数', 'reward', 'std', 'phase', 'stdL', 'stdR', 'R/L', 'swap'))
print('-'*110)
for label, nt, mr, sd, ph, sl, sr, ratio, sw in rows:
    print('%-34s %8d %8.1f %8.1f %8.1f %7.4f %7.4f %6.2f %6.0f' %
          (label, nt, mr, sd, ph, sl, sr, ratio, sw))

print()
print('旧奖励下（上一轮 cmp_warmstart 实测）: 560K 748.5/集 · 944K-run 峰值 tfevents 525.2 · 848K 未测 · 128K 未测')
print()
print('phase 一列 = 新奖励下实际拿到的分。它与旧值(~100)的差就是这次改动在该模型上的净扣分。')
print('stdL/stdR 越接近 1 越好；swap 是 last_landing_swapped 每集触发次数（0=完全不交替）。')
