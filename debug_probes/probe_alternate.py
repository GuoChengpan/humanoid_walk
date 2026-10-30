#!/usr/bin/env python3
# 直读 wrapper 自己的判定字段，确认「符号从不翻转」
# 注意：wrapper 在 step 末尾(第972行)才更新 left_contact_prev，所以落地要用自己记的接触跳变
import os, sys, gc
from collections import Counter
sys.path.insert(0, '/home/guo/workspace/humanoid_walk')
os.chdir('/home/guo/workspace/humanoid_walk')
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

model = SAC.load('best_model/best_model.zip', buffer_size=20000)

def run(seed):
    env = WalkRewardWrapper(gym.make('Humanoid-v4', max_episode_steps=1500))
    obs, _ = env.reset(seed=seed)
    d = env.unwrapped.data
    lands = []
    prev_lx = prev_rx = None
    stride = {'L': [], 'R': []}
    plc = prc = True
    lead_min = 0.0
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        lx = d.body('left_foot').xpos[0]
        rx = d.body('right_foot').xpos[0]
        clc = d.body('left_foot').xpos[2] < 0.08
        crc = d.body('right_foot').xpos[2] < 0.08
        diff = rx - lx
        if diff < lead_min:
            lead_min = diff
        jL = clc and not plc
        jR = crc and not prc
        if jL or jR:
            foot = 'L' if jL else 'R'
            if foot == 'L':
                if prev_lx is not None:
                    stride['L'].append(lx - prev_lx)
                prev_lx = lx
            else:
                if prev_rx is not None:
                    stride['R'].append(rx - prev_rx)
                prev_rx = rx
            lands.append((foot, diff, env.curr_position_state, env.last_landing_swapped,
                          info['alternate_penalty'], info['lead_credit'],
                          info['last_left_stride'], info['last_right_stride']))
        plc, prc = clc, crc
        if te or tr:
            break
    env.close()
    return lands, stride, lead_min

all_L, all_R, all_land = [], [], []
for s in (7, 11, 13, 17):
    lands, stride, lead_min = run(s)
    all_L += stride['L']; all_R += stride['R']; all_land += lands
    print('seed %2d  落地 %3d  真实步幅 左%+.4f 右%+.4f  右下界(right-left)最负 %+.4f'
          % (s, len(lands), np.median(stride['L']) if stride['L'] else float('nan'),
             np.median(stride['R']) if stride['R'] else float('nan'), lead_min))

L, R = np.array(all_L), np.array(all_R)
print('\n====== 真实步幅（同一只脚相邻两次落地的位移） ======')
print('左脚 n=%d  中位 %+.4f  p10 %+.4f  p90 %+.4f' % (len(L), np.median(L), np.percentile(L,10), np.percentile(L,90)))
print('右脚 n=%d  中位 %+.4f  p10 %+.4f  p90 %+.4f' % (len(R), np.median(R), np.percentile(R,10), np.percentile(R,90)))

print('\n====== 落地时 diff = right_x - left_x（交替 = 相邻落地符号翻转） ======')
for foot in ('L', 'R'):
    v = np.array([x[1] for x in all_land if x[0] == foot])
    print('%s脚落地 n=%d  中位 %+.4f  p10 %+.4f  p90 %+.4f  负值(左在前)占 %.1f%%'
          % (foot, len(v), np.median(v), np.percentile(v,10), np.percentile(v,90), 100.0*np.mean(v<0)))
d = np.array([x[1] for x in all_land])
flip = int(np.sum(d[:-1] * d[1:] < 0))
print('相邻落地符号翻转 %d / %d (%.1f%%)' % (flip, len(d)-1, 100.0*flip/(len(d)-1)))

print('\n====== wrapper 判定 ======')
print('位置状态分布', dict(Counter(x[2] for x in all_land)))
print('last_landing_swapped=True  %d / %d' % (sum(1 for x in all_land if x[3]), len(all_land)))
print('alternate_penalty >0(惩罚) %d   <0(奖励) %d   惩罚总量 %.1f'
      % (sum(1 for x in all_land if x[4] > 0), sum(1 for x in all_land if x[4] < 0), sum(x[4] for x in all_land)))
print('lead_credit>0 %d 次  合计 %.1f' % (sum(1 for x in all_land if x[5] > 0), sum(x[5] for x in all_land)))
print('position_swap_reward 触发 %d 次' % sum(1 for x in all_land if x[3]))

print('\n====== 前 14 次落地 ======')
for i, (f, dd, stt, sw, ap, lc, ls, rs) in enumerate(all_land[:14]):
    print('  %2d %s  diff=%+.3f  状态=%-12s  swap=%-5s  alt=%+.3f  lead=%+.2f'
          % (i, f, dd, stt, sw, ap, lc))

model.replay_buffer = None; del model; gc.collect()
