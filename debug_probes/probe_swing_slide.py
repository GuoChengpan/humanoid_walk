#!/usr/bin/env python3
# 把每只脚 53cm/周期的位移拆成 支撑相位移(滑) vs 摆动相位移(真迈步)，并测 hip_y 摆幅
import os, sys, gc
sys.path.insert(0, '/home/guo/workspace/humanoid_walk')
os.chdir('/home/guo/workspace/humanoid_walk')
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

model = SAC.load('best_model/best_model.zip', buffer_size=20000)
Z = 0.08

def run(seed):
    env = WalkRewardWrapper(gym.make('Humanoid-v4', max_episode_steps=1500))
    obs, _ = env.reset(seed=seed)
    d = env.unwrapped.data
    jids = env._get_leg_joint_ids()
    hi_l, hi_r = jids['left_hip_y']['qpos'], jids['right_hip_y']['qpos']
    # 每只脚独立跟踪：stance_start = 落地时刻的 x；lift_x = 离地时刻的 x
    rec = {
        'L': {'stance_start': None, 'lift_x': None, 'last_land_x': None, 'swing': [], 'slide': [], 'cycle': []},
        'R': {'stance_start': None, 'lift_x': None, 'last_land_x': None, 'swing': [], 'slide': [], 'cycle': []},
    }
    plc = prc = True
    HL, HR = [], []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        lx = d.body('left_foot').xpos[0]; rx = d.body('right_foot').xpos[0]
        clc = d.body('left_foot').xpos[2] < Z; crc = d.body('right_foot').xpos[2] < Z
        HL.append(d.qpos[hi_l]); HR.append(d.qpos[hi_r])
        for foot, x, c, jc in (('L', lx, clc, clc and not plc), ('R', rx, crc, crc and not prc)):
            q = rec[foot]
            if jc:                                   # 落地
                if q['lift_x'] is not None:
                    q['swing'].append(x - q['lift_x'])   # 摆动相位移
                if q['last_land_x'] is not None:
                    q['cycle'].append(x - q['last_land_x'])
                q['last_land_x'] = x
                q['stance_start'] = x
                q['lift_x'] = None
            if q['stance_start'] is not None and q['lift_x'] is None and not c:
                q['lift_x'] = x                      # 离地
                q['slide'].append(x - q['stance_start'])   # 支撑相位移
                q['stance_start'] = None
        plc, prc = clc, crc
        if te or tr:
            break
    env.close()
    return rec, np.array(HL), np.array(HR)

out = {}
HLs = []
for s in (7, 11, 13, 17):
    rec, HL, HR = run(s)
    for f in ('L', 'R'):
        for k in ('swing', 'slide', 'cycle'):
            out.setdefault((f, k), []).extend(rec[f][k])
    HLs.append((HL, HR))

print('====== 每周期位移的构成（4 seed 合计） ======')
for f in ('L', 'R'):
    c, w, s = np.array(out[(f,'cycle')]), np.array(out[(f,'swing')]), np.array(out[(f,'slide')])
    print('%s脚  周期位移 中位 %+.4f (n=%d) | 摆动相 %+.4f (n=%d) | 支撑相(滑) %+.4f (n=%d)'
          % (f, np.median(c), len(c), np.median(w), len(w), np.median(s), len(s)))
    if len(c):
        print('      摆动相占比 %.1f%%   支撑相占比 %.1f%%' % (100.0*np.median(w)/np.median(c), 100.0*np.median(s)/np.median(c)))

print('\n====== hip_y 摆幅与相位（4 seed） ======')
for i, (HL, HR) in enumerate(HLs):
    print('seed %d: 左幅 %.3f rad  右幅 %.3f rad  反相关 %.3f'
          % (7+4*i, HL.max()-HL.min(), HR.max()-HR.min(), np.corrcoef(HL, HR)[0,1]))
model.replay_buffer = None; del model; gc.collect()
