"""量化"交换门"离可触发还有多远：
在同一批落地序列上，用当前阈值 0.1m 和候选阈值 0.05m 分别跑一遍状态机，
统计 left_ahead 占比、LA->RA / RA->LA 直连转移次数，以及门控开启时 stride_bonus 能拿到多少。
不动训练代码，纯离线重放 560K 当前 best 的动作。
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

CURR = "best_model/best_model.zip"
CZ, SCALE = 0.08, 0.18
STRIDE_WEIGHT, STRIDE_MAX, MIN_SPACING = 20.0, 0.12, 0.2
SEEDS = [3, 5, 8, 13, 17, 19, 23, 29, 31, 37, 42, 44, 53, 55, 61, 71]


def step_factor(hip_gap, thr):
    lo, hi = thr, thr * 2.2
    return min(1.0, max(0.0, (hip_gap - lo) / (hi - lo)))


m = SAC.load(CURR, buffer_size=20000)
stat = {thr: dict(n_land=0, state={}, trans={}, swap=0, swap_gate_open=0,
                  stride_if_open=0.0, lf=[]) for thr in (0.10, 0.05, 0.03)}
for thr in stat:
    stat[thr]['prev'] = None

for sd in SEEDS:
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=sd)
    data = env.unwrapped.data
    prev_l = prev_r = True
    # 记录本步的落地信息，供状态机使用
    while True:
        a, _ = m.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        lx, rx = data.body("left_foot").xpos[0], data.body("right_foot").xpos[0]
        lc_ = data.body("left_foot").xpos[2] < CZ
        rc_ = data.body("right_foot").xpos[2] < CZ
        landed = (lc_ and not prev_l) or (rc_ and not prev_r)
        if landed:
            d = lx - rx                      # >0 左脚在前
            for thr in stat:
                if d > thr:
                    s = 'LA'
                elif d < -thr:
                    s = 'RA'
                else:
                    s = 'C'
                st = stat[thr]
                st['n_land'] += 1
                st['state'][s] = st['state'].get(s, 0) + 1
                if st['prev'] is not None:
                    key = (st['prev'], s)
                    st['trans'][key] = st['trans'].get(key, 0) + 1
                direct = ((st['prev'] == 'LA' and s == 'RA') or (st['prev'] == 'RA' and s == 'LA'))
                st['swap'] += int(direct)
                if direct:
                    gap = abs(lx - rx)
                    # 门控开启后，下一次落地若左(或右)脚 stride > 0.2 就能拿到 stride_bonus
                    st['swap_gate_open'] += 1
                    st['lf'].append(d)
                st['prev'] = s
        prev_l, prev_r = lc_, rc_
        if te or tr:
            break
    env.close()

m.replay_buffer = None
del m
gc.collect()

print("模型: %s   %d 集, 共 %d 次落地" % (CURR, len(SEEDS), stat[0.10]['n_land']))
print()
print("%-6s %10s %10s %10s %12s %12s %12s %10s" %
      ("阈值m", "LA占比", "RA占比", "C占比", "RA->LA", "LA->RA", "交换总数", "每次交换cm"))
for thr in (0.10, 0.05, 0.03):
    st = stat[thr]
    n = st['n_land']
    la = 100 * st['state'].get('LA', 0) / n
    ra = 100 * st['state'].get('RA', 0) / n
    c = 100 * st['state'].get('C', 0) / n
    r2l = st['trans'].get(('RA', 'LA'), 0)
    l2r = st['trans'].get(('LA', 'RA'), 0)
    sw = st['swap']
    if st['lf']:
        mag = 100 * np.mean(np.abs(st['lf']))
    else:
        mag = 0.0
    print("%-6.2f %9.2f%% %9.2f%% %9.2f%% %11d %11d %12d %9.1f"
          % (thr, la, ra, c, r2l, l2r, sw, mag))

print()
print("状态转移矩阵（当前阈值 0.10m）：")
st = stat[0.10]
for k in sorted(st['trans'], key=lambda x: -st['trans'][x]):
    print("  %-5s -> %-5s : %4d" % (k[0], k[1], st['trans'][k]))
print()
print("参考：stride 门要求下一次落地的脚间距 > %.1f m；当前落地间距分布见 cmp_alt.py 输出。" % MIN_SPACING)
print("参考：swap_reward = 5 x step_factor(hip_gap)，hip_gap 需超过阈值才 >0。")
