#!/usr/bin/env python3
# phase_reward activity 改 min 的验证：单位性质 + 端到端实测
import os, sys, gc
sys.path.insert(0, '/home/guo/workspace/humanoid_walk')
os.chdir('/home/guo/workspace/humanoid_walk')
import numpy as np, gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

env = WalkRewardWrapper(gym.make('Humanoid-v4'))
obs, _ = env.reset(seed=0)

def f(std_l, std_r, corr=-0.4116):
    """复刻 compute_hip_phase_reward 的核心算式（给定 std，跳过 corr 重算）"""
    return (1 - corr) / 2.0 * np.clip(min(std_l, std_r) / 0.8, 0, 1)

def f_old(std_l, std_r, corr=-0.4116):
    return (1 - corr) / 2.0 * np.clip(((std_l + std_r) / 2) / 0.8, 0, 1)

print('====== 单位性质 ======')
# 1. 左右对称（交换左右不改变分值）
bad = 0
for sl in np.linspace(0.0, 0.9, 19):
    for sr in np.linspace(0.0, 0.9, 19):
        if abs(f(sl, sr) - f(sr, sl)) > 1e-12:
            bad += 1
print('1. 左右交换对称性: %d 次不一致 (应为 0)' % bad)

# 2. 单调性：提升弱腿 std，分值必须上升
rev = 0
for base in (0.05, 0.15, 0.30, 0.50):
    xs, ys = [], []
    for sl in np.linspace(0.0, 0.9, 45):
        xs.append(sl); ys.append(f(sl, base))
    for i in range(1, len(xs)):
        if xs[i] > xs[i-1] and ys[i] < ys[i-1] - 1e-15:
            rev += 1
print('2. 弱腿 std 单调递增: %d 次反转 (应为 0，160 点 × 4 基线)' % rev)

# 3. 弱腿为 0 时得分为 0（旧式不为 0）
print('3. 左腿消失 std=0: 新式 %.4f  旧式 %.4f  → 旧式仍有 %.0f%% 的分'
      % (f(0.0, 0.2952), f_old(0.0, 0.2952), 100*f(0.0,0.2952)/f_old(0.0,0.2952) if f_old(0.0,0.2952)>0 else 0))
print('   右腿消失 std=0: 新式 %.4f  旧式 %.4f' % (f(0.2952, 0.0), f_old(0.2952, 0.0)))

# 4. 饱和点
print('4. 饱和: std=%.2f 时新式 %.4f (应=anti_phase 上限)' % (0.8, f(0.8, 0.8)))

# 5. 当前工作点
print()
print('====== 当前工作点 (944K 实测 std 中位) ======')
cur_l, cur_r = 0.0185, 0.2952
print('现状  std_l=%.4f std_r=%.4f  → 每帧 %.4f  × 1451 帧 ≈ %.1f/集' % (cur_l, cur_r, f(cur_l,cur_r), f(cur_l,cur_r)*1451))
print('旧式  同点                              → 每帧 %.4f  × 1451 帧 ≈ %.1f/集' % (f_old(cur_l,cur_r), f_old(cur_l,cur_r)*1451))
print('左追平 std_l=%.4f                    → 每帧 %.4f  × 1451 帧 ≈ %.1f/集' % (cur_r, f(cur_r,cur_r), f(cur_r,cur_r)*1451))

# 6. 梯度对比
dl = 1e-4
g_new = (f(cur_r - 1e-6, cur_r) - f(0.0, cur_r)) / cur_r   # 从 0 到追平的平均斜率
g_old = (f_old(cur_r, cur_r) - f_old(0.0, cur_r)) / cur_r
print()
print('====== 梯度（左腿从 0 摆到追平右腿，每单位 std 增益 /集） ======')
print('新式 每单位 std: %.1f/集   全段共 %.1f/集' % (g_new*1451, (f(cur_r,cur_r)-f(0.0,cur_r))*1451))
print('旧式 每单位 std: %.1f/集   全段共 %.1f/集' % (g_old*1451, (f_old(cur_r,cur_r)-f_old(0.0,cur_r))*1451))

# 7. 端到端：用真实模型跑一集，确认 phase_reward 真掉下来了
print()
print('====== 端到端实测（944K 模型，seed 7，1500 步） ======')
env.close(); del env; gc.collect()
model = SAC.load('best_model/best_model.zip', buffer_size=20000)
env = WalkRewardWrapper(gym.make('Humanoid-v4', max_episode_steps=1500))
obs, _ = env.reset(seed=7)
tot = 0.0; n = 0
while True:
    a, _ = model.predict(obs, deterministic=True)
    obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
    tot += info['phase_reward']; n += 1
    if te or tr: break
print('phase_reward 总量 %.1f/集  (%d 帧)   ← 旧值 100.6/集' % (tot, n))
print('占该集 eval 前 100 帧平均 reward 的比例 %.1f%%' % (100*tot/(n*0.30)) if n else '')
env.close()
model.replay_buffer = None; del model; gc.collect()
