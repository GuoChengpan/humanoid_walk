#!/usr/bin/env python3
# 解析 SAC_17 最新一轮：reward 分解 + 交替改善 + 1-1.2M 崩溃定位
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

ea = EventAccumulator('sac_humanoid_walk/SAC_17')
ea.Reload()

def S(t):
    try:
        ev = ea.Scalars(t); return np.array([e.step for e in ev]), np.array([e.value for e in ev])
    except Exception:
        return np.array([]), np.array([])

print('='*104)
print('一、eval 轨迹（2000 步一次，5 集）')
print('='*104)
st, r = S('eval/mean_reward'); sl, l = S('eval/mean_ep_length')
mx = int(st.max())
B = 6
print('%-11s %-11s %-11s %-11s %-11s %-11s %-11s' % ('区间','reward均值','reward峰','reward谷','len均值','len谷','n'))
for i in range(B):
    lo, hi = int(mx*i/B), int(mx*(i+1)/B)
    m = (st >= lo) & (st < hi)
    if m.sum() == 0: continue
    print('%-11s %-11.1f %-11.1f %-11.1f %-11.0f %-11.0f %-11d' % (
        '%d-%dK' % (lo//1000, hi//1000), np.mean(r[m]), np.max(r[m]), np.min(r[m]),
        np.mean(l[m]), np.min(l[m]), m.sum()))
print('\n全部 eval: 峰值 %.1f @ %d   谷值 %.1f @ %d' % (
    r.max(), st[np.argmax(r)], r.min(), st[np.argmin(r)]))
print('短集(<100步)次数 %d / %d   len<200 次数 %d' % ((l < 100).sum(), len(l), (l < 200).sum()))

print()
print('='*104)
print('二、关键项走势（按 10 个等分桶取均值）')
print('='*104)
tags = [
    ('rollout/ep_rew_mean', 'rollout奖励'),
    ('rollout/ep_len_mean', 'rollout长度'),
    ('gait/height_reward', '高度奖励'),
    ('gait/forward_reward', '前进奖励'),
    ('gait/phase_reward', '相位奖励(新)'),
    ('gait/stride_bonus', '步幅奖励'),
    ('gait/lead_credit', '领先信用'),
    ('gait/swing_forward_reward', '摆动前向'),
    ('gait/hip_xz_angle_penalty', '髋xz惩罚'),
    ('gait/support_sym_penalty', '支撑对称惩罚'),
    ('gait/alternate_penalty', '无交替惩罚'),
    ('gait/lateral_penalty', '侧向惩罚'),
    ('gait/flight_phase_penalty', '腾空惩罚'),
    ('gait/stride_balance_penalty', '步幅对称惩罚'),
    ('gait/ctrl_cost', '控制成本'),
]
N = 10
# 用 rollout 的步数范围做桶
_, rv = S('rollout/ep_rew_mean'); rng = (int(rv.min()) if len(rv) else 0, int(st.max()) if len(st) else 0)
ss, _ = S('gait/height_reward')
lo0, hi0 = (int(ss.min()), int(ss.max())) if len(ss) else (0, 1)
hdr = '%-20s' % '桶'
for _, nm in tags: hdr += '%15s' % nm[:14]
print(hdr)
for i in range(N):
    lo, hi = int(lo0 + (hi0-lo0)*i/N), int(lo0 + (hi0-lo0)*(i+1)/N)
    row = '%-20s' % ('%d-%dK' % (lo//1000, hi//1000))
    for t, _ in tags:
        x, y = S(t)
        m = (x >= lo) & (x < hi)
        row += '%15.4f' % (np.mean(y[m]) if m.sum() else float('nan'))
    print(row)

print()
print('='*104)
print('三、交替是否改善（核心指标走势）')
print('='*104)
alt_tags = [
    ('gait/stride_bonus_rate', '步幅奖励触发率'),
    ('gait/alternate_reward_rate', '交替奖励率'),
    ('gait/alternate_penalty_rate', '交替惩罚率'),
    ('gait/position_swap_rate', '位置交换率'),
    ('gait/stall_max', '连续不交换'),
    ('gait/last_left_stride', '左脚落地间距'),
    ('gait/last_right_stride', '右脚落地间距'),
    ('gait/negative_stride_rate', '负步幅率'),
    ('gait/activity_difference', '活动差异'),
    ('gait/speed', '速度'),
]
hdr = '%-14s' % '桶'
for _, nm in alt_tags: hdr += '%16s' % nm[:15]
print(hdr)
for i in range(N):
    lo, hi = int(lo0 + (hi0-lo0)*i/N), int(lo0 + (hi0-lo0)*(i+1)/N)
    row = '%-14s' % ('%d-%dK' % (lo//1000, hi//1000))
    for t, _ in alt_tags:
        x, y = S(t)
        m = (x >= lo) & (x < hi)
        row += '%16.4f' % (np.mean(y[m]) if m.sum() else float('nan'))
    print(row)

print()
print('='*104)
print('四、训练健康度（1-1.2M 崩溃定位）')
print('='*104)
hdr = '%-14s %-12s %-12s %-12s %-12s %-12s %-12s %-12s' % ('桶','actor_loss','critic','expl_std','action_std','ent_coef','fps','lr')
print(hdr)
for i in range(N):
    lo, hi = int(lo0 + (hi0-lo0)*i/N), int(lo0 + (hi0-lo0)*(i+1)/N)
    row = '%-14s' % ('%d-%dK' % (lo//1000, hi//1000))
    for t in ('train/actor_loss','train/critic_loss','train/exploration_std','train/action_std','train/ent_coef','time/fps','train/learning_rate'):
        x, y = S(t)
        m = (x >= lo) & (x < hi)
        row += '%12.4f' % (np.mean(y[m]) if m.sum() else float('nan'))
    print(row)

print()
print('='*104)
print('五、eval 逐次明细（看是骤降还是缓降）')
print('='*104)
print('%-11s %-11s %-11s' % ('step','reward','len'))
for s, rv_, lv in zip(st, r, l):
    if s % 20000 == 0 or s == st[0] or s == st[-1]:
        print('%-11d %-11.1f %-11.0f' % (s, rv_, lv))
