"""读 eval_logs/evaluations.npz：每一列 = 一次 eval 的 5 个 episode。
这是 eval 序列的地面真值 —— tensorboard 的 gait/* 是单样本，不能这么读。
"""
import numpy as np

d = np.load('eval_logs/evaluations.npz')
t, r, l = d['timesteps'], d['results'], d['ep_lengths']
n, k = r.shape
print('keys:', list(d.keys()), '| 列数(eval 次数) =', n, '| 每列集数 =', k)
print('timesteps: %d -> %d (间距 %d = 8 环境 x eval_freq 2000)' % (t[0], t[-1], t[1] - t[0]))
print()

print('最近 12 次 eval 的逐集明细（rew | len）:')
for c in range(n - 12, n):
    rw = ' '.join('%7.1f' % x for x in r[c])
    ln = ' '.join('%7d' % x for x in l[c])
    nf = int((l[c] < 1490).sum())
    print('  %8d 步  %s' % (t[c], rw))
    print('  %10s  %s   摔倒=%d' % ('', ln, nf))
print()

# 验证 ep_len 聚类就是"5 集里摔倒几个"的量化
flat_l = np.round(l.ravel()).astype(int)
flat_r = r.ravel()
print('==== ep_len 聚类 → 摔倒个数 解释 ====')
for target in [1500, 1207, 912, 622]:
    band = np.abs(flat_l - target) < 25
    if band.sum() == 0:
        continue
    est = round((1500 * 5 - band.sum() * 0) / 5) if False else None
    full = int((flat_l[band] >= 1490).sum())
    print('  ~%d 步: %4d 集  (1500,1500,1500,1500,~35 应得 %.0f)' % (target, band.sum(), (4 * 1500 + 35) / 5))
print('  理论值: 0 摔=%.0f  1 摔=%.0f  2 摔=%.0f  3 摔=%.0f  (按摔倒集 ~35 步估算)'
      % (5 * 1500 / 5, (4 * 1500 + 35) / 5, (3 * 1500 + 2 * 35) / 5, (2 * 1500 + 3 * 35) / 5))
print()

print('==== 分段：摔倒率 vs 走满集质量 ====')
rows = []
for a, b in [(0, 20), (20, 40), (40, 60), (60, 81)]:
    sl = r[a:b].ravel()
    ll = l[a:b].ravel()
    full = ll >= 1490
    fall = ~full
    nm = len(sl)
    fm = sl[full].mean() if full.any() else float('nan')
    sm = sl[fall].mean() if fall.any() else float('nan')
    s_len = ll[fall].mean() if fall.any() else float('nan')
    rows.append((a, b, nm, 100 * fall.mean(), fm, sm, s_len, sl.mean()))
    print('  eval %2d-%2d  (%6d-%6d 步)  %3d 集  摔倒 %6.1f%% (%2d 集)  走满集 reward %7.1f  摔倒集 reward %7.1f (步数 %5.0f)  全体均值 %7.1f'
          % (a, b, t[a], t[b - 1], nm, 100 * fall.mean(), int(fall.sum()), fm, sm, s_len, sl.mean()))
print()
print('注：n_eval_episodes=5 → 摔倒率估计粒度 20%，单次 eval 一次摔倒就拖 ~98 分。')
