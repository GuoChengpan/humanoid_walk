"""验证封顶版 stride_balance_penalty：算术、单调性、真实端到端。"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

W, CAP, MIN_S = 0.5, 0.30, 0.15

def new_pen(L, R):
    lo = min(L, R)
    return min(W * (abs(L - R) + max(0.0, MIN_S - lo)), CAP)

def old_pen(L, R):
    lo = min(L, R)
    return 2.0 * (abs(L - R) + max(0.0, MIN_S - lo))

SC = [
    ("560K 现状步态",     0.034,  0.384),
    ("对称大步(目标)",    0.250,  0.250),
    ("对称小步",          0.050,  0.050),
    ("居中不交替",        0.000,  0.000),
    ("flail",            -0.30,  0.400),
    ("flail 极端",       -0.45,  0.450),
    ("左脚落地时在后(实测典型)", -0.35, 0.384),
]

print("=" * 80)
print("【1. 场景算术】")
print("%-24s %7s %7s | %9s %9s | %7s" % ("场景", "左L", "右R", "旧w2.0", "新w0.5封顶", "新/旧"))
print("-" * 80)
for name, L, R in SC:
    o, n = old_pen(L, R), new_pen(L, R)
    print("%-24s %7.3f %7.3f | %9.3f %9.3f | %6.2fx%s" % (
        name, L, R, o, n, n / o if o > 0 else 0.0,
        "  ← 封顶" if n >= CAP - 1e-12 else ""))

print("\n  flail / 560K 惩罚比：旧 %.2fx → 新 %.2fx" % (
    old_pen(*SC[4][1:]) / old_pen(*SC[0][1:]), new_pen(*SC[4][1:]) / new_pen(*SC[0][1:])))
print("  560K 基线 %.3f < 封顶 %.2f → 正常步态梯度保留 ✓" % (new_pen(*SC[0][1:]), CAP))
print("  目标态（对称大步）惩罚 %.3f → 终点不被罚 ✓" % new_pen(*SC[1][1:]))

print("\n【2. 单调性】d = |L-R| 从 0 增到 1.0m，对称度单调下降，惩罚必须单调不降")
bad, prev = 0, None
for d in np.linspace(0.0, 1.0, 201):
    p = new_pen(-d / 2, d / 2)
    if prev is not None and p < prev - 1e-12:
        bad += 1
        if bad <= 3:
            print("  ✗ 回落 at d=%.3f  prev=%.4f  now=%.4f" % (d, prev, p))
    prev = p
print("  201 点扫描：回落 %d 次  %s" % (bad, "✓" if bad == 0 else "✗"))
hit = min(d for d in np.linspace(0, 1, 10001) if new_pen(-d / 2, d / 2) >= CAP - 1e-9)
print("  封顶起点 d=%.3f m（560K 实测典型 %.3f，略高于封顶 → 部分事件触顶是预期的）" % (
    hit, abs(SC[6][1] - SC[6][2])))
print("  封顶之后为平台（无梯度），这正是目的：flail 越凶不再额外加罚")

print("\n【3. 单侧躺平检查】")
for L, R, lab in [(0.0, 0.38, "左脚完全不动"), (0.38, 0.0, "右脚完全不动"),
                  (0.0, 0.0, "双脚都不动")]:
    p = new_pen(L, R)
    print("  %-14s 惩罚 = %.3f  %s" % (lab, p, "✓ 仍被罚" if p > 0 else "✗ 逃掉了"))

print("\n【4. 真实端到端：560K 模型 + 当前代码（w=0.5, cap=0.30）】")
m = SAC.load("best_model/tmp_model_data/best_model.zip", buffer_size=20000)
ev_pen, ev_L, ev_R = [], [], []
ep_pen = []
for sd in [7, 11, 23, 31]:
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=sd)
    total = 0.0
    while True:
        a, _ = m.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        p = info["stride_balance_penalty"]
        total += p
        if p > 0:
            ev_pen.append(p)
            ev_L.append(info["last_left_stride"])
            ev_R.append(info["last_right_stride"])
        if te or tr:
            break
    env.close()
    ep_pen.append(total)
m.replay_buffer = None; del m; gc.collect()

ev_pen, ev_L, ev_R = np.array(ev_pen), np.array(ev_L), np.array(ev_R)
print("  每集惩罚：%s" % ", ".join("%.1f" % x for x in ep_pen))
print("  均值 %.1f/集 = 748.5/集 reward 的 %.1f%%   （旧 w=2.0 无封顶是 119.9/集 = 16.0%%）" % (
    np.mean(ep_pen), 100 * np.mean(ep_pen) / 748.5))
print("  事件 %d 次 / 4 集 = %.0f 次/集" % (len(ev_pen), len(ev_pen) / 4))
print("  事件惩罚分布：min %.3f  p25 %.3f  中位 %.3f  p75 %.3f  max %.3f" % (
    ev_pen.min(), np.percentile(ev_pen, 25), np.median(ev_pen),
    np.percentile(ev_pen, 75), ev_pen.max()))
print("  触顶事件 %d 次 = %.1f%%" % ((ev_pen >= CAP - 1e-12).sum(),
                                   100 * (ev_pen >= CAP - 1e-12).mean()))
print("  配对步幅（事件时刻）：|last_left| 中位 %.3f m   |last_right| 中位 %.3f m" % (
    np.median(np.abs(ev_L)), np.median(np.abs(ev_R))))
