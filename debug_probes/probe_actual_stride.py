"""区分两件事：
  A) 左脚自己的步幅 = 本次左脚落地 x − 上次左脚落地 x   （脚真走了多远）
  B) 落地时左右脚间距 = left_x − right_x                 （相对位置，顺序是否翻转）

wrapper 里的 last_left_stride 是 B 不是 A。如果 A 大但 B 一直为负，
说明左脚在走，只是永远追不上右脚 —— 那"加强 swing 位移奖励"就打错了靶。
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

CZ = 0.08
SEEDS = [7, 11, 23, 31]
PATH = "best_model/best_model.zip"

m = SAC.load(PATH, buffer_size=20000)
A, B = {"L": [], "R": []}, {"L": [], "R": []}
ahead_seq, flips = [], 0

for sd in SEEDS:
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=sd)
    d = env.unwrapped.data
    pl = pr = True
    prev_lx = prev_rx = None
    prev_ahead = None
    for _ in range(1500):
        a, _ = m.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        lx = d.body("left_foot").xpos[0]
        rx = d.body("right_foot").xpos[0]
        clc = d.body("left_foot").xpos[2] < CZ
        crc = d.body("right_foot").xpos[2] < CZ
        if clc and not pl:
            if prev_lx is not None: A["L"].append(lx - prev_lx)
            B["L"].append(lx - rx)
            prev_lx = lx
        if crc and not pr:
            if prev_rx is not None: A["R"].append(rx - prev_rx)
            B["R"].append(rx - lx)
            prev_rx = rx
        if clc or crc:
            ahead = lx > rx
            if prev_ahead is not None and ahead != prev_ahead:
                flips += 1
                if len(ahead_seq) < 30: ahead_seq.append("L" if ahead else "R")
            prev_ahead = ahead
        pl, pr = clc, crc
        if te or tr: break
    env.close()

m.replay_buffer = None; del m; gc.collect()
for k in "LR":
    A[k], B[k] = np.array(A[k]), np.array(B[k])

print("=" * 78)
print("模型 %s" % PATH)
print("=" * 78)
print("\n【A 脚自己的步幅：本次落地 x − 上次同侧落地 x】")
print("%-8s %8s %10s %10s %10s %10s"%("脚","次数","中位","p10","p90","|值|中位"))
for k, name in [("L", "左脚"), ("R", "右脚")]:
    v = A[k]
    print("%-8s %8d %10.4f %10.4f %10.4f %10.4f" % (name, len(v), np.median(v),
        np.percentile(v, 10), np.percentile(v, 90), np.median(np.abs(v))))

print("\n【B 落地时左右间距：left_x − right_x（左脚）/ right_x − left_x（右脚）】")
for k, name in [("L", "左脚落地"), ("R", "右脚落地")]:
    v = B[k]
    print("%-8s %8d %10.4f %10.4f %10.4f" % (name, len(v), np.median(v),
        np.percentile(v, 10), np.percentile(v, 90)))

print("\n【关键对照】左脚：真实步幅 %.3f m  vs  落地间距 %.4f m" % (
    np.median(A["L"]), np.median(B["L"])))
if np.median(A["L"]) > 0.05 and np.median(B["L"]) < 0.02:
    print("  → 左脚在走（步幅 %dcm），但落地时仍在右脚后面 %.1fcm" % (
        np.median(A["L"])*100, -np.median(B["L"])*100))
    print("  → 问题不是「左脚不摆」，是「顺序从不翻转」")
elif np.median(A["L"]) < 0.05:
    print("  → 左脚步幅只有 %.1fcm，确实没怎么摆" % (np.median(A["L"])*100))

print("\n【前后顺序是否翻转】")
print("  4 集共翻转 %d 次" % flips)
print("  前 30 次序列: %s" % "".join(ahead_seq))
