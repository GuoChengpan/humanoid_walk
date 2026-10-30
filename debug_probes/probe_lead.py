"""定位 lead_credit 不触发的真实原因。

奖励链：swing_forward(权重1.0) → 左脚摆到前面 → lead_credit(10.0) → 真交换 → stride_bonus(20.0)
逐环量：左脚落地时在前还是在后？body_advance 门控过不过？lead_credit 实际触发几次？
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

SEEDS = [7, 11, 23, 31]
PATH = "best_model/best_model.zip"      # 944K，本轮峰值

m = SAC.load(PATH, buffer_size=20000)
s = dict(L=0, ahead=0, behind=0, center=0, adv_pass=0, adv_fail=0,
         lead=0, lead_sum=0.0, swing=0.0, sb=0)
diffs, advs = [], []

for sd in SEEDS:
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    obs, _ = env.reset(seed=sd)
    prev_ll, prev_lr = 0.0, 0.0
    while True:
        a, _ = m.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        ll, lr = info["last_left_stride"], info["last_right_stride"]
        # 落地帧 = last_*_stride 刚被赋值（非落地帧沿用旧值，用变化检测太脆，改用符号+单调检测）
        if ll != prev_ll or lr != prev_lr:
            if ll > -1.0:      # 有有效落地记录
                if ll != prev_ll:                      # 左脚刚落地
                    s["L"] += 1
                    diffs.append(ll)                   # ll = left_x - right_x
                    adv = info["body_advance"]
                    advs.append(adv)
                    if ll < -0.05: s["ahead"] += 1     # 左脚在前
                    elif ll > 0.05: s["behind"] += 1
                    else: s["center"] += 1
                    if adv > 0.05: s["adv_pass"] += 1
                    else: s["adv_fail"] += 1
                if lr != prev_lr: s["sb"] += 1
        if abs(info["lead_credit"]) > 1e-9:
            s["lead"] += 1; s["lead_sum"] += info["lead_credit"]
        if abs(info["swing_forward_reward"]) > 1e-9:
            s["swing"] += info["swing_forward_reward"]
        prev_ll, prev_lr = ll, lr
        if te or tr: break
    env.close()

m.replay_buffer = None; del m; gc.collect()
diffs = np.array(diffs); advs = np.array(advs)

print("=" * 74)
print("模型 %s  (%d 次左脚落地 / 4 集)" % (PATH, s["L"]))
print("=" * 74)
print("\n【左脚落地时 left_x - right_x（>0 左脚在前）】")
print("  左脚在前 %4d (%5.1f%%)   右脚在前 %4d (%5.1f%%)   居中 %4d (%5.1f%%)" % (
    s["ahead"], 100*s["ahead"]/s["L"], s["behind"], 100*s["behind"]/s["L"],
    s["center"], 100*s["center"]/s["L"]))
print("  数值：中位 %+.4f m   p10 %+.4f   p90 %+.4f" % (
    np.median(diffs), np.percentile(diffs, 10), np.percentile(diffs, 90)))
print("  lead_credit 需要 (right_x - left_x) < 0 即 left_x > right_x + 0")
print("  → 满足条件的左脚落地占 %.1f%%" % (100*s["ahead"]/s["L"]))

print("\n【body_advance 门控（阈值 0.05m，跨上次落地的前进量）】")
print("  左脚落地时 body_advance：中位 %.4f m   p10 %.4f   p90 %.4f" % (
    np.median(advs), np.percentile(advs, 10), np.percentile(advs, 90)))
print("  通过 %d (%.1f%%)   未过 %d (%.1f%%)" % (
    s["adv_pass"], 100*s["adv_pass"]/s["L"], s["adv_fail"], 100*s["adv_fail"]/s["L"]))

print("\n【奖励链各环实际产出（4 集合计）】")
print("  swing_forward_reward  %6.1f/集   （左脚前摆的直接奖励，权重硬编码 1.0）" % (s["swing"]/4))
print("  lead_credit           %6.1f/集   （需左脚在前，4 集仅触发 %d 帧）" % (s["lead_sum"]/4, s["lead"]))
print("  stride_balance_penalty  见 tensorboard")

print("\n【断点判定】")
if s["ahead"] == 0:
    print("  ✗ 左脚落地时从未在前 → lead_credit 触发条件结构性不可达（鸡生蛋）")
elif 100*s["ahead"]/s["L"] < 5:
    print("  ✗ 只有 %.1f%% 的左脚落地在前 → lead_credit 几乎拿不到信号" % (100*s["ahead"]/s["L"]))
if s["adv_fail"] > 0.3 * s["L"]:
    print("  ✗ body_advance 门控挡掉了 %.1f%% 的左脚落地" % (100*s["adv_fail"]/s["L"]))
