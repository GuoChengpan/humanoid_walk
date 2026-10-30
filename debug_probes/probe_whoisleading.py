"""直接量「谁在前」：左脚 x 是否真的越过右脚 x。

上一轮我量的「落地脚接触序列交替率 99.7%」不能回答这个——
双脚可以交替落地，但同一只脚永远在前方（RA→C→RA→C）。
本探针量纵向领先关系本身：每步 left_x - right_x 的符号翻转次数。
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

CZ, DL = 0.08, 0.05          # 死区 5cm：|diff| < 0.05 视为居中，不算领先
SEEDS = [7, 11, 23, 31, 42, 55]


def run(path, tag, det=True):
    m = SAC.load(path, buffer_size=20000)
    tot = dict(n=0, la=0, ra=0, c=0, flip=0, flip_l2r=0, flip_r2l=0, land_swap=0, land=0)
    seq_nc = []                    # 非居中状态的时序，前 40 个
    for sd in SEEDS:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        data = env.unwrapped.data
        last_nc = None          # 上一个非居中状态；RA→C→RA 不算交替，RA→C→LA 才算
        prev_l = prev_r = True
        n = 0
        while True:
            a, _ = m.predict(obs, deterministic=det)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            n += 1
            lx, rx = data.body("left_foot").xpos[0], data.body("right_foot").xpos[0]
            d = lx - rx
            state = "LA" if d > DL else ("RA" if d < -DL else "C")
            tot["n"] += 1
            tot["la" if state == "LA" else "ra" if state == "RA" else "c"] += 1
            if state != "C" and last_nc is not None and state != last_nc:
                tot["flip"] += 1
                if last_nc == "RA" and state == "LA":
                    tot["flip_r2l"] += 1
                if last_nc == "LA" and state == "RA":
                    tot["flip_l2r"] += 1
            if state != "C":
                if len(seq_nc) < 40:
                    seq_nc.append(state[0])
                last_nc = state
            # 落地瞬间的领先方（对应 wrapper 的 curr_position_state 语义）
            lc_ = data.body("left_foot").xpos[2] < CZ
            rc_ = data.body("right_foot").xpos[2] < CZ
            if (lc_ and not prev_l) or (rc_ and not prev_r):
                tot["land"] += 1
                if state == "LA":
                    tot["land_swap"] += 1
            prev_l, prev_r = lc_, rc_
            if te or tr:
                break
        env.close()
    m.replay_buffer = None
    del m
    gc.collect()

    t = tot
    print("\n### %s  (%d 集, det=%s, 共 %d 步)" % (tag, len(SEEDS), det, t["n"]))
    print("  领先方分布（|diff|>%.0fcm 才算领先）:" % (DL * 100))
    print("    左脚在前 LA: %6d (%.1f%%)" % (t["la"], 100 * t["la"] / t["n"]))
    print("    右脚在前 RA: %6d (%.1f%%)" % (t["ra"], 100 * t["ra"] / t["n"]))
    print("    居中     C : %6d (%.1f%%)" % (t["c"], 100 * t["c"] / t["n"]))
    print("  领先方翻转（LA↔RA，跳过居中）: %d 次   →  每 %.0f 步一次" % (t["flip"], t["n"] / max(t["flip"], 1)))
    print("    RA→LA %d   LA→RA %d" % (t["flip_r2l"], t["flip_l2r"]))
    print("  落地瞬间左脚领先: %d/%d (%.1f%%)" % (t["land_swap"], t["land"],
                                              100 * t["land_swap"] / max(t["land"], 1)))
    print("  领先方时序(前40个非居中状态, L=左前 R=右前): %s" % "".join(seq_nc))
    return 100 * t["flip"] / max(t["n"], 1), 100 * t["la"] / t["n"]


print("判定线：若「右脚在前」≈100%% 且翻转≈0，则确认为「同一只脚一直在前、没有交替」。")
run("best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip", "576K 基线")
run("best_model/best_model.zip", "当前 best 560K")
run("best_model/best_model.zip", "当前 best 560K（随机策略）", det=False)
