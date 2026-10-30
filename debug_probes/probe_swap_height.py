"""回答两个问题：
Q1) position_swap_reward 能否代替 |left_stride - right_stride| 对称惩罚？
    → 看它在左脚交换时能不能拿到分（阈值 18cm vs 左脚实际最大领先量）
    → 看它是否包含"左右步幅比较"信息（还是只看当次间距）
Q2) height_reward 是否还有"蹲着走"的限制？
    → 看实际躯干高度分布，是否被压在某个带里，是否有蹲踞模式
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

USER = "best_model/tmp_model_data/best_model.zip"
BASE = "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"
CZ, SCALE = 0.08, 0.18
SEEDS = [7, 11, 23, 31]
THRESH, SWAP_MIN, SWAP_SCALE = 0.05, 0.18, 0.22


def run(path, tag):
    m = SAC.load(path, buffer_size=20000)
    print("\n### %s  (num_timesteps=%s)" % (tag, m.num_timesteps))

    heights = []
    hr = 0.0
    steps = 0
    # Q1
    swap_vals = []          # 每次落地的 position_swap_reward
    swap_nonzero = 0
    gap_L = []              # 判为 left_ahead 落地时的 hip_x_gap
    gap_R = []
    gap_C = []
    swap_on_L = swap_on_R = 0
    swap_val_on_L, swap_val_on_R = [], []
    landings = 0

    for sd in SEEDS:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        data = env.unwrapped.data
        prev_l = prev_r = True
        prev_state = "centered"

        while True:
            a, _ = m.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            steps += 1
            h = float(data.qpos[2])
            heights.append(h)
            hr += float(info.get("height_reward", 0.0))

            lx, rx = data.body("left_foot").xpos[0], data.body("right_foot").xpos[0]
            lc_ = data.body("left_foot").xpos[2] < CZ
            rc_ = data.body("right_foot").xpos[2] < CZ
            if (lc_ and not prev_l) or (rc_ and not prev_r):
                landings += 1
                gap = abs(lx - rx)
                if lx > rx + THRESH:
                    st = "left_ahead"
                    gap_L.append(gap)
                elif rx > lx + THRESH:
                    st = "right_ahead"
                    gap_R.append(gap)
                else:
                    st = "centered"
                    gap_C.append(gap)
                sv = float(info.get("position_swap_reward", 0.0))
                swap_vals.append(sv)
                if sv > 0:
                    swap_nonzero += 1
                    if st == "left_ahead":
                        swap_on_L += 1
                        swap_val_on_L.append(sv)
                    elif st == "right_ahead":
                        swap_on_R += 1
                        swap_val_on_R.append(sv)
                prev_state = st
            prev_l, prev_r = lc_, rc_
            if te or tr:
                break
        env.close()
    m.replay_buffer = None
    del m
    gc.collect()
    N = len(SEEDS)
    H = np.array(heights)

    print("\n  [Q2] 躯干高度分布（%d 步）" % len(H))
    print("  mean %.4f  median %.4f  min %.4f  max %.4f  std %.4f"
          % (H.mean(), np.median(H), H.min(), H.max(), H.std()))
    for lo, hi, lab in [(0, 1.05, "<1.05 躺/摔"), (1.05, 1.25, "1.05-1.25 蹲"),
                        (1.25, 1.45, "1.25-1.45 平顶带"), (1.45, 1.55, "1.45-1.55 偏高"),
                        (1.55, 9.0, ">1.55 跳")]:
        p = 100 * np.mean((H >= lo) & (H < hi))
        print("    %-18s %6.2f%%" % (lab, p))
    print("  height_reward %9.1f/集   (上限 0.5/步 → 1500 步满分 750/集)" % (hr / N))
    print("  band 拿满的步数占比 %.1f%%"
          % (100 * np.mean((H >= 1.25) & (H <= 1.45))))
    print("  自然站高 1.393 落在平顶带 [1.25, 1.45] 内 → 蹲踞引力已移除（代码 701-707 行）")
    print("  低于 1.25 处 band 随高度线性上升（1.05→1.25），是向上的防躺梯度")

    print("\n  [Q1] position_swap_reward（共 %d 次落地）" % landings)
    print("  >0 的落地 %4d 次 (%.1f%%)   总 %.1f/集"
          % (swap_nonzero, 100 * swap_nonzero / max(landings, 1),
             sum(swap_vals) / N))
    print("  左脚交换落地 %d 次，其中 swap_reward>0 的 %d 次；这些落地 hip_x_gap 最大 %.4f m"
          % (len(gap_L), swap_on_L, max(gap_L) if gap_L else float('nan')))
    print("  右脚交换落地 %d 次，其中 swap_reward>0 的 %d 次；这些落地 hip_x_gap 中位 %.4f m"
          % (len(gap_R), swap_on_R, np.median(gap_R) if gap_R else float('nan')))
    print("  swap 生效阈值 hip_x_gap > %.2f m（step_factor 从 0 起算）" % SWAP_MIN)
    print("  左脚落地中 hip_x_gap > 0.18 的次数: %d / %d"
          % (int(np.sum(np.array(gap_L) > SWAP_MIN)), len(gap_L)))
    if swap_val_on_R:
        print("  右脚交换落地时 swap_reward 均值 %.2f（上限 %s）"
              % (np.mean(swap_val_on_R), 5.0))
    print("\n  swap_reward 是否含'左右步幅比较'信息：否。step_factor 只取当次 hip_x_gap，")
    print("  不读上一次落地的 gap，所以 (左脚30cm,右脚38cm) 和 (左脚30cm,右脚5cm) 拿同样的分")
    return hr / N


run(USER, "用户 560K")
run(BASE, "576K 基线")
