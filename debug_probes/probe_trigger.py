"""回答两个问题：
1) 步幅按"领先脚 - 落后脚"定义时，左右脚对称吗？（落地时刻的领先量分布）
2) 交替类奖励（alternate_penalty / swap_reward / lead_credit / stride_bonus）
   是"权重高但不触发"吗？给出触发率、单次幅度，和 dense stack 的真实比例。

复现 wrapper 自己的判定口径：
- diff = (right_x - left_x) / 0.18
- state: left_x > right_x + 0.05 → left_ahead ; right_x > left_x + 0.05 → right_ahead ; 否则 centered
- _check_cycle: prev_diff * curr_diff，|diff|<0.01 短路不更新 prev_diff，仅在触地上升沿调用
- stride_bonus 硬门控：left_stride > 0.2 且本跳 last_landing_swapped 为 True
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
THRESH = 0.05

# info key → 在总 reward 里的符号
TERMS = [("forward_reward", "+", "forward_reward"),
         ("height_reward", "+", "height_reward"),
         ("phase_reward", "+", "phase_reward"),
         ("alternate_penalty", "-", "alternate_penalty"),
         ("position_swap_reward", "+", "position_swap_reward"),
         ("lead_credit", "+", "lead_credit"),
         ("stride_bonus", "+", "stride_bonus"),
         ("stall_penalty", "-", "stall_penalty"),
         ("hip_xz_angle_penalty", "-", "hip_xz_angle_penalty"),
         ("support_sym_penalty", "-", "support_sym_penalty"),
         ("lateral_penalty", "-", "lateral_penalty"),
         ("negative_stride_penalty", "-", "negative_stride_penalty"),
         ("short_stride_penalty", "-", "short_stride_penalty")]


def run(path, tag):
    m = SAC.load(path, buffer_size=20000)
    print("\n### %s  (num_timesteps=%s, %d 集 x 1500 步, deterministic)"
          % (tag, m.num_timesteps, len(SEEDS)))

    tot_land = tot_L = tot_C = tot_R = 0
    alt_reward_ev = alt_pen_ev = 0
    prod_reward, prod_penalty = [], []
    lead_mag_L, lead_mag_R = [], []
    sums = {k: 0.0 for k, _, _ in TERMS}
    rew_tot = 0.0
    n_ep_done = 0
    stride_events = stride_bonus_events = 0

    for sd in SEEDS:
        env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
        obs, _ = env.reset(seed=sd)
        data = env.unwrapped.data
        prev_l = prev_r = True
        prev_diff = None

        while True:
            a, _ = m.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            rew_tot += float(r)
            for k, sign, _ in TERMS:
                sums[k] += float(info.get(k, 0.0)) * (1.0 if sign == "+" else -1.0)

            lx, rx = data.body("left_foot").xpos[0], data.body("right_foot").xpos[0]
            lc_ = data.body("left_foot").xpos[2] < CZ
            rc_ = data.body("right_foot").xpos[2] < CZ

            if (lc_ and not prev_l) or (rc_ and not prev_r):
                tot_land += 1
                diff = (rx - lx) / SCALE
                if abs(diff) < 0.01:
                    pass
                elif prev_diff is None:
                    prev_diff = diff
                else:
                    prod = prev_diff * diff
                    prev_diff = diff
                    if prod < 0:
                        alt_reward_ev += 1
                        prod_reward.append(abs(prod))
                    else:
                        alt_pen_ev += 1
                        prod_penalty.append(abs(prod))
                if lx > rx + THRESH:
                    tot_L += 1
                    lead_mag_L.append(lx - rx)
                elif rx > lx + THRESH:
                    tot_R += 1
                    lead_mag_R.append(rx - lx)
                else:
                    tot_C += 1
                # 本步落地是否被 swap 判定为"真交换"（wrapper 自己的判定，落地帧结算）
                if lc_ and not prev_l:
                    pass
            prev_l, prev_r = lc_, rc_
            if te or tr:
                break
        stride_events += int(env.stride_event_count)
        stride_bonus_events += int(env.stride_bonus_event_count)
        # 本集里有多少次左脚落地时 last_landing_swapped 为 False（落地帧即时读）
        n_ep_done += 1
        env.close()
    m.replay_buffer = None
    del m
    gc.collect()
    N = n_ep_done
    LA = np.array(lead_mag_L)
    RA = np.array(lead_mag_R)

    print("  reward/集   %9.1f" % (rew_tot / N))
    print("\n  [问题1] 落地时刻领先量（领先脚 - 落后脚）")
    print("  左脚在前  %4d (%5.1f%%)  中位 %.3f m  均值 %.3f m  最大 %.3f m"
          % (tot_L, 100 * tot_L / tot_land, np.median(LA), LA.mean(), LA.max()))
    print("  右脚在前  %4d (%5.1f%%)  中位 %.3f m  均值 %.3f m  最大 %.3f m"
          % (tot_R, 100 * tot_R / tot_land, np.median(RA), RA.mean(), RA.max()))
    print("  centered  %4d (%5.1f%%)  ← |间距|<=5cm，不算交换" % (tot_C, 100 * tot_C / tot_land))
    print("  右脚/左脚 领先量中位数比 = %.1f : 1" % (np.median(RA) / np.median(LA)))
    print("  若左脚也摆到右脚的中位领先量，交替 |product| 会从 %.3f 升到 %.3f (%.1fx)"
          % (np.mean(prod_reward), np.mean(prod_penalty),
             (np.mean(prod_penalty) / np.mean(prod_reward)) if prod_reward else float('nan')))

    print("\n  [问题2] 交替项触发（总落地 %d 次）" % tot_land)
    print("  交替奖励触发 %4d (%5.1f%%)   未交替惩罚触发 %4d (%5.1f%%)   无信息 %4d"
          % (alt_reward_ev, 100 * alt_reward_ev / tot_land,
             alt_pen_ev, 100 * alt_pen_ev / tot_land, tot_land - alt_reward_ev - alt_pen_ev))
    print("  交替事件   |product| 均值 %.3f  → 单次 %+.2f  (x2.3)"
          % (np.mean(prod_reward), -np.mean(prod_reward) * 2.3))
    print("  未交替事件 |product| 均值 %.3f  → 单次 %+.2f  (x2.3)"
          % (np.mean(prod_penalty), -np.mean(prod_penalty) * 2.3))
    print("  未交替事件里 |product|<0.3 的占比（小 product 来源=centered 带）: %d/%d"
          % ((np.array(prod_penalty) < 0.3).sum(), len(prod_penalty)))

    print("\n  reward 分解（每集，带符号；占 reward 比例）")
    tot = 0.0
    rows = []
    for k, sign, lab in TERMS:
        v = sums[k] / N
        rows.append((abs(v), lab, v))
        tot += v
    rows.sort(reverse=True)
    for _, lab, v in rows:
        print("    %-26s %10.1f/集  %7.1f%%" % (lab, v, 100 * v / (rew_tot / N)))
    print("    %-26s %10.1f   ← 分解项合计" % ("合计", tot / N))
    print("\n  左脚落地 %d 次，其中被 stride_bonus 硬门控放行 %d 次 (%.1f%%)"
          % (stride_events, stride_bonus_events,
             100 * stride_bonus_events / max(stride_events, 1)))
    return rew_tot / N


run(USER, "用户保存的 560K")
run(BASE, "576K 基线 w=2.3+门控0.5")
