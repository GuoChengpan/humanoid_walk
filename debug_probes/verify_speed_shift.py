"""批次②（速度峰值 1.3→0.95、σ 0.3→0.40）的验收探针：实测起步价 + 走路反事实表。

三件事：
 1) **起步价**：在当前策略（弹跳者）的同一批轨迹上，逐帧对比 旧gauss(peak1.3,σ0.3) 与
    新gauss(peak0.95,σ0.40)，再乘上实测的 forward_gate/walk_gate → 得到新代码下这笔钱的变化。
    同时报总回报的新旧差（其余项不受此改动影响，可精确重构）。
 2) **反飞行差价是否真的出现**：用实测的 L(单步长)、摆动时长、duty 构造"走路反事实"
    （duty=0.50/0.55/0.60 且步幅不变 ⇒ 对应速度），算它在新公式下拿多少 gauss×gate，与弹跳者比较。
    这一步是整批改动的**立论依据**，必须能复算。
 3) 顺带确认 speed_penalty 折点 1.6、移动门 0.3 都还在原位。
只读：不改模型、不碰训练进程。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

CANDS = [os.environ.get("PROBE_MODEL", "")] if os.environ.get("PROBE_MODEL") else sorted(
    glob.glob("/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/*.zip"),
    key=os.path.getmtime)[-2:]
DT = 0.015
HORIZON = 1500
SEEDS = (123, 2026)


def gauss(v, peak, sig, fw=0.4):
    return fw * np.exp(-((v - peak) ** 2) / (2 * sig ** 2))


def roll(path, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    acc = {k: [] for k in ("speed", "forward_gauss", "walk_gate", "forward_gate", "flat_gate",
                           "flight_frac", "duty_sum", "single_frac", "height_reward",
                           "survival_bonus", "stride_bonus", "position_swap_reward", "speed_penalty")}
    steps = []
    fell = False
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(act)
        for k in acc:
            if k in info:
                acc[k].append(float(info[k]))
        steps.append((float(info.get("ev_step_adv_L_mean", 0.0)), float(info.get("ev_step_adv_R_mean", 0.0))))
        if term:
            fell = True
            break
    del model, env, raw
    return {k: np.array(v) for k, v in acc.items()}, fell, len(acc["speed"]), np.array(steps)


def main():
    for path in CANDS:
        if not path or not os.path.exists(path):
            print(f"跳过不存在的候选: {path}")
            continue
        print(f"\n########## {os.path.basename(path)} ##########")
        rs = [roll(path, s) for s in SEEDS]
        d, fell, n, steps = rs[0]
        v, wg, fg, flag = d["speed"], d["walk_gate"], d["forward_gate"], d["flat_gate"]
        m = int(n * 0.7)
        new_raw = gauss(v, 0.95, 0.40)                 # forward_gauss 里 logged 的已含 forward_gate
        old_raw = gauss(v, 1.30, 0.30)
        # logged forward_gauss = 0.4·exp(...)·forward_gate；用同比例还原两种口径
        ratio = fg[m:].mean() if fg[m:].mean() > 0 else 1.0
        new_g = new_raw[m:] * fg[m:] * wg[m:]
        old_g = old_raw[m:] * fg[m:] * 0.446            # 上一轮代码里 gauss 也被 walk_gate 乘过
        # 上一轮的 walk_gate 用的是 ds 还是 single？本 run 已是 single 口径，直接沿用实测 wg
        old_g = old_raw[m:] * fg[m:] * wg[m:]
        print(f"[1] 起步价（seed{SEEDS[0]}，后70%帧，×1500=分/满集）")
        print(f"    v 均值={v[m:].mean():.3f}（>1.6 的帧占 {100*np.mean(v[m:]>1.6):.0f}%，超速税均值 {d['speed_penalty'][m:].mean():.3f}/步）")
        print(f"    gauss×gates: 旧公式 {old_g.mean()*1500:+.0f} → 新公式 {new_g.mean()*1500:+.0f}"
              f"  = **{new_g.mean()*1500-old_g.mean()*1500:+.0f} 分/集**")
        print(f"    实测 walk_gate={wg[m:].mean():.3f} flat_gate={flag[m:].mean():.3f} "
              f"single={d['single_frac'][m:].mean():.3f} flight={d['flight_frac'][m:].mean():.3f} "
              f"duty_sum={d['duty_sum'][m:].mean():.3f}   两种子摔={fell}/{rs[1][1]} 步数={rs[0][2]}/{rs[1][2]}")

        L = np.mean([x for pair in steps for x in pair if x > 0.02][:200]) or 0.217
        duty = float(d["duty_sum"][m:].mean()) / 2
        C = 2 * L / float(v[m:].mean())
        swing = C * (1 - duty)
        print(f"[2] 走路反事实（用实测 L={L:.3f}m、摆动时长={swing:.3f}s 保持不变的假设）")
        for dd in (0.50, 0.55, 0.60, 0.65):
            C2 = swing / (1 - dd)
            vv = 2 * L / C2
            # 完美错峰的走法：flight=0、ds=2·duty−1 ⇒ single = duty_sum − 2ds = 2 − 2·duty
            single2 = 2.0 - 2.0 * dd if dd >= 0.5 else 1.0
            gate2 = 0.35 + 0.65 * min(1.0, max(0.0, (single2 - 0.48) / 0.14))
            flat2 = 0.6 + 0.4 * gate2
            inc = gauss(vv, 0.95, 0.40) * ratio * gate2 * 1500
            hop_inc = new_g.mean() * 1500
            print(f"    duty={dd:.2f}: v={vv:.2f} single={single2:.2f} gate={gate2:.3f} "
                  f"gauss×gate={inc:+.0f} 分/集 | 平地(2108×{flat2:.3f})={2108*flat2:+.0f} | "
                  f"相对弹跳({hop_inc:+.0f}) 差价 {inc-hop_inc:+.0f}")
        print(f"[3] 折点/门控原位检查：speed_penalty 公式在 v>1.6 才生效 "
              f"(实测 max v={v.max():.2f}, 罚={d['speed_penalty'][m:].max():.2f}/步) | "
              f"height/survival 的移动门 v>0.3 未变")


if __name__ == "__main__":
    main()
