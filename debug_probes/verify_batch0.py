"""批次0+B 验证：修完接触判据、加滑移惩罚、松绑速度后，旧策略（944K 弹跳者）在新奖励下的收支。

对照基准：同一 seed 下改前实测 episode 回报 = 1887.6（旧判据 + flight 0.5 但只检出 4% +
超速罚 2.0×max(0,v-1.4) + 无滑移罚）。这里量改后，并把每一项摊开，看新激励是否把"弹跳"
打成亏本生意、同时没有把"蹭地爬行"变成更优解。
"""
import sys
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry
from probe_left_leg_chain import MODEL

SEED = 123
HORIZON = 1500
PRE_FIX_RETURN = 1887.6      # verify_contact_fix.py 改前实测（旧判据、旧权重）

PLUS = ["forward_reward", "height_reward", "net_balance", "phase_reward", "swing_forward_reward",
        "swing_drive_reward", "stride_bonus", "position_swap_reward", "contact_phase_reward",
        "last_contact_phase_reward"]
MINUS = ["support_sym_penalty", "hip_xz_angle_penalty", "lateral_penalty", "ctrl_cost",
         "stride_balance_penalty", "negative_stride_penalty", "short_stride_penalty",
         "stall_penalty", "alternate_penalty", "flight_phase_penalty", "slip_penalty"]


def main():
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(MODEL, env=env, device="cpu")
    model.replay_buffer = None

    obs, _ = env.reset(seed=SEED)
    acc = {k: 0.0 for k in PLUS + MINUS}
    seen = set()
    tot = 0.0
    n = 0
    speeds, slips_l, slips_r = [], [], []
    fell = False
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(act)
        tot += float(rew)
        n += 1
        seen |= set(info.keys())
        for k in PLUS + MINUS:
            acc[k] += float(info.get(k, 0.0))
        speeds.append(info.get("speed", 0.0))
        slips_l.append(info.get("slip_speed_L", 0.0))
        slips_r.append(info.get("slip_speed_R", 0.0))
        if term:
            fell = True
            break

    print(f"改后 episode 回报 = {tot:.0f} / {n} 步   (改前实测 {PRE_FIX_RETURN:.0f}) → 差 {tot-PRE_FIX_RETURN:+.0f}")
    print(f"是否摔倒: {fell}\n")
    print("%-26s %12s %12s" % ("项", "每步", "每集(×1500)"))
    known = 0.0
    for k in PLUS:
        if k not in seen:
            continue
        v = acc[k] / n
        if abs(v) < 1e-9:
            continue
        known += v * 1500
        print("%-26s %+12.4f %+12.0f" % (k, v, v * 1500))
    for k in MINUS:
        if k not in seen:
            continue
        v = acc[k] / n
        if abs(v) < 1e-9:
            continue
        known -= v * 1500
        print("%-26s %+12.4f %+12.0f" % ("-" + k, -v, -v * 1500))
    residual = tot - known
    print("%-26s %+12.4f %+12.0f   ← survival(0.5×门) + 未记录项 + 舍入" % (
        "未记账残差", residual / 1500, residual))

    print("\n相位（真接触，50 步滑窗末值）: 双腾空=%.3f 双支撑=%.3f 左支撑=%.3f 右支撑=%.3f" % (
        env.flight_frac, env.double_support_frac, env.stance_frac_L, env.stance_frac_R))
    sl, sr = np.array(slips_l), np.array(slips_r)
    print("滑移速度(接触帧才计): 左 mean=%.2f p50=%.2f | 右 mean=%.2f p50=%.2f m/s" % (
        sl[sl > 0].mean() if (sl > 0).any() else 0, np.median(sl[sl > 0]) if (sl > 0).any() else 0,
        sr[sr > 0].mean() if (sr > 0).any() else 0, np.median(sr[sr > 0]) if (sr > 0).any() else 0))
    sp = np.array(speeds)
    print("speed: mean=%.2f p90=%.2f >1.6占比=%.0f%%" % (sp.mean(), np.percentile(sp, 90), 100*np.mean(sp > 1.6)))
    print("滑移项成本 = %.0f/集  腾空项 = %.0f/集" % (
        -acc["slip_penalty"] / n * 1500, -acc["flight_phase_penalty"] / n * 1500))
    env.close()


if __name__ == "__main__":
    main()
