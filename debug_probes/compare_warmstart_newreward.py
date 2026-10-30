"""warm-start 候选对比：全部用**新奖励**打分，并按机制指标排序。

历史上那些 eval 分数（2622 / 1993 / 2057）是在各自 lineage 的旧奖励下测的，改判据+改奖励后不可比。
这里对每个候选跑固定种子的确定性 rollout，看两件事：
 1) 新奖励下的回报（决定预填充数据的质量和 critic 起点）
 2) 机制指标（reach 左/右、双腾空、双支撑、髋行程、抬腿、真单步前进、滑移）——决定它是不是
    "离目标更近"的起点，而不只是"更会拿旧分"。
只读：不改任何模型文件。
"""
import gc
import glob
import sys
import zipfile

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/best_model/"
CANDS = [
    ("608K 旧(上上轮)", ROOT + "tmp_model_data/SAC_18/18_3_min(左右步幅)/best_model.zip"),
    ("608K 本轮(ev1743)", ROOT + "best_model.zip"),
]
SEEDS = (123, 2026)
HORIZON = 1500

SUM_KEYS = ["reach_reward", "slip_penalty", "flight_phase_penalty", "hip_xz_angle_penalty",
            "support_sym_penalty", "stride_bonus", "forward_reward", "height_reward",
            "survival_bonus", "alternate_penalty", "position_swap_reward", "position_swap_gated",
            "forward_gauss", "stride_bonus_gated", "forward_gauss_gated"]
LAST_KEYS = ["flight_frac", "double_support_frac", "single_frac", "walk_gate", "ds_landing_rate",
             "stance_frac_L", "stance_frac_R", "speed", "pitch_angle"]
EV_KEYS = ["ev_reach_L_mean", "ev_reach_R_mean", "ev_lift_L_mean", "ev_lift_R_mean",
           "ev_rom_L_mean", "ev_rom_R_mean", "ev_step_adv_L_mean", "ev_step_adv_R_mean",
           "ev_slip_L_mean", "ev_slip_R_mean", "ev_knee_ext_L_mean", "ev_knee_ext_R_mean",
           "ev_spacing_L_mean", "ev_spacing_R_mean", "ev_spacing_L_negfrac", "ev_spacing_R_negfrac"]


def num_ts(path):
    """从 zip 的 data.json 里取 num_timesteps（SB3 存的是 numpy 标量的 JSON 文本）。"""
    try:
        import re
        with zipfile.ZipFile(path) as z:
            d = z.read("data").decode("utf-8", "ignore")
        m = re.search(r'"num_timesteps":\s*\\?"?(\d+)', d)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def rollout(path, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    sums = {k: 0.0 for k in SUM_KEYS}
    last = {}
    n = 0
    total = 0.0
    fell = False
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(act)
        n += 1
        total += float(rew)
        for k in SUM_KEYS:
            sums[k] += float(info.get(k, 0.0))
        for k in LAST_KEYS + EV_KEYS:
            if k in info:
                last[k] = float(info[k])
        if term:
            fell = True
            break
    out = dict(steps=n, fell=fell, sums=sums, last=last, ret=total)
    del model, env, raw
    gc.collect()
    return out


def main():
    print("候选（新奖励 det 1500 步 × %d 种子）\n" % len(SEEDS))
    hdr = ("%-22s %6s %4s %7s | %6s %6s | %5s %5s %5s %5s | %6s %6s %6s | %6s %6s" % (
        "model", "步数", "摔", "新回报", "reachL", "reachR", "腾空", "双撑", "单撑", "门控",
        "liftL", "romL", "romR", "spL", "spR"))
    print(hdr)
    print("-" * len(hdr))
    for name, path in CANDS:
        if not glob.glob(path):
            print(f"{name:22s} 文件不存在")
            continue
        rs = [rollout(path, s) for s in SEEDS]
        g = lambda r, k, d=0.0: r["last"].get(k, d)
        mean = lambda vals: float(np.mean(vals))
        print("%-22s %6.0f %4s %7.0f | %+6.3f %+6.3f | %5.2f %5.2f %5.2f %5.2f | %6.3f %6.1f %6.1f | %+6.3f %+6.3f" % (
            name, mean([r["steps"] for r in rs]),
            "".join("Y" if r["fell"] else "." for r in rs),
            mean([r["ret"] for r in rs]),
            mean([g(r, "ev_reach_L_mean") for r in rs]),
            mean([g(r, "ev_reach_R_mean") for r in rs]),
            mean([g(r, "flight_frac") for r in rs]),
            mean([g(r, "double_support_frac") for r in rs]),
            mean([g(r, "single_frac") for r in rs]),
            mean([g(r, "walk_gate") for r in rs]),
            mean([g(r, "ev_lift_L_mean") for r in rs]),
            mean([g(r, "ev_rom_L_mean") for r in rs]),
            mean([g(r, "ev_rom_R_mean") for r in rs]),
            mean([g(r, "ev_spacing_L_mean") for r in rs]),
            mean([g(r, "ev_spacing_R_mean") for r in rs])))
        # 第二行：逐项收支（按新奖励的记账口径）+ 门控毛收入池 + 套利量
        r0 = rs[0]
        s, l = r0["sums"], r0["last"]
        pool = s["forward_gauss"] + s["stride_bonus"] + s["position_swap_reward"]
        print("%-22s   收支: reach奖 %+.0f | 滑罚 %+.0f | 飞罚 %+.0f | 髋税 %+.0f | 对称 %+.0f | "
              "步奖 %+.0f | fwd %+.0f | 高度 %+.0f | 生存 %+.0f | **交替奖 %+.0f** | 门控毛池 %+.0f" % (
                  "", s["reach_reward"], -s["slip_penalty"], -s["flight_phase_penalty"],
                  -s["hip_xz_angle_penalty"], -s["support_sym_penalty"], s["stride_bonus"],
                  s["forward_reward"], s["height_reward"], s["survival_bonus"],
                  -s["alternate_penalty"], pool))
        print("%-22s   套利/机制: 落地间距 spL %+.3f(负占%.0f%%) spR %+.3f(负占%.0f%%) | pitch %+.4f | "
              "双撑期落地占 %.3f | slipL %.3f kneeExtL %.1f kneeExtR %.1f" % (
                  "", l.get("ev_spacing_L_mean", float("nan")),
                  100 * l.get("ev_spacing_L_negfrac", float("nan")),
                  l.get("ev_spacing_R_mean", float("nan")),
                  100 * l.get("ev_spacing_R_negfrac", float("nan")),
                  l.get("pitch_angle", float("nan")), l.get("ds_landing_rate", float("nan")),
                  l.get("ev_slip_L_mean", float("nan")), l.get("ev_knee_ext_L_mean", float("nan")),
                  l.get("ev_knee_ext_R_mean", float("nan"))))
        print("%-22s   zip内 num_timesteps=%d  两种子(回报,步数) %s" % (
            "", num_ts(path), [(round(r["ret"]), r["steps"], "摔" if r["fell"] else "满")
                               for r in rs]))


if __name__ == "__main__":
    main()
