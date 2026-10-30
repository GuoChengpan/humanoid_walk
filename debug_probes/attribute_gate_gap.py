"""把 608K 在三种门控口径下的回报差算清楚，用来归因文档基线 [2052,2154] 与今天 1815/1878 的 −200。

原理（精确，不是近似）：`walk_gate` 在 reward 求和里只乘三项
    forward_gauss(已含 forward_gate) / stride_bonus / position_swap
所以同一条轨迹上，任意门控口径的回报 = 实际回报 + (新门 − 该口径门) × 三项毛额。
info 里恰好给出未乘 walk_gate 的三项原值 → 一次 rollout 同时算出三种口径，
且三者动作序列完全相同（det），差值就是纯记账差，不含任何轨迹噪声。
只读：不改代码、不写模型。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODEL = os.environ.get("PROBE_MODEL",
                       "/home/guo/workspace/humanoid_walk/best_model/tmp_model_data/"
                       "SAC_18/18_3_min(左右步幅)/best_model.zip")
SEEDS = (123, 2026)
HORIZON = 1500
POOL = ["forward_gauss", "stride_bonus", "position_swap_reward"]
TERMS = POOL + ["reach_reward", "height_reward", "survival_bonus", "slip_penalty",
                "flight_phase_penalty", "hip_xz_angle_penalty", "support_sym_penalty",
                "alternate_penalty", "stride_balance_penalty", "negative_stride_penalty",
                "lateral_penalty", "ctrl_cost", "phase_reward", "net_balance",
                "swing_forward_reward", "contact_phase_reward", "forward_gate", "speed_penalty",
                "single_frac", "double_support_frac", "flight_frac", "walk_gate"]


def one(path, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    acc = {k: 0.0 for k in TERMS}
    ret_new = ret_old = ret_none = 0.0
    n = 0
    fell = False
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(act)
        n += 1
        g = float(info["walk_gate"])
        dsf = float(info["double_support_frac"])
        single = float(info["single_frac"])
        # 滑窗前 20 步 wrapper 不更新门控（保持初值 1.0），三种口径要同样处理才可比
        warm = (single == 0.0 and dsf == 0.0 and float(info["flight_frac"]) == 0.0)
        pool = sum(float(info[k]) for k in POOL)
        old = 1.0 if warm else 0.35 + 0.65 * min(1.0, dsf / 0.15)              # R''#1 原式
        new = 1.0 if warm else 0.35 + 0.65 * np.clip((single - 0.50) / 0.25, 0.0, 1.0)  # 批次②
        assert warm or abs(new - g) < 1e-9, (new, g)
        ret_new += float(rew)
        ret_old += float(rew) + (old - new) * pool
        ret_none += float(rew) + (1.0 - new) * pool
        for k in TERMS:
            if k in info:
                acc[k] += float(info[k])
        if term:
            fell = True
            break
    del model, env, raw
    acc["_steps"], acc["_fell"] = n, fell
    return acc, (ret_new, ret_old, ret_none)


def main():
    print(f"模型={MODEL}\n")
    tot = []
    for sd in SEEDS:
        acc, rets = one(MODEL, sd)
        tot.append(rets)
        print(f"seed {sd}: 步数={acc['_steps']} 摔={'是' if acc['_fell'] else '否'} | "
              f"回报  新single门={rets[0]:7.0f}  旧ds门={rets[1]:7.0f}  无门控={rets[2]:7.0f}"
              f"   (门控代价 新={rets[2]-rets[0]:+.0f} 旧={rets[2]-rets[1]:+.0f})")
        print(f"          门控毛池={sum(acc[k] for k in POOL):+.0f}"
              f" [gauss {acc['forward_gauss']:+.0f} / stride {acc['stride_bonus']:+.0f} / "
              f"swap {acc['position_swap_reward']:+.0f}]  "
              f"mean single={acc['single_frac']/acc['_steps']:.3f} "
              f"ds={acc['double_support_frac']/acc['_steps']:.3f}")
    m = lambda i: float(np.mean([t[i] for t in tot]))
    print(f"\n两种子均值：新门控 {m(0):.0f} | 旧ds门控 {m(1):.0f} | 无门控 {m(2):.0f}")
    print(f"文档记录：608K 开跑前 [2052,2154]（均值 2103）；今天 compare 脚本 1815/1878（均值 {m(0):.0f}）")
    print(f"→ 无门控口径 − 文档基线 = {m(2)-2103:+.0f}；旧ds门控 − 文档基线 = {m(1)-2103:+.0f}")
    print("\n逐项收支（seed123，用于和文档 §6 的项级记录对齐）：")
    acc1, _ = one(MODEL, SEEDS[0])
    st = acc1["_steps"]
    for k in ["height_reward", "survival_bonus", "reach_reward", "alternate_penalty",
              "forward_gauss", "stride_bonus", "position_swap_reward", "swing_forward_reward",
              "contact_phase_reward", "phase_reward", "net_balance", "slip_penalty",
              "flight_phase_penalty", "hip_xz_angle_penalty", "support_sym_penalty",
              "stride_balance_penalty", "negative_stride_penalty", "lateral_penalty",
              "ctrl_cost", "speed_penalty", "forward_gate"]:
        v = acc1[k]
        if k == "forward_gate":
            print(f"  {k:24s} 均 {v/st:.3f}")
        else:
            print(f"  {k:24s} {v:+9.1f}")


if __name__ == "__main__":
    main()
