"""为什么 `stride_bonus` 一直是 0？把它的**三道门**逐个量出来（只读，两份权重对照）。

代码里的结算（train_humanoid_sac.py:1505-1519）等价于：
  左落地时  LS = left_x − right_x      右落地时  RS = right_x − left_x     （正值 = 落到对方前面）
  lo = min(LS, RS)   ← 用的是"最近一次左落地"与"最近一次右落地"两个**不同时刻**的快照
  条件A  last_landing_swapped == True （本次落地必须真的交换了前后关系）
  条件B  lo ≥ STRIDE_FLOOR = 0.10 m
  条件C  之后还要 ×walk_gate ×_ds_factor(ds≥0.05 才拿满) × 时长折算(现为 1)
本脚本把 LS/RS/min(LS,RS)/swapped 的分布全部量出来，回答：到底是哪道门在拦，以及"min"这个写法是不是结构性不可满足。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/"
POL = [("⑧ 弹跳 672K", ROOT + "best_model/tmp_model_data/SAC_18/18_加重腾空惩罚/best_model_snapshots/best_model_672000.zip"),
       ("⑨ 碎步 1696K", ROOT + "best_model/tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip")]
HORIZON, SEEDS = 1500, (123, 2026)
FLOOR, TOP, W = 0.10, 0.32, 35.0


def run(path, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    w = env
    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    prev = {"L": True, "R": True}
    rows = []
    for _ in range(HORIZON):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        cL, cR = bool(w.left_contact_prev), bool(w.right_contact_prev)
        for side, c in (("L", cL), ("R", cR)):
            if c and not prev[side]:
                ls = float(info.get("last_left_stride", 0.0))
                rs = float(info.get("last_right_stride", 0.0))
                rows.append(dict(side=side, ls=ls, rs=rs, both=(ls != 0.0 and rs != 0.0),
                                 swap=bool(w.last_landing_swapped),
                                 sb=float(info.get("stride_bonus", 0.0)),
                                 ds=float(info.get("double_support_frac", np.nan))))
        prev = {"L": cL, "R": cR}
        if term or trunc:
            break
    del model, env, raw
    return rows


def main():
    for name, path in POL:
        rows = []
        for s in SEEDS:
            rows += run(path, s)
        both = [r for r in rows if r["both"]]   # 前两次落地时对面那只脚还没有记录，剔除
        lo = np.array([min(r["ls"], r["rs"]) for r in both])
        ls = np.array([r["ls"] for r in both])
        rs = np.array([r["rs"] for r in both])
        swap = np.array([r["swap"] for r in rows])
        sb = np.array([r["sb"] for r in rows])
        print("\n=== %s ｜落地事件 %d 次（两种子合计）===" % (name, len(rows)))
        print("  LS(左落地点相对右脚) p10 %+.3f p50 %+.3f p90 %+.3f m｜>0 占比 %.0f%%" % (
            np.percentile(ls, 10), np.median(ls), np.percentile(ls, 90), 100 * (ls > 0).mean()))
        print("  RS(右落地点相对左脚) p10 %+.3f p50 %+.3f p90 %+.3f m｜>0 占比 %.0f%%" % (
            np.percentile(rs, 10), np.median(rs), np.percentile(rs, 90), 100 * (rs > 0).mean()))
        print("  lo=min(LS,RS)      p50 %+.3f p90 %+.3f｜**≥0.10 m 的落地占比 %.1f%%**（条件B）" % (
            np.median(lo), np.percentile(lo, 90), 100 * (lo >= FLOOR).mean()))
        print("  真交换 swapped 为真的落地占比 %.1f%%（条件A）｜A且B 同时满足 %.1f%%" % (
            100 * swap.mean(), 100 * np.mean([1.0 if (r["swap"] and min(r["ls"], r["rs"]) >= FLOOR) else 0.0
                                               for r in both])))
        print("  实际发放 stride_bonus：>0 的落地 %.1f%%，均值 %.2f 分/落地 ⇒ %.0f 分/集" % (
            100 * (sb > 0).mean(), sb.mean(), sb.sum() / len(SEEDS)))
        # 若把 min 改成 max / 或改成只看当次落地的间距，各能拿多少（只算，不改代码）
        for tag, val in (("min(LS,RS)", lo), ("max(LS,RS)", np.array([max(r["ls"], r["rs"]) for r in both])),
                         ("当次落地自身间距 |LS| 或 |RS|", np.array([abs(r["ls"]) if r["side"] == "L" else abs(r["rs"]) for r in both]))):
            clip = np.clip((val - FLOOR) / (TOP - FLOOR), 0, 1)
            print("  假设口径换成 %-26s ⇒ 可发 %.0f 分/集（W=35，全发不经门控）" % (tag, (clip * W).sum() / len(SEEDS)))


if __name__ == "__main__":
    main()
