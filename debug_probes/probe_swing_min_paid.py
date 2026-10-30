"""核对 ③A(swing_min) 在确定性 rollout 里到底有没有发钱——det 汇总只有 2 分/集，
而训练 TB 报 ~17 分/集。要么是我记错了口径，要么这条通道在 det 下几乎恒为 0（=白装）。

看三件事：
 1) 每次落地的 min(rom_L, rom_R)° 分布，以及 ≥floor(4°)/≥top(14°) 的占比
 2) swing_min_reward 本体 vs 门控后 的每集合计
 3) 左腿是否**根本没有 rom**（累积窗复位时序问题），还是 rom 有但都 <4°
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk/debug_probes")

import price_batch4 as P

P.PER_STEP += ["swing_min_reward", "last_rom_L_deg", "last_rom_R_deg",
               "ev_reach_L_mean", "ev_reach_R_mean"]

for name, path in P.CANDS:
    for seed in P.SEEDS:
        ps, lands, last, n = P.rollout(path, seed)
        rl = np.array(ps["last_rom_L_deg"])
        rr = np.array(ps["last_rom_R_deg"])
        mn = np.minimum(rl, rr)
        raw = np.array(ps["swing_min_reward"])
        print("%s seed%-6d | 每步 rom_L p50/p90/max %.1f/%.1f/%.1f°  rom_R %.1f/%.1f/%.1f°" % (
            name, seed, np.percentile(rl, 50), np.percentile(rl, 90), rl.max(),
            np.percentile(rr, 50), np.percentile(rr, 90), rr.max()))
        print("   min(rom) %.1f° | 非零步占比 %.0f%% | ≥4°(起分) %.0f%% | ≥14°(满) %.0f%%" % (
            np.median(mn), 100 * (mn > 0.01).mean(), 100 * (mn >= 4).mean(), 100 * (mn >= 14).mean()))
        print("   swing_min 每集合计：本体 %.1f → 门控后 %.1f 分/集 | 落地数 %d" % (
            raw.sum(), np.array(ps["swing_min_reward_gated"]).sum(), len(lands)))
        print("   ev_reach_L %.3f R %.3f | 高度%.0f 生存%.0f\n" % (
            last["ev_reach_L_mean"], last["ev_reach_R_mean"],
            sum(ps["height_reward_gated"]), sum(ps["survival_bonus_gated"])))
