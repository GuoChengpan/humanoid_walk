"""批次④ 的方向差判定：每个变体让"走路解比弹跳解多拿多少"变化了几点。

绝对值（起步价）由 price_batch4.py 实测；这里只补另一半——**走路反事实**。
④ 的正当性完全在方向差上：一个变体如果从弹跳手里拿走的钱 ≤ 它同时从走路手里拿走的钱，
它就是帮倒忙，不管它看着多"结构性"。

走路反事实的假设（逐条写明，可反驳）：
  · 几何：§16 实测 L=0.194 m/步、摆动 0.204 s ⇒ duty 0.50 时 v≈0.95（C=2L/v）
  · 相序：真交替 ⇒ single_frac≈0.76 ⇒ walk_gate=1.0、flat_gate=1.0（用真公式，不是估的）
  · 落地数：v/L × 22.5 s ≈ 110 次，两腿各 55
  · 落地间距 = 一个步长 ≈ 0.194 m ⇒ diff = 0.194/0.18 = 1.078，相邻两次符号相反
  · reach：左 −0.05（floor/top=−0.20/0.08 的斜坡值 0.536）、右 0.08 以上（饱和 1.0）——这一条是**假设**，
    所以额外给 reach_L=−0.10/0.00 的敏感性
  · 其余通道（姿态/相位/控制成本/各类税/负步幅/对称罚）按住弹跳实测值不动：它们与"弹跳还是走路"无关
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk/debug_probes")

from price_batch4 import (CANDS, HORIZON, PER_STEP, SEEDS, STEP_S, alt_income, gauss_raw,
                          rollout, APW, DS, FW, SG, TS)

WALK_V, WALK_L = 0.95, 0.194
WALK_SINGLE = 0.76
WALK_SPACING = 0.194
WALK_REACH = (-0.05, 0.08)
REACH_FLOOR, REACH_TOP = -0.20, 0.08
STRIDE_FLOOR, STRIDE_TOP, STRIDE_WEIGHT = 0.10, 0.32, 20.0
FG_REF = 0.15
SWING_MIN_W = 2.0


def ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def gates(single):
    g = 0.35 + 0.65 * ramp(single, 0.48, 0.62)
    return g, 0.6 + 0.4 * g


def walk_channels(n_steps=HORIZON):
    land = round(WALK_V / WALK_L * n_steps * STEP_S)
    wg, fg_ = gates(WALK_SINGLE)
    diff = WALK_SPACING / DS
    fg_gate = min(1.0, WALK_SPACING / FG_REF)          # min(左右间距)/0.15，走路=1.29→饱和
    per_land_stride = ramp(WALK_SPACING, STRIDE_FLOOR, STRIDE_TOP) * STRIDE_WEIGHT
    rl, rr = ramp(WALK_REACH[0], REACH_FLOOR, REACH_TOP), ramp(WALK_REACH[1], REACH_FLOOR, REACH_TOP)
    return dict(
        n=land, walk_gate=wg, flat_gate=fg_,
        forward_gauss_gated=FW * 1.0 * n_steps * fg_gate * wg,
        speed_penalty=0.0,
        stride_bonus_gated=per_land_stride * land * wg,
        alt_product=diff * diff * APW * land * wg,
        alt_min=diff * APW * land * wg,
        reach_w2=land / 2 * (2.0 * rl + 2.0 * rr),
        reach_w4=land / 2 * (4.0 * rl + 4.0 * rr),
        swing_min_gated=SWING_MIN_W * land * wg,
    )


def main():
    ps_list, lands_list, lasts = [], [], []
    for name, path in CANDS[:1]:                     # 448K：当前 best_model.zip，就是下轮的 warm-start
        for seed in SEEDS:
            ps, lands, last, n = rollout(path, seed)
            ps_list.append(ps); lands_list.append(lands); lasts.append(last)
    print("走路反事实 vs 448K 弹跳实测（det 1500 步 × %d 种子）\n" % len(SEEDS))

    hop = dict(n=HORIZON)
    wg_arr = np.mean([np.array(p["walk_gate"]) for p in ps_list], axis=0)
    hop["forward_gauss_gated"] = float(np.mean([sum(p["forward_gauss_gated"]) for p in ps_list]))
    hop["speed_penalty"] = float(np.mean([sum(p["speed_penalty"] if "speed_penalty" in p else [0] * len(p["reward"])) for p in ps_list]))
    hop["stride_bonus_gated"] = float(np.mean([sum(p["stride_bonus_gated"]) for p in ps_list]))
    hop["reach_w2"] = float(np.mean([sum(p["reach_reward"]) for p in ps_list]))
    hop["reach_w4"] = hop["reach_w2"] * 2
    hop["swing_min_gated"] = float(np.mean([sum(p["swing_min_reward_gated"]) for p in ps_list]))
    hop["height_gated"] = float(np.mean([sum(p["height_reward_gated"]) for p in ps_list]))
    hop["survival_gated"] = float(np.mean([sum(p["survival_bonus_gated"]) for p in ps_list]))
    hop["position_swap_gated"] = float(np.mean([sum(p["position_swap_gated"]) for p in ps_list]))
    alt_c = alt_min = 0.0
    for ps, lands in zip(ps_list, lands_list):
        c, m = alt_income(lands, np.array(ps["walk_gate"]), cap=1e9)
        alt_c += c; alt_min += m
    hop["alt_product"] = alt_c / len(ps_list)
    hop["alt_min"] = alt_min / len(ps_list)
    hop["total"] = float(np.mean([sum(p["reward"]) for p in ps_list]))
    named = ["forward_gauss_gated", "stride_bonus_gated", "alt_product", "reach_w2",
             "swing_min_gated", "height_gated", "survival_gated", "position_swap_gated"]
    hop["other"] = hop["total"] - sum(hop[x] for x in named) - (-hop["speed_penalty"])

    # 走路：与步态结构无关的项按住弹跳实测（other / 高度钱按 raw×flat_gate 重算）
    w = walk_channels()
    height_raw = hop["height_gated"] / float(np.mean([p["flat_gate"][200] for p in ps_list]))
    surv_raw = hop["survival_gated"] / float(np.mean([p["flat_gate"][200] for p in ps_list]))
    walk = dict(other=hop["other"], forward_gauss_gated=w["forward_gauss_gated"],
                speed_penalty=0.0, stride_bonus_gated=w["stride_bonus_gated"],
                alt_product=w["alt_product"], alt_min=w["alt_min"],
                reach_w2=w["reach_w2"], reach_w4=w["reach_w4"],
                swing_min_gated=w["swing_min_gated"],
                height_gated=height_raw * w["flat_gate"], survival_gated=surv_raw * w["flat_gate"],
                position_swap_gated=hop["position_swap_gated"] / float(np.mean(wg_arr[:200])) * w["walk_gate"])
    walk["total"] = (walk["other"] + walk["forward_gauss_gated"] + walk["stride_bonus_gated"]
                     + walk["alt_product"] + walk["reach_w2"] + walk["swing_min_gated"]
                     + walk["height_gated"] + walk["survival_gated"] + walk["position_swap_gated"])
    walk["total_min"] = walk["total"] - walk["alt_product"] + walk["alt_min"]

    print("通道(分/集)                 弹跳实测      走路反事实     走路−弹跳")
    for label, hk, wk in [("forward_gauss×gate", "forward_gauss_gated", "forward_gauss_gated"),
                          ("stride_bonus(min间距)", "stride_bonus_gated", "stride_bonus_gated"),
                          ("alternate(product)", "alt_product", "alt_product"),
                          ("alternate(sign×min)", "alt_min", "alt_min"),
                          ("reach(w=2)", "reach_w2", "reach_w2"),
                          ("reach(w=4)", "reach_w4", "reach_w4"),
                          ("swing_min(③A)", "swing_min_gated", "swing_min_gated"),
                          ("height×flat_gate", "height_gated", "height_gated"),
                          ("survival×flat_gate", "survival_gated", "survival_gated"),
                          ("position_swap×gate", "position_swap_gated", "position_swap_gated"),
                          ("其余(按住不动)", "other", "other")]:
        print("  %-26s %10.0f %14.0f %+11.0f" % (label, hop[hk], walk[wk], walk[wk] - hop[hk]))
    print("  %-26s %10.0f %14.0f %+11.0f   ← 现奖励下的总差价" % (
        "合计(现形式)", hop["total"], walk["total"], walk["total"] - hop["total"]))
    print("  %-26s %10.0f %14.0f %+11.0f   ← 只把 alternate 换成 min" % (
        "合计(alternate=min)", hop["total"] - hop["alt_product"] + hop["alt_min"],
        walk["total_min"], walk["total_min"] - (hop["total"] - hop["alt_product"] + hop["alt_min"])))
    print("\n走路反事实参数：落地%d次 walk_gate=%.2f flat_gate=%.2f | 间距%.3f m→stride_bonus %.0f" % (
        w["n"], w["walk_gate"], w["flat_gate"], WALK_SPACING, w["stride_bonus_gated"]))
    print("reach_L 敏感性（走路 reach_w2）：", ", ".join(
        "reach_L=%.2f→%.0f" % (x, w["n"] / 2 * (2.0 * ramp(x, REACH_FLOOR, REACH_TOP)
                                                 + 2.0 * ramp(WALK_REACH[1], REACH_FLOOR, REACH_TOP)))
        for x in (-0.15, -0.10, -0.05, 0.0)))


if __name__ == "__main__":
    main()
