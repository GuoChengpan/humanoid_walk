"""逐落地账本：把 det rollout 的收入拆成"每次落地发多少"，用来算**步频补贴**。

为什么要它（2026-10-11 用户问："峰值降到 0.95 了，速度为什么总回到 1.2+"）：
`stride_bonus / alternate / reach / swing_min / swing_forward` 都是**落地事件结算**的，
所以"每秒落次越多 = 钱越多"。落次数 = 22.5s × v / L ⇒ 这笔收入对 v 的边际是
    d(money)/dv = (22.5/L) × 每落地金额
而 forward_gauss 拉回 0.95 的边际是
    d/dv = 1500 × walk_gate × forward_weight × (−(v−target)/σ²) × exp(...)
两者符号相反。谁大，速度就停在哪一侧——这决定"速度回不到 0.95"到底是门控没牙还是结构补贴。

只读：加载 best_model_snapshots 里的历史快照（不碰 best_model.zip，那个正被训练进程覆写）。
"""
import glob
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
sys.path.insert(0, "/home/guo/workspace/humanoid_walk/debug_probes")

import price_batch4 as P

LANDED = ["stride_bonus_gated", "alternate_penalty", "reach_reward", "swing_min_reward_gated",
          "position_swap_gated"]
P.PER_STEP += ["swing_forward_reward", "phase_reward", "contact_phase_reward", "net_balance",
               "height_reward_gated", "survival_bonus_gated", "ctrl_cost", "slip_penalty",
               "hip_xz_angle_penalty", "support_sym_penalty", "lateral_penalty",
               "negative_stride_penalty", "stride_balance_penalty", "stride_bonus_gated"]

FW, TS, SG = P.FW, P.TS, P.SG
DT, H = 0.015, 1500


def main():
    cands = sorted(glob.glob("/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/"
                             "best_model_*.zip"),
                   key=lambda p: int(re.search(r"_(\d+)\.zip", p).group(1)))
    if not cands:
        print("没有快照可用")
        return
    path = cands[-1]
    ts = int(re.search(r"_(\d+)\.zip", path).group(1))
    print("权重：%s（zip 内 num_timesteps=%d）" % (path.split("/")[-1], ts))
    # 每步量分两类：*逐步求和*（分/集）与 *逐步取均值*（gate、速度）——混了会把速度算成 1933
    SUM_KEYS = [k for k in set(P.PER_STEP) if k not in ("speed", "walk_gate", "flat_gate",
                                                        "forward_gate")]
    MEAN_KEYS = ["speed", "walk_gate", "flat_gate"]
    tot, mn, lands = {}, {}, []
    for seed in P.SEEDS:
        ps, ll, last, n = P.rollout(path, seed)
        lands.append(len([x for x in ll if x[1] is not None]))
        for k in SUM_KEYS:
            tot[k] = tot.get(k, 0.0) + sum(ps[k]) / len(P.SEEDS)
        for k in MEAN_KEYS:
            mn[k] = mn.get(k, 0.0) + float(np.mean(ps[k])) / len(P.SEEDS)
        tot["reward"] = tot.get("reward", 0.0) + sum(ps["reward"]) / len(P.SEEDS)
    nl = float(np.mean(lands))
    v, gate = mn["speed"], mn["walk_gate"]
    print("det：总分 %.0f，落地 %.0f 次/集，速度 %.3f m/s，walk_gate %.2f，flat_gate %.2f" % (
        tot["reward"], nl, v, gate, mn["flat_gate"]))
    print("\n每落地结算的收入（分/集 与 分/落地）：")
    per_landing_total = 0.0
    for k, label in [("stride_bonus_gated", "stride_bonus(门控后)"),
                     ("alternate_penalty", "alternate(门控后，负值=收入)"),
                     ("reach_reward", "reach"),
                     ("swing_min_reward_gated", "swing_min(⑤)"),
                     ("swing_forward_reward", "swing_forward"),
                     ("position_swap_gated", "position_swap(门控后)")]:
        val = tot.get(k, 0.0)
        # alternate_penalty 在 reward 里是 **减去** 的，所以它作为收入的符号要翻过来
        inc = -val if k == "alternate_penalty" else val
        per_landing_total += inc
        print("  %-28s %8.0f 分/集   %6.3f 分/落地" % (label, inc, inc / max(nl, 1)))
    print("  %-28s %8.0f 分/集   %6.3f 分/落地  ← **步频补贴的分子**" % (
        "合计（每落地类）", per_landing_total, per_landing_total / max(nl, 1)))

    L = v * H * DT / max(nl, 1)
    slope_events = (H * DT / L) * (per_landing_total / max(nl, 1))     # = 每落地合计 × 每 m/s 的落次数
    gauss_slope = H * gate * FW * (-(v - TS) / SG ** 2) * np.exp(-((v - TS) ** 2) / (2 * SG ** 2))
    print("\n实测步长 L = %.3f m（= v·22.5s/落地数）" % L)
    print("  步频补贴边际   d(money)/dv = %+.0f 分/集 per (m/s)   [每 m/s 多 %.0f 次落地 × %.3f 分/次]" % (
        slope_events, H * DT / L, per_landing_total / max(nl, 1)))
    print("  高斯拉回边际   d(gauss)/dv = %+.0f 分/集 per (m/s)   [v=%.2f, target=%.2f, σ=%.2f]" % (
        gauss_slope, v, TS, SG))
    print("  净边际 = %+.0f ⇒ %s" % (slope_events + gauss_slope,
          "速度**没有**回到 0.95 的内在动力（事件补贴赢）" if slope_events + gauss_slope > 0
          else "高斯赢，速度应往 0.95 靠"))
    for s in (0.40, 0.35, 0.30, 0.25):
        g = H * gate * FW * (-(v - TS) / s ** 2) * np.exp(-((v - TS) ** 2) / (2 * s ** 2))
        print("    若 σ=%.2f：高斯边际 %+6.0f，净 %+6.0f ⇒ %s" % (
            s, g, slope_events + g, "会把速度压回 0.95-1.05" if slope_events + g < 0 else "仍压不住"))


if __name__ == "__main__":
    main()
