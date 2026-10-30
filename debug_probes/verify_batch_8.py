"""批次⑧ 落地验收：**实测**新价格对现策略的起步冲击（不是估的），并核对门控算术。

对照基准都是**今天在同一份权重、同样 det×两种子(123/2026)、同一 wrapper 路径**上量到的旧值：
  · 272K 本轮 best：总分 2283，通道见 price_batch4_walk_cf.py 的"弹跳实测"列
  · 752K warm-start：总分 1859（§33 的 k=0 被动行）
旧→新的差 = 这批的起步冲击 = 策略醒来发现世界变穷了多少。B1 的先例是 **−443 ⇒ 第 1 次 eval = −47、
约 200K 步才重拟合**，所以冲击的大小决定这一晚跑出来的是"实验"还是"塌方"。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/best_model/"
CKPTS = [("272K 本轮best", ROOT + "best_model_snapshots/best_model_272000.zip", 2283.0),
         ("752K warm", ROOT + "tmp_model_data/last_TMP/best_model.zip", 1859.0)]
HORIZON, SEEDS = 1500, (123, 2026)
CHANNELS = ["height_reward_gated", "survival_bonus_gated", "position_swap_gated",
            "stride_bonus", "alternate_penalty", "reach_reward", "swing_min_reward",
            "forward_gauss_gated", "speed_penalty", "flight_phase_penalty"]


def run(path):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    model = SAC.load(path, env=env, device="cpu")
    rows = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        n, tot = 0, 0.0
        acc = {k: 0.0 for k in CHANNELS}
        info = {}
        for _ in range(HORIZON):
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            n += 1
            tot += float(r)
            for k in CHANNELS:
                acc[k] += float(info.get(k, 0.0))
            if term or trunc:
                break
        rows.append(dict(floor=env.dense_ds_floor, ref=env.dense_ds_ref,
                         n=n, tot=tot, acc=acc, ds=float(info.get("double_support_frac", np.nan)),
                         gate=float(info.get("dense_ds_factor", np.nan)),
                         rom=float(info.get("ev_rom_L_mean", np.nan)),
                         single=float(info.get("single_frac", np.nan)),
                         flight=float(info.get("flight_frac", np.nan))))
    del model, env, raw
    return rows


def main():
    print("批次⑧ 参数：flight 税 0.25→0.60｜σ 0.40→0.28｜STRIDE_WEIGHT 20→35｜"
          "dense×(floor+0.35·clip(ds/0.05))｜OSC 已关\n")
    for name, path, old_total in CKPTS:
        rs = run(path)
        m = lambda k: float(np.mean([x[k] for x in rs]))
        tot = m("tot")
        print("── %s ──" % name)
        print("   存活 %s｜总分 %.0f（旧 %.0f ⇒ 起步冲击 **%+.0f 分/集 = %.0f%%**）｜det ds %.4f｜门控系数 %.3f"
              "｜rom_L %.2f°｜flight %.3f" % (
                  [x["n"] for x in rs], tot, old_total, tot - old_total,
                  100.0 * (tot - old_total) / old_total, m("ds"), m("gate"), m("rom"), m("flight")))
        # 算术核对：gate 必须等于 0.5 + 0.5·clip(ds/0.05)
        ds = m("ds")
        fl, rf = rs[0]["floor"], rs[0]["ref"]
        expect = fl + (1.0 - fl) * min(max(ds / rf, 0.0), 1.0)
        print("   门控算术核对：ds=%.4f → 期望 %.3f，实测 %.3f ⇒ %s" % (
            ds, expect, m("gate"), "✅" if abs(expect - m("gate")) < 1e-6 else "❌ 不符"))
        gsum = {k: float(np.mean([x["acc"][k] for x in rs])) for k in CHANNELS}
        print("   通道(分/集)：" + "  ".join("%s=%.0f" % (k.replace("_gated", "").replace("_reward", ""), v)
                                        for k, v in gsum.items()))
        if "272K" in name:
            print("   旧通道参照：gauss 288｜stride 198｜alternate 132｜reach 130｜swing_min 16｜"
                  "height 1339｜survival 690｜swap 183｜税 −694（其中 flight −172）")
    print("\n判读（预先声明）：")
    print("   冲击 |Δ| ≤ ~450 分/集 → 与 B1 同量级，我们扛过一次，可判读；")
    print("   冲击 ≤ ~900 → 偏大但仍可判读；")
    print("   冲击 > ~1200 → 这一晚大概率只会记录到塌方（全体摔倒、eval 长期为负），答不了你的问题，")
    print("                  建议把 dense_ds_floor 从 0.5 抬到 0.65 再开跑。")
    print("\n恶性最优对照（**floor 的真正约束不是冲击大小，而是这个**）：")
    print("   低姿双撑挪动者（躯干 1.15→height band 0.5、single≈0→flat_gate 0.74、v≈0.4、无腾空不吃税）")
    print("     ≈ 555(height)+555(survival)+100(swap)+9(gauss) = **1219 分/集**")
    print("   floor=0.50 ⇒ 弹跳 1182 < 1219 ⇒ 挪动者反超，且它满足 ep_len=1500 且 ds≥0.03 ⇒ K1 假阳性")
    print("   floor=0.65 ⇒ 弹跳 ≈1549 > 1219 ⇒ 匍匐不占优；理想走路 ≈4394 仍远在两者之上")


if __name__ == "__main__":
    main()
