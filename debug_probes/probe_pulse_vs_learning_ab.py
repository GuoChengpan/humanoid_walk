"""批次⑦ 的**归因 A/B**：左腿摆幅到底是"策略学的"还是"脉冲白送的"？

背景：⑦ 改了动作接口（`a[left_hip_y] ← (1−k)·a_net + k·pulse`），训练与评测吃同一个 k，
所以 det `ev_rom_L` 里混着两份贡献，单看它没法归因。这里把 **k 强制成 0**（等价于没有振荡器、
纯网络输出）与 k=0.30/0.35 在**同一份权重**上各跑 det 两种子，比较：
  · 同一权重换 k → 脉冲的贡献
  · 同一 k 换权重 → 训练的贡献

可读权重只有两份（训练进程不落盘最新权重，只有新 best 时才存快照）：
  752K = 本轮 warm-start（⑦ 开始前**没被本轮训练过**）
  272K = 本轮 eval 最优快照（md5 与 best_model.zip 一致，被 960K 的策略取代，只能代表早期）
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/best_model/"
CKPTS = [
    ("752K warm(本轮未训)", ROOT + "tmp_model_data/last_TMP/best_model.zip"),
    ("272K 本轮best", ROOT + "best_model_snapshots/best_model_272000.zip"),
]
KS = (0.0, 0.30, 0.35)
HORIZON = 1500
SEEDS = (123, 2026)


def run(path, k):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    env.osc_k_forced = k
    model = SAC.load(path, env=env, device="cpu")
    rows = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        rom, n, tot = [], 0, 0.0
        info = {}
        for _ in range(HORIZON):
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            n += 1
            tot += float(r)
            rom.append(float(info.get("last_rom_L_deg", 0.0)))
            if term or trunc:
                break
        rows.append(dict(
            n=n, tot=tot, per_step=tot / max(n, 1),
            rom_ev=float(info.get("ev_rom_L_mean", np.nan)),
            romR_ev=float(info.get("ev_rom_R_mean", np.nan)),
            rom_p90=float(np.percentile(rom, 90)),
            reach=float(info.get("ev_reach_L_mean", np.nan)),
            ds=float(info.get("double_support_frac", np.nan)),
            flight=float(info.get("flight_frac", np.nan)),
        ))
    del model, env, raw
    return rows


def main():
    print("det × 两种子(123/2026)｜A=0.55｜wrapper 部署路径｜k 强制档位\n")
    hdr = "%-18s %-5s %7s %8s %8s %8s %8s %8s %8s"
    print(hdr % ("权重", "k", "ep_len", "总分", "分/步", "ev_romL", "ev_romR", "reachL", "ds"))
    allres = {}
    for name, path in CKPTS:
        for k in KS:
            rs = run(path, k)
            allres[(name, k)] = rs
            m = lambda key: float(np.mean([x[key] for x in rs]))
            print(hdr % (name, "%.2f" % k, "%.0f" % m("n"), "%.0f" % m("tot"), "%.2f" % m("per_step"),
                         "%.2f" % m("rom_ev"), "%.2f" % m("romR_ev"), "%.3f" % m("reach"),
                         "%.3f" % m("ds")))
        print()

    print("=== 归因 ===")
    for name, _ in CKPTS:
        a = allres[(name, 0.0)]
        b = allres[(name, 0.35)]
        ma = lambda key: float(np.mean([x[key] for x in a]))
        mb = lambda key: float(np.mean([x[key] for x in b]))
        print("%-18s k=0 → 0.35：ev_romL %.2f° → %.2f°（脉冲贡献 %+.2f°），ds %.3f → %.3f，"
              "reachL %.3f → %.3f，总分 %.0f → %.0f" % (
                  name, ma("rom_ev"), mb("rom_ev"), mb("rom_ev") - ma("rom_ev"),
                  ma("ds"), mb("ds"), ma("reach"), mb("reach"), ma("tot"), mb("tot")))
    n0, n1 = CKPTS[0][0], CKPTS[1][0]
    print("\n训练贡献（都取 k=0，即纯网络、无脉冲）：ev_romL %.2f° → %.2f°，ds %.3f → %.3f，总分 %.0f → %.0f" % (
        float(np.mean([x["rom_ev"] for x in allres[(n0, 0.0)]])),
        float(np.mean([x["rom_ev"] for x in allres[(n1, 0.0)]])),
        float(np.mean([x["ds"] for x in allres[(n0, 0.0)]])),
        float(np.mean([x["ds"] for x in allres[(n1, 0.0)]])),
        float(np.mean([x["tot"] for x in allres[(n0, 0.0)]])),
        float(np.mean([x["tot"] for x in allres[(n1, 0.0)]]))))
    print("⚠ 272K 是本轮的**早期**快照（此后只更新到 best_model.zip 的话也停在 272K），"
          "它不等于当前 ~960K 的策略；当前权重要等 learn() 结束写 sac_humanoid_walk_final.zip。")


if __name__ == "__main__":
    main()
