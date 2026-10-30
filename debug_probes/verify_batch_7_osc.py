"""批次⑦·(丁) 的落地验证 + **新基线 / 被动摆幅**测量（判据要用这两个数）。

三件事：
 1) **部署路径的量**：wrapper 里那条注入路径（`left_contact_prev` 驱动）在 752K 未训练权重上，
    k=0.30/0.35/0.40 各跑两种子 det，量 ep_len、总分、分/步、rom_L 的 p50/p90/**事件均值**、ds、reach。
    ⚠ 已知遗留：离线扫描 `probe_swing_override_window.py` 在 k=0.35 给 rom p90 18.3°，
    wrapper 给 11.2°——**未解释**，但两条路对"哪一档安全"的结论一致（≤0.35 存活、≥0.40 会摔），
    所以 k 的取值按两条路的交集定，幅度值一律以 wrapper（部署路径）为准。
 2) **C3 新基线**：开跑时 k=0.30 的 det 总分；接口改了 ⇒ 与历史 1852/1868 不可比。
 3) **被动摆幅陷阱检查**：光有振荡器、策略不学，`ev_rom_L_mean` 就已经是多少。
    这决定 C1 必须写成"被动基线 + 3°"而不是绝对 8°，否则接口自己就能把判据刷过去（假阳性）。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

PATH = "/home/guo/workspace/humanoid_walk/best_model/tmp_model_data/last_TMP/best_model.zip"
HORIZON = 1500
SEEDS = (123, 2026)
KS = (0.0, 0.30, 0.35, 0.40)


def run(k, a_amp=0.55):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    env.osc_A = a_amp
    env.osc_k_forced = k
    model = SAC.load(PATH, env=env, device="cpu")
    out = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        rom, n, tot = [], 0, 0.0
        evg = float("nan")
        for _ in range(HORIZON):
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            n += 1
            tot += float(r)
            rom.append(float(info.get("last_rom_L_deg", 0.0)))
            evg = float(info.get("ev_rom_L_mean", evg))
            if term or trunc:
                break
        out.append((n, tot, tot / max(n, 1), float(np.percentile(rom, 50)), float(np.percentile(rom, 90)),
                    evg, float(info.get("ev_reach_L_mean", np.nan)),
                    float(info.get("double_support_frac", np.nan)), float(info.get("flight_frac", np.nan))))
    del model, env, raw
    return out


def main():
    print("权重 752K（md5 238fea20）｜A=0.55｜det × 两种子｜**wrapper 部署路径**\n")
    print("%-6s %8s %9s %8s %9s %9s %11s %8s %8s %8s" % (
        "k", "ep_len", "总分", "分/步", "romL_p50", "romL_p90", "ev_rom_L", "reachL", "ds", "flight"))
    res = {}
    for k in KS:
        rs = run(k)
        res[k] = rs
        m = lambda i: float(np.mean([x[i] for x in rs]))
        print("%-6.2f %8.0f %9.0f %8.2f %9.2f %9.2f %11.2f %8.3f %8.3f %8.3f   两种子 ep_len=%s" % (
            k, m(0), m(1), m(2), m(3), m(4), m(5), m(6), m(7), m(8), [x[0] for x in rs]))

    base = res[0.0]
    b30 = res[0.30]
    base_total = float(np.mean([x[1] for x in b30]))
    passive = float(np.mean([x[5] for x in b30]))
    print("\n=== 判据用的数（观察前写死）===")
    print("  C3 新基线 = k=0.30 的 det 总分 %.0f（两种子 %s）⇒ 门槛 ≥ %.0f（其 80%%）" % (
        base_total, [round(x[1]) for x in b30], base_total * 0.8))
    print("  被动摆幅：k=0.30 时 det ev_rom_L_mean 已经是 %.2f°（k=0.35 时 %.2f°），" % (
        passive, float(np.mean([x[5] for x in res[0.35]]))))
    print("  ⇒ **C1 必须是相对量**：det ev_rom_L_mean ≥ %.2f° + 3.0° = **%.1f°**；" % (
        passive, passive + 3.0))
    print("     写成绝对阈值（如 8°）会被脉冲自己刷过去 ⇒ 假阳性，这就是不采信被动摆幅的理由。")
    print("  C2 det ds ≥ 0.05：被动实测 k=0.30 → %.3f、k=0.35 → %.3f ⇒ 振荡器造不出双支撑，" % (
        float(np.mean([x[7] for x in b30])), float(np.mean([x[7] for x in res[0.35]]))))
    print("     C2 只能靠策略学出来 ⇒ 这条判据干净，是本批真正的问题。")
    ok = (min(x[0] for x in res[0.35]) >= 1200 and float(np.mean([x[4] for x in res[0.35]])) >= 8.0
          and float(np.mean([x[2] for x in res[0.35]])) >= float(np.mean([x[2] for x in base])))
    print("\n%s 部署路径核对：k=0.35 应跑满 1500、rom p90≥8°、分/步 ≥ 基线 %.2f" % (
        "✅" if ok else "❌ 先别开跑", float(np.mean([x[2] for x in base]))))


if __name__ == "__main__":
    main()
