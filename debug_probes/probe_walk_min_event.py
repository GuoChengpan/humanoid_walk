"""走路的最小动作单元到底有没有发生过？（只读，det×两种子，三份权重）

标准（对齐用户的说法，不再用大腿当参照）：一次落地同时满足
  (a) 落点在**躯干**前方（foot_x − body_x > 0）   ⇒ 腿真的往前插，而不是被身体拖着落地
  (b) 落地瞬间**另一只脚撑着地**                ⇒ 是迈出去踩实，不是在空中换腿（弹跳/跑步的落地不满足）
(a)&(b) 同时成立 = 交替步态的最小事件。若三份权重里这个事件占比≈0，说明它从未被采样到，
任何"给它加钱"的改动都没有梯度可循（要先造出它，而不是先给它定价）。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

B = "/home/guo/workspace/humanoid_walk/best_model/"
POL = [("⑧ 弹跳 672K", B + "tmp_model_data/SAC_18/18_加重腾空惩罚/best_model_snapshots/best_model_672000.zip"),
       ("⑨ 碎步 1696K", B + "tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip")]
extra = sorted(glob.glob(B + "tmp_model_data/**/best_model_*.zip", recursive=True), key=os.path.getmtime)
if extra:
    POL.append(("最新归档", extra[-1]))
HORIZON, SEEDS = 1500, (123, 2026)


def scan(path):
    tot = ahead = conj = planted = 0
    advs, strides = [], []
    for seed in SEEDS:
        raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
        patch_mirror_symmetry(raw.unwrapped.model)
        env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
        w, u = env, env.unwrapped
        m = SAC.load(path, env=env, device="cpu")
        obs, _ = env.reset(seed=seed)
        prev = {"L": True, "R": True}
        st = {s: dict(pfx=None) for s in "LR"}
        for _ in range(HORIZON):
            a, _ = m.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            cur = {"L": bool(w.left_contact_prev), "R": bool(w.right_contact_prev)}
            bx = float(u.data.qpos[0])
            fx = {"L": float(u.data.body("left_foot").xpos[0]), "R": float(u.data.body("right_foot").xpos[0])}
            for s in "LR":
                if cur[s] and not prev[s]:
                    o = "R" if s == "L" else "L"
                    d = fx[s] - bx
                    tot += 1
                    ahead += int(d > 0)
                    planted += int(cur[o])
                    if d > 0 and cur[o]:
                        conj += 1
                        advs.append(d)
                    if st[s]["pfx"] is not None:
                        strides.append(fx[s] - st[s]["pfx"])
                    st[s]["pfx"] = fx[s]
                prev[s] = cur[s]
            if term or trunc:
                break
        del m, env, raw
    return tot, ahead, conj, planted, advs, strides


def main():
    print("走路最小事件 = 落地在躯干前方 且 另一脚同时撑地（det×两种子）\n")
    print("%-14s %6s %10s %12s %12s %12s %11s" % (
        "权重", "落地数", "脚在躯干前", "且另一脚撑地", "该事件占比", "领先量 p50", "同脚间距 p50"))
    for name, p in POL:
        if not os.path.exists(p):
            continue
        tot, ahead, conj, planted, advs, strides = scan(p)
        print("%-14s %6d %9.1f%% %11.1f%% %12.2f%% %12s %11s" % (
            name, tot, 100.0 * ahead / max(tot, 1), 100.0 * planted / max(tot, 1),
            100.0 * conj / max(tot, 1),
            ("%.3f m" % float(np.median(advs))) if advs else "从未发生",
            ("%.3f m" % float(np.median(strides))) if strides else "—"))
    print("\n判读：若三份权重的「该事件占比」都在 1% 以下 ⇒ 这个动作单元从未被采样到，")
    print("      先给它定价是无效的（没有正样本可强化）；要先用别的方式把它造出来（外部示范/结构约束/辅助）。")


if __name__ == "__main__":
    main()
