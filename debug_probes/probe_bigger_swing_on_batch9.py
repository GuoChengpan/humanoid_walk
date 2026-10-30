"""在 **⑨ 自己的权重**上重测"更大摆动到底行不行"（只读；不再引用 §17 那次弹跳权重上的结论）。

要判的两种说法：
  (A) 用户：**reward 没到位**，策略是在**套利**当前价格 ⇒ 那么"强迫它把腿摆更大"时，它应当**照样能活**，
      而且步幅/收入立刻变好 ⇒ 说明能力在、只是没被选；
  (B) H1：卡点是"大摆幅本身不稳定"⇒ 强迫更大摆幅会**缩短存活**，甚至摔 ⇒ 说明价格修不出来。
做法：对 actor 输出的髋前后动作维度做增益放大 `a[idx] ← clip(g·a[idx], −1, 1)`，g 越大摆幅越大；
四种组合 × 两种子，量：存活步数、去抖后的真实步幅、ds/flight/rom、v_net、整集回报与**分/步**。
"分/步"是关键：如果强迫大步之后分/步**上升**，说明现奖励确实偏爱大步（A 的反证 ⇒ 问题在搜索/信用分配）；
如果分/步**下降**且存活变短，说明大步在这台机体上是要付代价的（B）。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

PATH = "/home/guo/workspace/humanoid_walk/best_model/best_model_snapshots/best_model_1120000.zip"
HORIZON, SEEDS, DT = 1500, (123, 2026), 0.02
VARIANTS = [("基线 g=1/1", 1.0, 1.0), ("只放大左 g=2/1", 2.0, 1.0),
            ("双腿 g=2/2", 2.0, 2.0), ("双腿 g=3/3", 3.0, 3.0)]


def hip_y_indices(model):
    idx = []
    for name in ("left_hip_y", "right_hip_y"):
        found = None
        for j in range(model.nu):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) == name:
                found = j
                break
        idx.append(found)
    return idx


import mujoco


def run(gL, gR, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    w = env
    u = env.unwrapped
    iL, iR = hip_y_indices(u.model)
    model = SAC.load(PATH, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    bx0 = None
    sat_frames, sat_hits = 0, 0
    stride_sum = {"L": [], "R": []}
    prev_contact = {"L": True, "R": True}
    prev_bx_at_land = {"L": None, "R": None}
    last_land = {"L": -99, "R": -99}
    n, tot, sb_sum = 0, 0.0, 0.0
    info = {}
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        act = np.asarray(act, dtype=np.float64).copy()
        act[iL] = float(np.clip(gL * act[iL], -1.0, 1.0))
        act[iR] = float(np.clip(gR * act[iR], -1.0, 1.0))
        obs, r, term, trunc, info = env.step(act)
        n += 1
        tot += float(r)
        sb_sum += float(info.get("stride_bonus", 0.0))
        c = {"L": bool(w.left_contact_prev), "R": bool(w.right_contact_prev)}
        bx = float(u.data.qpos[0])
        if bx0 is None:
            bx0 = bx
        sat_frames += 1
        sat_hits += int(abs(act[iL]) >= 0.999 or abs(act[iR]) >= 0.999)
        for side in ("L", "R"):
            if c[side] and not prev_contact[side]:
                if n - last_land[side] >= 5:                       # 去抖：5 帧(0.1 s)内算同一次落地
                    if prev_bx_at_land[side] is not None:
                        stride_sum[side].append(bx - prev_bx_at_land[side])
                    prev_bx_at_land[side] = bx
                    last_land[side] = n
            prev_contact[side] = c[side]
        if term or trunc:
            break
    out = dict(n=n, tot=tot, per_step=tot / max(n, 1), sb=sb_sum,
               stride_L=float(np.mean(stride_sum["L"])) if stride_sum["L"] else float("nan"),
               stride_R=float(np.mean(stride_sum["R"])) if stride_sum["R"] else float("nan"),
               steps=len(stride_sum["L"]) + len(stride_sum["R"]),
               ds=float(info.get("double_support_frac", np.nan)),
               fl=float(info.get("flight_frac", np.nan)),
               romL=float(info.get("ev_rom_L_mean", np.nan)), romR=float(info.get("ev_rom_R_mean", np.nan)),
               vnet=(bx - bx0) / max(n * DT, 1e-6),
               clip_pen=sat_hits / max(sat_frames, 1))
    del model, env, raw
    return out


def main():
    print("权重=⑨ 的 1120K 快照｜det × 两种子(123/2026)｜对左/右髋前后动作做增益放大\n")
    hdr = "%-16s %6s %8s %8s %8s %8s %7s %7s %7s %6s %6s %6s"
    print(hdr % ("变体", "存活", "分/步", "整集分", "strideL", "strideR", "步数", "ds", "flight", "v_net", "romL", "romR"))
    print("（stride = 去抖后同脚相邻落地间的躯干位移；起付线 0.10 m；分/步是关键对比量）")
    base = None
    for name, gL, gR in VARIANTS:
        rs = [run(gL, gR, s) for s in SEEDS]
        m = lambda k, _rs=rs: float(np.mean([x[k] for x in _rs]))
        if base is None:
            base = {k: m(k) for k in ("n", "per_step", "tot", "stride_L", "stride_R", "steps", "ds", "fl", "romL", "romR")}
        print(hdr % (name, "%.0f" % m("n"), "%.3f" % m("per_step"), "%.0f" % m("tot"),
                     "%.3f" % m("stride_L"), "%.3f" % m("stride_R"), "%.0f" % m("steps"),
                     "%.3f" % m("ds"), "%.3f" % m("fl"), "%.2f" % m("vnet"),
                     "%.1f" % m("romL"), "%.1f" % m("romR")))
        d_ps = m("per_step") - base["per_step"]
        d_n = m("n") - base["n"]
        print("%-16s   → 相对基线：分/步 %+.3f｜存活 %+d 步｜stride %+.3f/%+.3f m｜动作饱和比例 %.0f%%" % (
            "", d_ps, d_n, m("stride_L") - base["stride_L"], m("stride_R") - base["stride_R"],
            100.0 * m("clip_pen")))

    print("\n判读（写在前）：")
    print("  A 成立（是价格/搜索问题）：强迫大步后 **存活不缩短** 且 stride 跨过 0.10 m 且 分/步 不降 ⇒")
    print("     能力在、奖励也在付它，只是 SGD 没选 ⇒ 该改的是信用分配/结构，不是继续加惩罚。")
    print("  B 成立（是大摆幅不稳定）：强迫大步后 **存活明显缩短**（哪怕分/步上升）⇒")
    print("     价格修不出来，得给稳定性（卸载/低重力/参考先验）。")
    print("  混合：stride 跨线但存活缩短 ⇒ 两个问题同时存在，要分开报。")


if __name__ == "__main__":
    main()
