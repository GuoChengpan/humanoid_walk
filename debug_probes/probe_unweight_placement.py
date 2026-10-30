"""落点辅助 + 竖直支撑力（卸载）同时开：C 方案的可行性判定（只读，⑨ 权重，不训练）

动机（上一轮的实测）：把脚拉到躯干前方 0.152 m 是**做得到**的，但 ⑨ 的策略一被这样拉就摔
（k=0.3 → 存活 1500→114，分/步 +1.82→−0.80）。假设：失败的是"接住身体"所需的支撑力，
而不是动作本身 ⇒ 给一个竖直支撑力 `F = s·M·g`（吊架/卸载）后，同样的落点辅助应该能活。

判定（写在跑之前，跑完不改口径）：
  通过 = 存在某个 (s, k≥0.3) 同时满足：ep_len ≥ 1200、走路最小事件 ≥ 30%、分/步 > 0、ds ≥ 0.03
  不通过 = 对所有 s ≤ 0.75 的组合，要么存活 <300，要么最小事件 <10%
对照意义：s=0 的行就是上一轮那组已知会摔的基线；k=0 的行检验"纯卸载会不会自己变好"。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

B = "/home/guo/workspace/humanoid_walk/best_model/"
P9 = B + "tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip"
HORIZON, SEEDS, DT = 1500, (123, 2026), 0.02
KP, TARGET = 8.0, 0.10                       # 上一轮里唯一真把脚拉到前方(+0.152m)的那一档
SGRID = (0.0, 0.25, 0.50, 0.75)
KGRID = (0.0, 0.3, 0.5)
SIGN = -1.0                                   # 上一轮标定：hip_y 为负才是前摆


def make():
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    u = env.unwrapped
    ix = {}
    for j in range(u.model.nu):
        nm = mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        if nm:
            ix[nm] = j
    mass = float(np.sum(u.model.body_mass))
    return raw, env, u, ix, mass


def run(k, s, seed):
    raw, env, u, ix, mass = make()
    w = env
    m = SAC.load(P9, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    fz = s * mass * 9.81
    prev = {"L": True, "R": True}
    land = ev = conj = planted = 0
    advs, strides = [], []
    pfx = {"L": None, "R": None}
    n, tot = 0, 0.0
    info = {}
    for _ in range(HORIZON):
        a, _ = m.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float64).copy()
        bx = float(u.data.qpos[0])
        for side, sname, foot in (("L", "left", "left_foot"), ("R", "right", "right_foot")):
            if not prev[side]:
                place = float(u.data.body(foot).xpos[0]) - bx
                j = ix[sname + "_hip_y"]
                a[j] = float(np.clip((1 - k) * a[j] + k * np.clip(SIGN * KP * (TARGET - place), -1, 1), -1, 1))
        u.data.qfrc_applied[:] = 0.0
        u.data.qfrc_applied[2] = fz                      # free joint 的第 3 个 DOF = 世界 z 方向
        obs, r, term, trunc, info = env.step(a)
        n += 1
        tot += float(r)
        cur = {"L": bool(w.left_contact_prev), "R": bool(w.right_contact_prev)}
        bx = float(u.data.qpos[0])
        for side, foot in (("L", "left_foot"), ("R", "right_foot")):
            fx = float(u.data.body(foot).xpos[0])
            if cur[side] and not prev[side]:
                o = "R" if side == "L" else "L"
                d = fx - bx
                land += 1
                ev += int(d > 0)
                planted += int(cur[o])
                if d > 0 and cur[o]:
                    conj += 1
                    advs.append(d)
                if pfx[side] is not None:
                    strides.append(fx - pfx[side])
                pfx[side] = fx
            prev[side] = cur[side]
        if term or trunc:
            break
    out = dict(n=n, per_step=tot / max(n, 1), land=land,
               ev=100.0 * ev / max(land, 1), conj=100.0 * conj / max(land, 1), pl=100.0 * planted / max(land, 1),
               adv=float(np.median(advs)) if advs else float("nan"),
               stride=float(np.median(strides)) if strides else float("nan"),
               ds=float(info.get("double_support_frac", np.nan)), fl=float(info.get("flight_frac", np.nan)),
               vnet=(float(u.data.qpos[0]) - float(u.init_qpos[0])) / max(n * DT, 1e-6))
    del m, env, raw
    return out


def main():
    print("⑨ 碎步权重｜落点辅助 target=+0.10 m（k 是混合比例）｜竖直支撑 F = s·M·g（M=%.1f kg）" % (
        float(np.sum(make()[4]))))
    print("%-14s %6s %8s %8s %9s %9s %8s %8s %7s %7s %7s" % (
        "s / k", "存活", "分/步", "落地数", "最小事件%", "撑地落地%", "领先量", "同脚间距", "ds", "flight", "v_net"))
    for s in SGRID:
        for k in KGRID:
            rs = [run(k, s, seed) for seed in SEEDS]
            m = lambda key: float(np.mean([x[key] for x in rs]))
            print("%-14s %6.0f %8.2f %8.0f %9.1f %9.1f %8s %8s %7.3f %7.3f %7.2f%s" % (
                "s=%.2f k=%.1f" % (s, k), m("n"), m("per_step"), m("land"), m("conj"), m("pl"),
                "%.3f" % m("adv") if np.isfinite(m("adv")) else "—",
                "%.3f" % m("stride") if np.isfinite(m("stride")) else "—",
                m("ds"), m("fl"), m("vnet"),
                "   ← 通过" if (m("n") >= 1200 and m("conj") >= 30 and m("per_step") > 0 and m("ds") >= 0.03) else ""))
        print()
    print("判读（跑之前写好）：有任意一行标到「通过」⇒ C（卸载+落点辅助+退火）值得做成正式一批；"
          "全部 s 下 conj<10% 或 n<300 ⇒ 卸载也救不了「接住身体」，那就只剩 B(需要真参考) 或 A/D。")


if __name__ == "__main__":
    main()
