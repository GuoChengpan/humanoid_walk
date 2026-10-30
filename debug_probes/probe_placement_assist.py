"""落点辅助实验：在 ⑨ 的权重上，摆动期用 PD 把脚**拉到躯干前方**，看走路最小事件能不能被造出来。

只读实验（不加载进训练、不改 train_humanoid_sac.py）：`a[hip_y] ← clip((1−k)·a_net + k·a_pd)`，
其中 `a_pd = clip(Kp·(target − (foot_x − body_x))/0.1, −1, 1)`，**只在该脚离地（摆动期）生效**，
所以它不改变策略的节奏，只把"已经在空中的那条腿"往身体前方拉。
先做符号标定（常数动作 → 看脚相对躯干前后移动），再做 k×target 网格。

要看的是三件事同时成立：
  (1) 走路最小事件占比（落地在躯干前方 且 另一脚撑地）从 ⑨ 基线的 1.7% 明显抬升（目标 >30%）；
  (2) 存活与 ds 不明显恶化（⑨ 基线 ep_len 1500、ds 0.13、flight 0.22）；
  (3) 每步净收入不被辅助搞成负的（否则训练时它学不到东西）。
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
KP = 8.0                       # 每差 1 m 给 8 的动作量（0.1 m 误差 → 0.8）
GRID = [(0.0, 0.08), (0.3, 0.06), (0.3, 0.10), (0.5, 0.06), (0.5, 0.10), (0.7, 0.10)]


def make():
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    u = env.unwrapped
    idx = {}
    for j in range(u.model.nu):
        nm = mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        if nm:
            idx[nm] = j
    return raw, env, u, idx


def calibrate():
    print("=== 符号标定：把某条髋的动作钉成常数 200 步，看脚落在躯干前还是后（单位 m，正=前方）===")
    raw, env, u, ix = make()
    m = SAC.load(P9, env=env, device="cpu")
    best = {}
    for side in ("left", "right"):
        vals = []
        line = "  %-6s hip_y 常数 →  mean(foot_x − body_x):" % side
        for c in (-1.0, -0.5, 0.0, 0.5, 1.0):
            obs, _ = env.reset(seed=123)
            acc, n = 0.0, 0
            for _ in range(200):
                a, _ = m.predict(obs, deterministic=True)
                a = np.asarray(a, dtype=np.float64).copy()
                a[ix[side + "_hip_y"]] = c
                obs, r, term, trunc, info = env.step(a)
                acc += float(u.data.body(side + "_foot").xpos[0]) - float(u.data.qpos[0])
                n += 1
                if term or trunc:
                    break
            line += "  a=%+.1f:%+.3f" % (c, acc / max(n, 1))
            vals.append((c, acc / max(n, 1)))
        print(line)
        best[side] = 1.0 if vals[-1][1] > vals[0][1] else -1.0   # a=+1 与 a=-1 谁让脚更靠前
    del m, env, raw
    return best


def run(k, target, seed, sgn):
    raw, env, u, ix = make()
    m = SAC.load(P9, env=env, device="cpu")
    w = env
    obs, _ = env.reset(seed=seed)
    tot = ev = conj = land = planted = 0
    advs, strides, ds_l, fl_l = [], [], [], []
    prev = {"L": True, "R": True}
    pfx = {"L": None, "R": None}
    n = 0
    for _ in range(HORIZON):
        a, _ = m.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float64).copy()
        bx = float(u.data.qpos[0])
        for side, foot in (("L", "left_foot"), ("R", "right_foot")):
            sname = "left" if side == "L" else "right"
            j = ix[sname + "_hip_y"]
            if not prev[side]:                                   # 摆动期才给落点辅助
                place = float(u.data.body(foot).xpos[0]) - bx
                a[j] = float(np.clip((1 - k) * a[j] + k * np.clip(sgn[sname] * KP * (target - place), -1, 1), -1, 1))
        obs, r, term, trunc, info = env.step(a)
        n += 1
        tot += float(r)
        cur = {"L": bool(w.left_contact_prev), "R": bool(w.right_contact_prev)}
        bx = float(u.data.qpos[0])
        for side, foot in (("L", "left_foot"), ("R", "right_foot")):
            fx = float(u.data.body(foot).xpos[0])
            if cur[side] and not prev[side]:
                o = "R" if side == "L" else "L"
                land += 1
                d = fx - bx
                if d > 0:
                    ev += 1
                if cur[o]:
                    planted += 1
                if d > 0 and cur[o]:
                    conj += 1
                    advs.append(d)
                if pfx[side] is not None:
                    strides.append(fx - pfx[side])
                pfx[side] = fx
            prev[side] = cur[side]
        if term or trunc:
            break
    ds_l = float(info.get("double_support_frac", np.nan))
    fl_l = float(info.get("flight_frac", np.nan))
    out = dict(n=n, per_step=tot / max(n, 1), land=land, ev=100.0 * ev / max(land, 1),
               pl=100.0 * planted / max(land, 1), conj=100.0 * conj / max(land, 1),
               adv=float(np.median(advs)) if advs else float("nan"),
               stride=float(np.median(strides)) if strides else float("nan"),
               ds=ds_l, fl=fl_l, romL=float(info.get("ev_rom_L_mean", np.nan)))
    del m, env, raw
    return out


def main():
    sgn = calibrate()
    print("  ⇒ 采用符号：left_hip_y %+0.0f｜right_hip_y %+0.0f" % (sgn["left"], sgn["right"]))
    if True:
        tag = "自动标定"
        print("%-16s %6s %8s %8s %9s %9s %9s %8s %8s %8s" % (
            "k / target", "存活", "分/步", "落地数", "脚在身前%", "撑地落地%", "最小事件%", "领先量", "同脚间距", "ds"))
        for k, tg in GRID:
            rs = [run(k, tg, s, sgn) for s in SEEDS]
            m = lambda key: float(np.mean([x[key] for x in rs]))
            print("%-16s %6.0f %8.3f %8.0f %9.1f %9.1f %9.2f %8s %8s %8.3f" % (
                "k=%.1f t=%.2f" % (k, tg), m("n"), m("per_step"), m("land"), m("ev"), m("pl"), m("conj"),
                "%.3f" % m("adv") if np.isfinite(m("adv")) else "—",
                "%.3f" % m("stride") if np.isfinite(m("stride")) else "—", m("ds")))


if __name__ == "__main__":
    main()
