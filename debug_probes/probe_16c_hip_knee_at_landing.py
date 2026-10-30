"""reach 不动，怪谁？把每次落地拆成 (髋角, 膝角) 并对照几何地图。

已知（⑮ 1.19M vs ⑬⑭ 停训）：ev_knee_ext_R 36→61、rom_R 34→54、lift_R 0.076→0.146 全在涨，
而 reach_R 全程 ≈0。几何前缘（probe_16b）显示在当前身高 +0.248 m 的 reach 是可达的。
⇒ 唯一还没查的量：**落地那一瞬的 left/right_hip_y**。

两张表：
  表 A 地图 reach(hip_y, knee)（纯运动学，单腿，脚踩地）——告诉你"要到 +0.15 需要什么样的髋/膝配对"
  表 B 实测：每次落地帧的 (hip_y, knee, reach)，左右分开，给 p10/p50/p90
判读线（跑前写死）：
  · 若实测 hip_y 的 p90 已经落在"要 reach=+0.15 所需的髋角"上，而 reach 仍 ≈0 ⇒ **膝/踝侧的问题**；
  · 若实测 hip_y 从不进入那个区间 ⇒ **髋没往前放**：而髋是被模板 T1 驱动的那一路（osc_A=0.15 写死），
    此时该动的是模板的相位/振幅，不是价格，也不是膝。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

DT, FR = 0.015, 1500
SEEDS = tuple(int(x) for x in os.environ.get("K_SEEDS", "123,2026,7,31,55,77").split(","))
MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
TARGET = 0.15

raw = gym.make("Humanoid-v4")
m, d = raw.unwrapped.model, raw.unwrapped.data
patch_mirror_symmetry(m)
Q0 = m.qpos0.copy()
jid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s) for s in
       ("left_hip_y", "left_knee", "right_hip_y", "right_knee")}
adr = {k: int(m.jnt_qposadr[v]) for k, v in jid.items() if v >= 0}
rng = {k: (float(m.jnt_range[v][0]), float(m.jnt_range[v][1])) for k, v in jid.items() if v >= 0}


def foot_geom(side):
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{side}_foot").id]


def reach_at(side, hip, knee, H=1.269):
    gids = foot_geom(side)
    d.qpos[:] = Q0
    d.qpos[2] = H
    d.qpos[adr[f"{side}_hip_y"]] = hip
    d.qpos[adr[f"{side}_knee"]] = knee
    other = "right" if side == "left" else "left"
    d.qpos[adr[f"{other}_hip_y"]] = np.deg2rad(-20)
    d.qpos[adr[f"{other}_knee"]] = np.deg2rad(-120)
    mujoco.mj_forward(m, d)
    low = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in gids)
    if low < -0.02:
        return None
    return (float(d.geom_xpos[gids[0]][0] - d.body(f"{side}_thigh").xpos[0]), low)


print("=== 表 A 地图：reach(髋°,膝°) @ 躯干高 1.269，脚须踩地 ±2cm ===")
print(f"{'髋_y°':>8}" + "".join(f"{'膝'+str(int(k))+'°':>10}" for k in (-5, -20, -40, -60, -90)))
HY = [0, -15, -30, -45, -60, -75]
for hy in HY:
    row = f"{hy:8.0f}"
    for kn in (-5, -20, -40, -60, -90):
        r = reach_at("left", np.deg2rad(hy), np.deg2rad(kn))
        row += f"{(r[0] if r and abs(r[1]) <= 0.02 else float('nan')):+10.3f}" if r else f"{'—':>10}"
    print(row)
print("  （正 reach = 脚在大腿前方；— = 该配对下脚踩不到地）")

print("\n=== 表 B 实测：每次落地的 (髋,膝,reach) ===")
model = SAC.load(MODEL)
rec = {"left": [], "right": []}
for sd in SEEDS:
    raw2 = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw2.unwrapped.model)
    w = WalkRewardWrapper(raw2, swing_drive_weight=2.0, osc_enable=True)
    mm, dd = w.unwrapped.model, w.unwrapped.data
    aL, aR = int(mm.jnt_qposadr[jid["left_hip_y"]]), int(mm.jnt_qposadr[jid["right_hip_y"]])
    kL, kR = int(mm.jnt_qposadr[jid["left_knee"]]), int(mm.jnt_qposadr[jid["right_knee"]])
    gl, gr = foot_geom("left"), foot_geom("right")
    prev = {"left": True, "right": True}
    obs, _ = w.reset(seed=sd)
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = w.step(np.asarray(a, dtype=np.float64))
        for side, gi, ai, ki, th in (("left", gl, aL, kL, "left_thigh"), ("right", gr, aR, kR, "right_thigh")):
            hit = any((dd.geom_xpos[g][2] - mm.geom_size[g][0]) < 0.02 for g in gi)
            if hit and not prev[side]:                    # 落地沿
                rec[side].append((np.degrees(dd.qpos[ai]), np.degrees(dd.qpos[ki]),
                                  float(dd.geom_xpos[gi[0]][0] - dd.body(th).xpos[0])))
            prev[side] = hit
        if term or trunc:
            break

for side in ("left", "right"):
    A = np.array(rec[side])
    if len(A) < 5:
        print(f"  {side}: 落地次数 {len(A)} 太少")
        continue
    p = lambda col: np.percentile(A[:, col], [10, 50, 90])
    print(f"  {side:5s} 落地 {len(A):4d} 次  髋_y° p10/p50/p90 = {p(0)[0]:+6.1f}/{p(0)[1]:+6.1f}/{p(0)[2]:+6.1f}"
          f"   膝° = {p(1)[0]:+6.1f}/{p(1)[1]:+6.1f}/{p(1)[2]:+6.1f}"
          f"   reach m = {p(2)[0]:+.3f}/{p(2)[1]:+.3f}/{p(2)[2]:+.3f}")
    need = []
    for h in HY:
        for k in (-5, -20, -40, -60):
            r = reach_at("left", np.deg2rad(h), np.deg2rad(k))
            if r and abs(r[1]) <= 0.02 and abs(r[0] - TARGET) < 0.05:
                need.append(h)
                break
    print(f"        表 A 里能给出 reach≈+{TARGET:.2f} 的髋角 = {sorted(set(need)) or '网格内无'}"
          f"；实测髋角 p90={p(0)[2]:+.1f}° ⇒ 髋{'进过' if need and p(0)[2] >= min(need) else '**没进过**'}那个区间")
print("\n⚠ 表 A 用 qpos0 的其余关节、躯干钉 1.269，是静态地图；动态落地时骨盆还会俯仰，"
      "所以它是'需要什么样的髋/膝配对'的量级参考，不是逐帧真值。")
