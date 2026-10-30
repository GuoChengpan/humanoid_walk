"""reach 参照的**几何**泄漏常数：只转躯干、别的都不动，看三种参照的 reach 各变多少。

上一支探针（probe_reach_reference）在真实 rollout 上做相关，得到
  corr/pitch：H(同侧大腿)=−0.24、T(躯干 qpos[0])=−0.43、P(髋中心)=−0.25
——**换成躯干反而更耦合**。但真实 rollout 的相关混了"行为共变"（策略在某个姿态下本来就更爱往前落），
不能当泄漏系数。这里做纯几何反事实：
  取若干帧快照 → 只把躯干四元数绕世界 y 轴转 Δpitch → mj_forward → 重新量
  Δ(脚x − 参照x)。得到的 m/rad 就是"前倾能白买多少 reach"的真值。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODELS = [p for p in [
    "/home/guo/workspace/humanoid_walk/best_model/best_model.zip",
    "/home/guo/workspace/humanoid_walk/best_model/tmp_model_data/SAC_18/18_3_min(左右步幅)/best_model.zip",
] if os.path.exists(p)]
HORIZON = 1500
N_SNAP = 400
D = 0.10            # rad，≈5.7°


def snap_points(path):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(path, env=env, device="cpu")
    m, data = env.unwrapped.model, env.unwrapped.data
    fl, fr = m.body("left_foot").id, m.body("right_foot").id
    hl, hr = m.body("left_thigh").id, m.body("right_thigh").id
    tr = m.body("torso").id
    snaps, k = [], 0
    obs, _ = env.reset(seed=123)
    for t in range(6000):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        k += 1
        if term or trunc:
            obs, _ = env.reset(seed=123)
            k = 0
        if 40 < t and len(snaps) < N_SNAP and t % 14 == 0:
            snaps.append((data.xpos[fl][0], data.xpos[fr][0], data.xpos[hl][0],
                          data.xpos[hr][0], data.xpos[tr][0], data.qpos[3:7].copy()))
    del model, env
    return snaps, m, data


def quat_about_y(theta):
    return np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])   # w,x,y,z


def mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*z2 + z1*w2])


def main():
    for path in MODELS:
        snaps, m, data = snap_points(path)
        fl, fr, hl, hr, tr = (m.body("left_foot").id, m.body("right_foot").id,
                              m.body("left_thigh").id, m.body("right_thigh").id, m.body("torso").id)
        dH_L, dH_R, dT, dP = [], [], [], []
        for lx, rx, hlx, hrx, tx0, q0 in snaps:
            row = {}
            for theta in (0.0, D):
                q = mul(quat_about_y(theta), q0) if theta else q0
                data.qpos[3:7] = q
                m.__class__ and __import__("mujoco").mj_forward(m, data)
                row[theta] = (data.xpos[fl][0], data.xpos[fr][0], data.xpos[hl][0],
                              data.xpos[hr][0], data.xpos[tr][0])
            b0, b1 = row[0.0], row[D]
            dH_L.append(b1[0] - b1[2] - (b0[0] - b0[2]))       # Δ(左脚x − 左髋x)
            dH_R.append(b1[1] - b1[3] - (b0[1] - b0[3]))
            dT.append((b1[0] - b1[4]) - (b0[0] - b0[4]))       # Δ(左脚x − 躯干x)
            dP.append((b1[0] - 0.5 * (b1[2] + b1[3])) - (b0[0] - 0.5 * (b0[2] + b0[3])))
            data.qpos[3:7] = q0
        f = lambda a: (np.mean(a), np.std(a))
        print(f"\n### {path.split('/')[-2]}/{path.split('/')[-1]}   快照 {len(snaps)} 帧，"
              f"仅躯干绕世界 y 轴转 Δpitch=+{D:.2f} rad（≈{np.degrees(D):.1f}°，正=按本仓库 pitch 正方向）")
        for nm, arr_ in (("H 同侧大腿(现行·左)", dH_L), ("H 同侧大腿(现行·右)", dH_R),
                         ("T 躯干 qpos[0]", dT), ("P 髋中心(两腿均值)", dP)):
            mu, sd = f(arr_)
            print(f"  {nm:22s} Δreach = {mu:+.4f}±{sd:.4f} m / {D:.2f}rad"
                  f"   → 泄漏系数 {mu/D:+.3f} m/rad = {mu/D*0.01745*100:+.2f} cm/度")


if __name__ == "__main__":
    main()
