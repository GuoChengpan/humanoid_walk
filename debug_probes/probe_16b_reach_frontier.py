"""反向版假设：**不是"迈步压身高所以被罚"，是"蹲着不要钱，而蹲着才容许缩脚落地"**。

纯运动学，不加载权重、不跑 rollout：把躯干放在高度 H，只动左髋/左膝（各自**真实限位**内），
问"左脚踩得到地（±2 cm）时，脚心离大腿最远能到多少、最近能到多少"。
右脚固定成弯曲提起的姿态，只要求它不陷进地面 ⇒ 姿势物理成立。

判读线（跑之前写死）：
  R_max(H) = 该身高下"踩地可达"的最大前伸；R_min(H) = 最小（可以是负=收在身后）。
  (a) 若当前实测身高 H≈1.27 处 R_min ≲ 0 ⇒ **缩脚落地是几何可行解**，蹲着确实容许原地踏步；
  (b) 若 R_max 随 H 上升而**下降**（高个子腿更直、够不着远处）那抬身高不会强迫放脚，反向假设死；
      若 R_min 随 H 上升而**上升**（越高越难把脚收到身子下面）⇒ 抬身高**强迫**放脚，
      那 ⑰ 该动的是 height 平顶的**下沿**（1.25 抬上去），而不是再加一笔放脚奖。
  参照：人体落地瞬间脚相对膝/踝 ≈ +0.10~0.20 m（9-clip 步长 0.448 的一半量级）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco

from train_humanoid_sac import patch_mirror_symmetry

raw = gym.make("Humanoid-v4")
m = raw.unwrapped.model
patch_mirror_symmetry(m)
d = raw.unwrapped.data
Q0 = m.qpos0.copy()                      # 钉住原始 qpos0，绝不改（§42 的教训）

jid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s)
       for s in ("left_hip_y", "left_knee", "right_hip_y", "right_knee")}
adr = {k: int(m.jnt_qposadr[v]) for k, v in jid.items() if v >= 0}
rng = {k: (float(m.jnt_range[v][0]), float(m.jnt_range[v][1])) for k, v in jid.items() if v >= 0}
print("关节限位（度）：" + "  ".join(f"{k}[{np.degrees(a):.0f},{np.degrees(b):.0f}]" for k, (a, b) in rng.items()))

gl = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("left_foot").id]
gr = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("right_foot").id]
print(f"左脚 {len(gl)} 个 geom，半径 {float(m.geom_size[gl[0]][0])*100:.1f} cm；qpos0[2]={Q0[2]:.3f}")
fthigh = lambda: float(d.body("left_thigh").xpos[0])
flow = lambda: float(min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in gl))
orow = lambda: float(min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in gr))

HY = np.linspace(*rng["left_hip_y"], 71)
KN = np.linspace(*rng["left_knee"], 71)
RES = []
for H in [1.15, 1.20, 1.269, 1.32, 1.40, 1.45, 1.52, 1.594]:
    rr = []
    for hy in HY:
        for kn in KN:
            d.qpos[:] = Q0
            d.qpos[2] = H
            d.qpos[adr["left_hip_y"]] = hy
            d.qpos[adr["left_knee"]] = kn
            d.qpos[adr["right_hip_y"]] = np.deg2rad(-20)
            d.qpos[adr["right_knee"]] = np.deg2rad(-120)
            mujoco.mj_forward(m, d)
            if orow() < -0.02:                       # 右脚不许陷地
                continue
            if abs(flow()) <= 0.02:                  # 左脚踩在地面 ±2 cm
                rr.append(d.geom_xpos[gl[0]][0] - fthigh())
    if rr:
        a = np.array(rr)
        RES.append((H, float(a.max()), float(a.min()), float(np.mean(a)), len(a)))
        print(f"  H={H:.3f}  踩地可达 reach：最大={a.max():+.3f} 最小={a.min():+.3f} 均值={np.mean(a):+.3f} m（{len(a)} 组）")
    else:
        print(f"  H={H:.3f}  **没有任何 (髋,膝) 组合能让左脚踩到地面** ⇒ 该身高几何不可达")
    d.qpos[:] = Q0

print("\n判读：")
cur = [x for x in RES if abs(x[0] - 1.269) < 1e-6]
if cur:
    h, mx, mn, me, n = cur[0]
    print(f"  当前身高 H=1.269：缩脚(reach≈0){'可行' if mn <= 0.02 <= mx else '不可行'}"
          f"，可达区间 [{mn:+.3f}, {mx:+.3f}]")
for key, name in ((1, "R_max"), (2, "R_min")):
    seq = [(x[0], x[key]) for x in RES]
    sl = np.polyfit([s[0] for s in seq], [s[1] for s in seq], 1)[0]
    print(f"  {name} 随身高的斜率 = {sl:+.3f} m reach / m 身高",
          "⇒ 越高越能把脚伸远" if sl > 0.05 else ("⇒ 越高越伸不远" if sl < -0.05 else "⇒ 与身高基本无关"))
print("⚠ 纯几何前缘，不含 PD 力矩限制：可达 ≠ 策略有力气维持。")
