"""⑰ 决策前的第 3 件测量：证伪/证实"腰扭转 40°"，并符号校验"撅腚"。

上一支探针（probe_18b）读到 `abdomen_z` 中位 −40.4°（限位 ±45°）、`abdomen_y` +28.4°（限位 −75..+30，
也就是**顶到前弯硬限位**）。这两条都是强断言，而我在这个项目里已经栽过三次符号/口径错
（大腿角沿用 up 轴的 sgn、膝弯用关节坐标、ds 用末帧当集均）。所以这一支只做三件事：

  T1 符号校验：用正向运动学**显式**验证 abdomen_y>0 是不是"往前弯"、abdomen_z<0 是哪个方向，
     不靠四元数公式的直觉。
  T2 饱和度：分别统计两腰关节距限位 2°/5° 以内的帧占比——顶限位意味着这不是"姿势偏好"而是
     "被约束住的控制量"，谈加税之前必须知道。
  T3 摆动平面：对踝相对髋的水平投影做 PCA，给出摆动主轴方位角与长短轴比。
     人是矢状面（方位角≈0°、轴比≫1）。若这里方位角很大，那"脚朝前落地"失败的原因就不是
     髋不够前摆，而是**摆动平面被腰转走了**——⑰ 的定价对象要跟着换。

控制项：同一批 det rollout 复现 ⑯ 的 duty(0.345/0.328) 与 land/s(7.78)，对不上就不读表。
"""
import importlib.util
import os
import sys
import gc

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

spec = importlib.util.spec_from_file_location("b18d_main", "train_humanoid_sac.py")
W = importlib.util.module_from_spec(spec)
sys.modules["b18d_main"] = W
spec.loader.exec_module(W)

MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
SEEDS = tuple(int(x) for x in os.environ.get("S18_SEEDS", "123,2026,7,31,55,77,91,101").split(","))
FR = int(os.environ.get("S18_FRAMES", "1500"))

# ---------------- T1 符号校验：拿一个干净模型做正向运动学 ----------------
ref = gym.make("Humanoid-v4").unwrapped
md, dd = ref.model, ref.data


def fk(name, deg):
    """把关节 name 设为 deg，其余归零，返回 torso/pelvis/左踝的世界坐标。"""
    dd.qpos[:] = 0.0
    dd.qpos[2] = 1.4
    dd.qpos[3:7] = [1, 0, 0, 0]
    dd.qpos[md.jnt_qposadr[md.joint(name).id]] = np.radians(deg)
    # 自由关节的平移/姿态分量在 qpos[0:7]，上面已置为直立
    for _ in range(3):
        mujoco_fwd(dd)
    # data.xmat 在这个 binding 里是扁平 9 元行优先矩阵：世界 = R·局部
    # ⇒ 局部 x 轴 = 元素 (0,3,6)，局部 z 轴 = 元素 (2,5,8)
    hd = lambda b: np.degrees(np.arctan2(dd.xmat[md.body(b).id][3], dd.xmat[md.body(b).id][0]))
    pitch = lambda b: np.degrees(np.arctan2(dd.xmat[md.body(b).id][2], dd.xmat[md.body(b).id][8]))
    return dd.body("torso").xpos.copy(), dd.body("pelvis").xpos.copy(), \
        dd.body("left_thigh").xpos.copy(), dd.body("left_shin").xpos.copy(), hd, pitch


import mujoco


def mujoco_fwd(d):
    mujoco.mj_forward(md, d)


print("=== T1 符号校验（单关节置位，其余为 0）===")
for nm, dg in (("abdomen_y", +25.0), ("abdomen_y", -25.0), ("abdomen_z", -40.0), ("abdomen_z", +40.0),
               ("left_hip_y", -35.0), ("left_hip_y", +10.0)):
    tp, pv, th, sh, hd, pitch = fk(nm, dg)
    # 躯干俯仰用 torso 局部 z 轴在世界矢状面的倾角：>0 = 上身倒向 +x（前倾）。
    # 大腿方位用 thigh 局部 x 轴（= 骨头的"前方"），近竖直时踝-髋水平投影会变噪声，不能用来定号。
    print(f"  {nm:11s}={dg:+6.1f}°  ⇒ pelvis 俯仰 {pitch('pelvis'):+6.1f}°   左大腿俯仰 {pitch('left_thigh'):+6.1f}°"
          f"   左小腿俯仰 {pitch('left_shin'):+6.1f}°   pelvis 朝向 {hd('pelvis'):+6.1f}°"
          f"   膝相对髋(矢状) {pitch('left_shin') - pitch('left_thigh'):+6.1f}°")

model = SAC.load(MODEL)
del model.replay_buffer
gc.collect()


def roll(seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    W.patch_mirror_symmetry(raw.unwrapped.model)
    w = W.WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    w.reset(seed=seed)
    m, d = w.unwrapped.model, w.unwrapped.data
    A = {k: [] for k in ("abz", "aby", "abx", "dutyL", "dutyR", "land", "ank", "hip", "head", "rootp")}
    prev = (False, False)
    land = 0
    for _ in range(FR):
        a, _ = model.predict(w.unwrapped._get_obs(), deterministic=True)
        _, _, ter, tru, _ = w.step(a)
        if ter or tru:
            break
        g_ = lambda n: float(d.qpos[m.jnt_qposadr[m.joint(n).id]])
        A["abz"].append(g_("abdomen_z")); A["aby"].append(g_("abdomen_y")); A["abx"].append(g_("abdomen_x"))
        hitL, hitR, _, _ = w._foot_contact_state(m, d)
        A["dutyL"].append(int(hitL)); A["dutyR"].append(int(hitR))
        land += int((not prev[0] and hitL) + (not prev[1] and hitR))
        prev = (hitL, hitR)
        A["ank"].append(d.body("left_foot").xpos.copy()); A["ank"].append(d.body("right_foot").xpos.copy())
        A["hip"].append(d.body("left_thigh").xpos.copy()); A["hip"].append(d.body("right_thigh").xpos.copy())
        R = d.xmat[m.body("pelvis").id]
        A["head"].append(np.degrees(np.arctan2(R[3], R[0])))
        qw, qx, qy, qz = d.qpos[3:7]
        A["rootp"].append(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qx * qz), -1, 1))))
    return A, len(A["dutyL"]), land


data = [roll(s) for s in SEEDS]
data = [x for x in data if x[1] > 300]
n_fr = sum(x[1] for x in data)
print(f"\n=== 控制：{len(data)} 个 seed / {n_fr} 帧 ===")
dL = np.concatenate([np.array(a["dutyL"]) for a, _, _ in data])
dR = np.concatenate([np.array(a["dutyR"]) for a, _, _ in data])
print(f"  duty L {dL.mean():.3f} R {dR.mean():.3f}  ⇒ duty_sum={dL.mean()+dR.mean():.3f}"
      f"（上一支探针 0.345/0.328=0.673）")
print(f"  ds 帧占比 {(dL * dR).mean():.4f}   腾空 {(1 - dL - dR + dL * dR).mean():.4f}"
      f"   ——**ds>0 的充要条件是每腿 duty>0.5**，现在 {min(dL.mean(), dR.mean()):.3f}")

abz = np.degrees(np.concatenate([np.array(a["abz"]) for a, _, _ in data]))
aby = np.degrees(np.concatenate([np.array(a["aby"]) for a, _, _ in data]))
abx = np.degrees(np.concatenate([np.array(a["abx"]) for a, _, _ in data]))
print("\n=== T2 饱和度（限位：abdomen_z ±45°，abdomen_y −75..+30°，abdomen_x ±35°）===")
for nm, v, lim in (("abdomen_z", abz, 45.0), ("abdomen_y", aby, 30.0), ("abdomen_x", abx, 35.0)):
    near = np.mean(np.abs(np.abs(v) - lim) < 5.0)
    print(f"  {nm:11s} p10/p50/p90 {np.percentile(v,10):+6.1f}/{np.median(v):+6.1f}/{np.percentile(v,90):+6.1f}°"
          f"   距限位<5° 的帧占比 {near*100:5.1f}%   <2° {np.mean(np.abs(np.abs(v)-lim)<2)*100:5.1f}%")

print("\n=== T3 摆动平面（踝相对髋的水平投影 PCA；人≈矢状面，方位角 0°）===")
head = np.concatenate([np.array(a["head"]) for a, _, _ in data])
print(f"  pelvis 朝向 p10/p50/p90 = {np.percentile(head,10):+.1f}/{np.median(head):+.1f}/"
      f"{np.percentile(head,90):+.1f}°   圆均值 {np.degrees(np.arctan2(np.sin(np.radians(head)).mean(), np.cos(np.radians(head)).mean())):+.1f}°"
      f"（前进方向 = 世界 +x = 0°）")
for side in ("left", "right"):
    vec = []
    for a, nfr, _ in data:
        ank = np.array(a["ank"])[0::2] if side == "left" else np.array(a["ank"])[1::2]
        hip = np.array(a["hip"])[0::2] if side == "left" else np.array(a["hip"])[1::2]
        r = ank[:nfr] - hip[:nfr]
        r = r - r.mean(0)
        for f in r:
            vec.append([f[0], f[1]])
    V = np.array(vec)
    C = np.cov(V.T)
    ev, evec = np.linalg.eigh(C)
    major = evec[:, -1]
    az = np.degrees(np.arctan2(major[1], major[0])) % 180
    az = az if az <= 90 else az - 180
    print(f"  {side:5s} 摆动主轴方位 {az:+5.1f}°（0=纯前后，±90=纯左右）  "
          f"轴比 √(λ大/λ小)={np.sqrt(ev[-1]/max(ev[0],1e-12)):.2f}   "
          f"前向行程 {V[:,0].max()-V[:,0].min():.3f} m  侧向行程 {V[:,1].max()-V[:,1].min():.3f} m")
print("  人参照：摆动几乎纯矢状（侧向行程/前向行程 ≪1，主轴方位≈0°）")

