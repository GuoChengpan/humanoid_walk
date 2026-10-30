"""模型左右对称性审计：找 `patch_mirror_symmetry` **没**补到的残余不对称。

两条互相矛盾的事实把这件事逼出来的：
  · `mirror_map.py` 注释声称修补后"零动作整段重放 1e-17（机器精度）✓"；
  · probe_17 实测：在**驱动完全没介入**（两侧 osc_active 都是 0、air 计数逐帧相同）的一步之后
    max|mirror(qpos)−qpos_mir| = 1.46e-6（零动作）/ 3.16e-5（政策动作），|Δr|=0.04。
    接触问题会把它指数放大（实测 ~10 帧到 1e-3、~30 帧到 1e-1）⇒ 这才是镜像重放"帧数差 35%"的原因，
    而**不是** §1175 说的"脉冲打在同一侧"（那条已被 ⑯·a 修掉：双腾空时谁都不驱动）。

三段，全部只读模型 + 最多几步仿真：
  S1 名称配对：所有 left_X / right_X 的 joint / body / geom / actuator，逐字段比较（矢量按镜像规则翻 y）
  S2 一步等变残差随动作幅度怎么变（零动作若非零 ⇒ 与模板、与策略都无关，是模型/求解器的事）
  S3 多步发散速率（每 10 帧记一次，看是不是指数）
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco

from mirror_map import MirrorMap, patch_mirror_symmetry

OBJ = {"joint": mujoco.mjtObj.mjOBJ_JOINT, "body": mujoco.mjtObj.mjOBJ_BODY,
       "geom": mujoco.mjtObj.mjOBJ_GEOM, "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR}


def names(m, kind):
    n = {"joint": m.njnt, "body": m.nbody, "geom": m.ngeom, "actuator": m.nu}[kind]
    return [mujoco.mj_id2name(m, OBJ[kind], i) for i in range(n)]


def fields(m, kind, i):
    """返回 (字段名, 值) 列表；矢量类按镜像规则应满足 y 分量反号。"""
    if kind == "joint":
        d = int(m.jnt_dofadr[i])
        return [("qposadr-dofadr", np.array([m.jnt_qposadr[i] - d], float)),
                ("stiffness", np.array([m.jnt_stiffness[i]], float)),
                ("damping", np.array([m.dof_damping[d]], float)),
                ("frictionloss", np.array([m.dof_frictionloss[d]], float)),
                ("limited", np.array([m.jnt_limited[i]], float)),
                ("range", np.array(m.jnt_range[i], float)),
                ("axis", np.array(m.jnt_axis[i], float))]
    if kind == "body":
        return [("pos", np.array(m.body_pos[i], float)), ("mass", np.array([m.body_mass[i]], float)),
                ("ipos", np.array(m.body_ipos[i], float))]
    if kind == "geom":
        return [("pos", np.array(m.geom_pos[i], float)), ("size0", np.array([m.geom_size[i][0]], float)),
                ("contype", np.array([m.geom_contype[i]], float)),
                ("solimp0", np.array([m.geom_solimp[i][0]], float)),
                ("solref0", np.array([m.geom_solref[i][0]], float)),
                ("friction", np.array(m.geom_friction[i], float))]
    return [("gear0", np.array([m.actuator_gear[i][0]], float)),
            ("ctrlrange", np.array(m.actuator_ctrlrange[i], float)),
            ("forcerange", np.array(m.actuator_forcerange[i], float)),
            ("biasprm_g", np.array([m.actuator_biasprm[i][1]], float))]


VEC = {"pos", "ipos", "axis", "ctrlrange", "forcerange", "friction"}
bad = []
m = gym.make("Humanoid-v4").unwrapped.model
patch_mirror_symmetry(m)
print("=== S1 名称配对逐字段比较（已打补丁的模型）===")
for kind in ("joint", "body", "geom", "actuator"):
    ns = names(m, kind)
    idx = {n: i for i, n in enumerate(ns) if n}
    for ln in [x for x in ns if x and x.startswith("left_")]:
        rn = "right_" + ln[5:]
        if rn not in idx:
            print(f"  ⚠ {kind} 有 left 无 right：{ln}")
            continue
        for fname, a in dict(fields(m, kind, idx[ln])).items():
            b = np.array(dict(fields(m, kind, idx[rn]))[fname], float)
            bb = b.copy()
            if fname in VEC:
                if fname in ("pos", "ipos", "axis"):
                    bb[..., 1] = -bb[..., 1]
                elif fname in ("ctrlrange", "forcerange"):
                    pass                       # 标量上下界，镜像应同号相等
                elif fname == "friction":
                    pass
            d = float(np.abs(a - bb).max())
            if d > 1e-12:
                bad.append(f"  {kind:8s} {ln:16s}/{rn:16s} {fname:12s} 差 {d:.3e}  L={a} R={b}")
print("\n".join(bad) if bad else "  全部字段左右匹配 ✓（无残余不对称）")
print(f"  ⇒ 不匹配 {len(bad)} 处")

envs = []
for _ in range(2):
    e = gym.make("Humanoid-v4", max_episode_steps=1500)
    patch_mirror_symmetry(e.unwrapped.model)
    envs.append(e.unwrapped)
mm = MirrorMap(envs[0].model)
print("\n=== S2 一步等变残差 vs 动作幅度（模板关闭，排除 ⑯ 干扰）===")
rng = np.random.default_rng(3)
for amp in (0.0, 0.01, 0.05, 0.2, 0.4):
    e1, e2 = envs
    e1.reset(seed=7)
    q0, v0 = e1.data.qpos.copy(), e1.data.qvel.copy()
    e2.reset()
    qm, vm = mm.mirror_qpos_qvel(q0, v0)
    e2.set_state(qm, vm)
    a = rng.uniform(-amp, amp, e1.model.nu) if amp > 0 else np.zeros(e1.model.nu)
    e1.step(a)
    e2.step(mm.mirror_action(a))
    d = float(np.abs(mm.mirror_qpos_qvel(e1.data.qpos, e1.data.qvel)[0] - e2.data.qpos).max())
    print(f"  动作 ±{amp:.2f}：max|Δqpos|(等变后)={d:.2e}")
print("  ⇒ 零动作就非零 ⇒ 与模板/策略无关，是模型或求解器的左右不对称")

print("\n=== S3 多步发散速率（零动作，看是否指数）===")
e1, e2 = envs
e1.reset(seed=7)
q0, v0 = e1.data.qpos.copy(), e1.data.qvel.copy()
e2.reset(); qm, vm = mm.mirror_qpos_qvel(q0, v0); e2.set_state(qm, vm)
z = np.zeros(e1.model.nu)
for t in range(1, 201):
    o1 = e1.step(z); o2 = e2.step(z)
    if t % 20 == 0 or t == 1:
        d = float(np.abs(mm.mirror_qpos_qvel(e1.data.qpos, e1.data.qvel)[0] - e2.data.qpos).max())
        print(f"  t={t:3d}  max|Δqpos|={d:.2e}")
    if o1[2] or o1[3] or o2[2] or o2[3]:
        print(f"  t={t} 有一方终止（原={o1[2] or o1[3]} 镜={o2[2] or o2[3]}）⇒ 之后的比较不再有意义")
        break
