"""D2 归因子测试：这套 17 电机 + 焊死球脚的身体，**任何一个固定 PD 增益**能不能站住？

为什么要先问这个：gate_d2_openloop_ik.py 里连"静态保持 400 帧"都失败（z 掉到 0.118，
43% 的 ctrl 打满 ±0.4）。这可能是
  (a) 结构本身不可行（无踝 + 球脚 → 没有踝策略，重心无法维持在支撑点上方），或
  (b) 我的 IK 目标/PD 增益写错。
本脚本只把全部 17 个关节朝一个**固定姿态目标**做 PD（不做任何步态生成），扫增益。
判读：
  若存在增益能站 1000 帧 ⇒ 失败原因是 (b)，D2 生成器值得修好再判；
  若全部站不住        ⇒ (a) 成立，"写死到关节层"在本机不可行，D1/D2 一起降级。
"""
import sys
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco

DT = 0.015


def hold(model, data, kp, kd, qdes_fn, frames=1000):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    # drop the body until the two foot spheres just touch the floor
    z = min(data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot")][2]
            - model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot")][0]
            for s in ("left", "right"))
    data.qpos[2] -= (z - 0.001)
    mujoco.mj_forward(model, data)
    sat = 0
    for t in range(frames):
        for a in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            j = model.actuator_trnid[a, 0]
            qd = qdes_fn(name, t)
            err = qd - data.qpos[model.jnt_qposadr[j]]
            tq = kp * err - kd * data.qvel[model.jnt_dofadr[j]]
            data.ctrl[a] = float(np.clip(tq / model.actuator_gear[a, 0], -0.4, 0.4))
            sat += int(abs(data.ctrl[a]) >= 0.399)
        for _ in range(int(round(DT / model.opt.timestep))):
            mujoco.mj_step(model, data)
        if data.qpos[2] < 0.7:
            break
    zmax = data.qpos[2]
    xy = float(np.hypot(data.qpos[0], data.qpos[1]))
    return dict(frames=t + 1, z=float(zmax), drift=xy,
                sat=sat / max((t + 1) * model.nu, 1))


def main():
    raw = gym.make("Humanoid-v4", max_episode_steps=1500)
    model, data = raw.unwrapped.model, raw.unwrapped.data
    print("actuator gears:", {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a):
                             float(model.actuator_gear[a, 0]) for a in range(model.nu)})
    lean = np.radians(8.0)
    targets = {
        "all-zero": lambda n, t: 0.0,
        "lean+slight-knee": lambda n, t: (lean if n == "abdomen_y"
                                          else (-0.10 if n.endswith("knee") else 0.0)),
        "lean+soft-knees": lambda n, t: (lean if n == "abdomen_y"
                                         else (-0.25 if n.endswith("knee") else 0.0)),
    }
    best = []
    for tname, fn in targets.items():
        for kp in (200.0, 400.0, 800.0, 1600.0, 3200.0):
            for kd in (20.0, 40.0, 80.0):
                r = hold(model, data, kp, kd, fn)
                r.update(target=tname, kp=kp, kd=kd)
                best.append(r)
                flag = "STANDS" if r["frames"] >= 1000 else "falls"
                print(f"{tname:20s} kp={kp:7.0f} kd={kd:4.0f} -> frames={r['frames']:5d} "
                      f"z={r['z']:5.2f} drift={r['drift']:5.2f} sat={r['sat']:.2f}  {flag}", flush=True)
    st = [r for r in best if r["frames"] >= 1000]
    print(f"\n判读: {len(st)}/{len(best)} 组能站住 1000 帧"
          f" ⇒ {'失败原因是我的 PD/IK 目标写错，D2 值得修好重判' if st else '(a) 成立：固定关节层控制在本机站不住'}")
    if st:
        print("   最佳:", min(st, key=lambda r: r["sat"]))
    raw.close()


if __name__ == "__main__":
    main()
