"""D2 闸门：**完全不训练、零学习**的开环可行性检查。

结构写死的部分：左右严格反相的绝对相位时钟 + 人体占空比（duty 0.63 ⇒ 摆相 0.37）+
支撑期"相对地形保持落点"（不是固定正弦；这正是之前 1800 组开环弦波 0 存活所缺的性质）+
平面 2 连杆 IK 把脚的目标位置换成 (hip_y, knee) + PD 转 motor ctrl。
策略将来只调 7 个参数；本闸门把参数固定成人体实测中位，先看结构本身站不站得住。

预注册放行线（第一次跑之前就写死，未改动）：
  存在参数组使 1500 帧内  frames >= 600  且  净前进 >= 0.50 m  且  落脚误差 p50 <= 0.10 m
上一版全组 21 帧即倒，查出两个实现错误，已修：
  (a) IK 目标符号：LUT 存的是"接触点在髋下方多深"，目标却传了 -hgt ⇒ 每帧都把脚往回上方拽；
  (b) PD 太软：42 kg 的机体，膝无关节刚度，kp_knee=120 时 0.1 rad 误差只有 12 N·m。
先跑 static-hold 子测试（增益能不能站住），再跑参数图。
"""
import sys
import itertools
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco

DT = 0.015
HP = dict(kp_hip=600.0, kd_hip=40.0, kp_knee=400.0, kd_knee=25.0, kp_other=150.0, kd_other=10.0)


class LegGate:
    """LUT: (hip_y, knee) -> foot contact (forward offset from hip, depth below hip)."""

    def __init__(self, model, side="left"):
        self.m, self.side = model, side
        self.names = [f"{side}_hip_y", f"{side}_knee", f"{side}_hip_x", f"{side}_hip_z"]
        self.aidx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.names}
        self.jadr = {}
        self.vadr = {}
        for n in self.names:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.jadr[n] = model.jnt_qposadr[j]
            self.vadr[n] = model.jnt_dofadr[j]
        self.hip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_thigh")
        self.geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot")
        self.r = float(model.geom_size[self.geom][0])
        self.hy = np.arange(-1.90, 0.31, 0.03)
        self.kn = np.arange(-2.75, -0.05, 0.03)
        self.d = np.zeros((len(self.hy), len(self.kn), 2))
        d = mujoco.MjData(model)
        for i, a in enumerate(self.hy):
            for jj, b in enumerate(self.kn):
                d.qpos[:] = 0.0
                d.qpos[self.jadr[f"{side}_hip_y"]] = a
                d.qpos[self.jadr[f"{side}_knee"]] = b
                mujoco.mj_forward(model, d)
                off = d.geom_xpos[self.geom] - d.xpos[self.hip_body]
                self.d[i, jj] = (off[0], -off[2] + self.r)     # forward, depth below hip

    def solve(self, dx, depth):
        err = (self.d[:, :, 0] - dx) ** 2 + (self.d[:, :, 1] - depth) ** 2
        i, j = np.unravel_index(np.argmin(err), err.shape)
        return float(self.hy[i]), float(self.kn[j])


def settle(model, data, legs, steps=400):
    """static-hold sub-test: can the PD hold the robot upright with both feet on the floor?"""
    sat = 0
    for t in range(steps):
        for s, lg in legs.items():
            H = data.xpos[lg.hip_body][2]
            qh, qk = lg.solve(0.02, H)                 # knee slightly ahead of hip, foot on floor
            put(lg, f"{s}_hip_y", qh, data, model, "hip")
            put(lg, f"{s}_knee", qk, data, model, "knee")
            for n in (f"{s}_hip_x", f"{s}_hip_z"):
                put(lg, n, 0.0, data, model, "other")
        put_abs(model, data, "abdomen_x", 0.0)
        put_abs(model, data, "abdomen_y", np.radians(8.0))
        put_abs(model, data, "abdomen_z", 0.0)
        for a in range(model.nu):
            if abs(data.ctrl[a]) >= 0.399:
                sat += 1
        for _ in range(int(round(DT / model.opt.timestep))):
            mujoco.mj_step(model, data)
    return dict(z=float(data.qpos[2]), frames=steps, sat_frac=sat / (steps * model.nu))


def put(lg, name, qdes, data, model, kind):
    a = lg.aidx[name]
    err = qdes - data.qpos[lg.jadr[name]]
    vel = -data.qvel[lg.vadr[name]]
    kp, kd = {"hip": (HP["kp_hip"], HP["kd_hip"]), "knee": (HP["kp_knee"], HP["kd_knee"]),
              "other": (HP["kp_other"], HP["kd_other"])}[kind]
    data.ctrl[a] = float(np.clip((kp * err + kd * vel) / model.actuator_gear[a, 0], -0.4, 0.4))


def put_abs(model, data, jname, qdes):
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    err = qdes - data.qpos[model.jnt_qposadr[j]]
    vel = -data.qvel[model.jnt_dofadr[j]]
    data.ctrl[a] = float(np.clip((HP["kp_other"] * err + HP["kd_other"] * vel)
                                 / model.actuator_gear[a, 0], -0.4, 0.4))


def run_config(model, data, legs, cycle, lead, lift, lean_deg, steps=1500):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    z0 = min(data.geom_xpos[lg.geom][2] - lg.r for lg in legs.values())
    data.qpos[2] -= (z0 - 0.001)
    mujoco.mj_forward(model, data)
    contact_prev = {s: True for s in legs}
    held_x = {}
    ds = fl = sg = sat = 0
    errs, adv, zs = [], [], []
    frac_sw = 0.37
    for t in range(steps):
        phase = (t / cycle) % 1.0
        for s, lg in legs.items():
            off = 0.0 if s == "left" else 0.5
            p = (phase - off) % 1.0
            H = data.xpos[lg.hip_body][2]
            g = lg.geom
            if p < frac_sw:                       # swing: lift-off point -> new placement ahead of hips
                u = p / frac_sw
                tgt_x = held_x.get(s, data.geom_xpos[g][0]) * (1 - u) + (data.qpos[0] + lead) * u
                hgt = lift * np.sin(np.pi * u)
                qh, qk = lg.solve(tgt_x - data.xpos[lg.hip_body][0], H - hgt)
            else:                                 # stance: hold the world x it landed at, on the floor
                if held_x.get(s) is None:
                    held_x[s] = data.geom_xpos[g][0]
                qh, qk = lg.solve(held_x[s] - data.xpos[lg.hip_body][0], H)
            put(lg, f"{s}_hip_y", qh, data, model, "hip")
            put(lg, f"{s}_knee", qk, data, model, "knee")
            for n in (f"{s}_hip_x", f"{s}_hip_z"):
                put(lg, n, 0.0, data, model, "other")
        put_abs(model, data, "abdomen_x", 0.0)
        put_abs(model, data, "abdomen_y", np.radians(lean_deg))
        put_abs(model, data, "abdomen_z", 0.0)
        for a in range(model.nu):
            sat += int(abs(data.ctrl[a]) >= 0.399)
        for _ in range(int(round(DT / model.opt.timestep))):
            mujoco.mj_step(model, data)
        on = {s: data.geom_xpos[lg.geom][2] - lg.r < 0.02 for s, lg in legs.items()}
        ds += all(on)
        fl += not any(on)
        sg += (on["left"] != on["right"])
        for s in ("left", "right"):
            if on[s] and not contact_prev[s]:
                errs.append(data.geom_xpos[legs[s].geom][0] - data.xpos[legs[s].hip_body][0])
            held_x[s] = data.geom_xpos[legs[s].geom][0] if on[s] else held_x.get(s)
            contact_prev[s] = on[s]
        zs.append(data.qpos[2])
        adv.append(data.qpos[0])
        if data.qpos[2] < 0.7:
            break
    n = max(len(zs), 1)
    errs = np.array(errs)
    return dict(cycle=cycle, lead=lead, lift=lift, lean=lean_deg, frames=n,
                advance=float(adv[-1] - adv[0]), ds=ds / n, flight=fl / n, single=sg / n,
                steps_per_s=len(errs) / (n * DT),
                place_p50=float(np.median(errs)) if len(errs) else np.nan,
                place_err_p50=float(np.median(np.abs(errs - lead))) if len(errs) else np.nan,
                sat=sat / (n * model.nu), zmin=float(np.min(zs)))


def main():
    raw = gym.make("Humanoid-v4", max_episode_steps=1500)
    model, data = raw.unwrapped.model, raw.unwrapped.data
    legs = {s: LegGate(model, s) for s in ("left", "right")}
    print("static-hold (PD 能否站住 400 帧):", settle(model, data, legs), flush=True)
    rows = []
    for cycle, lead, lift, lean in itertools.product((45, 57, 70), (0.15, 0.22, 0.30),
                                                     (0.06, 0.10), (0.0, 10.0)):
        r = run_config(model, data, legs, cycle, lead, lift, lean)
        rows.append(r)
        print(f"cycle={cycle} lead={lead:.2f} lift={lift:.2f} lean={lean:4.1f} -> frames={r['frames']:5d} "
              f"adv={r['advance']:6.2f}m steps/s={r['steps_per_s']:5.2f} place_p50={r['place_p50']:6.3f} "
              f"err={r['place_err_p50']:6.3f} ds={r['ds']:.3f} fl={r['flight']:.3f} "
              f"sat={r['sat']:.2f} zmin={r['zmin']:.2f}", flush=True)
    ok = [r for r in rows if r["frames"] >= 600 and r["advance"] >= 0.50 and r["place_err_p50"] <= 0.10]
    print(f"\n放行线（frames>=600 & advance>=0.50m & place_err<=0.10m）: "
          f"{'PASS ' + str(len(ok)) + '/' + str(len(rows)) + ' 组' if ok else 'FAIL ⇒ D2 生成器不成立'}")
    raw.close()


if __name__ == "__main__":
    main()
