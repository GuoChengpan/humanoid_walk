"""D4 闸门：把"写死"从**轨迹层**换到**权限层**。

D1 的教训（§45，实测）：按时钟强制 hip_y 波形**并没有强制落地节拍** —— 落地仍 8–12 次/s，
因为策略还能用 knee/hip_x 在支撑期内反复轻点。所以"写死交替"若还要试，唯一没测过的形式是
**禁止策略在支撑期指挥那条腿**：stance 窗口内该腿的 hip_y/knee 由位置伺服锁在触地那一刻的构型，
策略只能在摆相窗口里动这条腿。这样落地频率在结构上就被时钟上限锁住。

预注册放行线（跑前写死，事后不挪）：
  存在 variant 使 3 seed 的 det rollout 同时满足
    ep_len >= 400
    落地次数/s <= 3.0          ← 这才是"写死成功"的判据（D1 全部 8–12 ⇒ 等于没写死）
    spacing(|Lx-Rx| 落地时中位) >= 0.10 m
  若落地/s 仍 > 6 ⇒ 权限门控也控制不住接触时刻 ⇒ 写死路线（D1/D2/D4）全部判死，转 B2。

variant：hold = 锁 hip_y+knee（硬/软增益）；holdY = 只锁 hip_y。摆相内 hip_y 用 §44 的人体节律脉冲。
"""
import sys
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/"
POL = ROOT + "best_model/tmp_model_data/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip"
HORIZON, SEEDS, DT = 1500, (123, 2026, 7), 0.015
CYCLE, SWING, A, DUTY = 57.0, 21.0, 0.35, 0.63     # 人体节律（§44）；A 取 D1 里给出人体量级步幅的档
VARIANTS = {"hold_both_hard": (2000.0, 60.0, True), "hold_both_soft": (300.0, 20.0, True),
            "hold_hipY_only": (2000.0, 60.0, False)}


class D4Authority(gym.Wrapper):
    def __init__(self, env, kp, kd, hold_knee):
        super().__init__(env)
        m = env.unwrapped.model
        self.m = m
        self.kp, self.kd, self.hold_knee = kp, kd, hold_knee
        self.idx, self.jadr, self.vadr, self.geom = {}, {}, {}, {}
        for s in ("left", "right"):
            for jn in (f"{s}_hip_y", f"{s}_knee"):
                self.idx[jn] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                self.jadr[jn] = m.jnt_qposadr[j]
                self.vadr[jn] = m.jnt_dofadr[j]
            self.geom[s] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot")
        self.phase = 0.0
        self.hold_q = {s: {} for s in ("left", "right")}
        self.on_prev = {s: True for s in ("left", "right")}

    def reset(self, **kw):
        self.phase = 0.0
        self.hold_q = {s: {} for s in ("left", "right")}
        return self.env.reset(**kw)

    def _swing(self, side):
        off = 0.0 if side == "left" else 0.5
        p = (self.phase - off) % 1.0
        return p < SWING / CYCLE, (p / (SWING / CYCLE))

    def _to_action(self, a, ctrl):
        lo, hi = self.m.actuator_ctrlrange[a]
        return float(np.clip(2.0 * (np.clip(ctrl, lo, hi) - lo) / (hi - lo) - 1.0, -1.0, 1.0))

    def step(self, action):
        d = self.env.unwrapped.data
        self.phase = (self.phase + 1.0 / CYCLE) % 1.0
        a = np.asarray(action, dtype=np.float64).copy()
        for s in ("left", "right"):
            in_sw, u = self._swing(s)
            on = d.geom_xpos[self.geom[s]][2] - self.m.geom_size[self.geom[s]][0] < 0.02
            if (not self.on_prev[s]) or not in_sw:                 # touchdown -> snapshot config
                self.hold_q[s] = {jn: float(d.qpos[self.jadr[jn]])
                                  for jn in ([f"{s}_hip_y", f"{s}_knee"] if self.hold_knee
                                             else [f"{s}_hip_y"])}
            self.on_prev[s] = on
            if in_sw:
                if not self.hold_knee:                              # 摆相：策略管 knee
                    a[self.idx[f"{s}_hip_y"]] = float(np.clip(-A * np.sin(np.pi * u), -1.0, 1.0))
                else:
                    a[self.idx[f"{s}_hip_y"]] = float(np.clip(-A * np.sin(np.pi * u), -1.0, 1.0))
                    a[self.idx[f"{s}_knee"]] = float(np.clip(action[self.idx[f"{s}_knee"]], -1, 1))
            else:                                                   # stance：锁住该腿
                for jn, qh in self.hold_q[s].items():
                    ctrl = self.kp * (qh - d.qpos[self.jadr[jn]]) - self.kd * d.qvel[self.vadr[jn]]
                    a[self.idx[jn]] = self._to_action(self.idx[jn], ctrl / self.m.actuator_gear[self.idx[jn], 0])
        return self.env.step(a)


def run(name, variant, seed):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    env = D4Authority(w, *variant)
    model = SAC.load(POL, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    prev = {s: True for s in ("left", "right")}
    lands, spacings, los, gauges = 0, [], [], []
    hl = {s: [] for s in ("left", "right")}
    steps = 0
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        steps += 1
        d = w.env.unwrapped.data
        for s in ("left", "right"):
            j = mujoco.mj_name2id(w.env.unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_hip_y")
            hl[s].append(float(d.qpos[w.env.unwrapped.model.jnt_qposadr[j]]))
        gauges.append((info.get("double_support_frac", np.nan), info.get("flight_frac", np.nan),
                       info.get("speed", np.nan)))
        cL, cR = bool(w.left_contact_prev), bool(w.right_contact_prev)
        for s, c in (("left", cL), ("right", cR)):
            if c and not prev[s]:
                lands += 1
                spacings.append(float(w.foot_diff))
                los.append(min(float(info.get("last_left_stride", 0.0)),
                               float(info.get("last_right_stride", 0.0))))
        prev = {"left": cL, "right": cR}
        if term or trunc:
            break
    g = np.array(gauges)
    n = max(steps, 1)
    corr = float(np.corrcoef(np.array(hl["left"])[20:-20], np.array(hl["right"])[20:-20])[0, 1]) \
        if steps > 60 else np.nan
    out = dict(variant=name, seed=seed,
               ep_len=steps, land_per_s=lands / (n * DT),
               spacing_p50=float(np.median(spacings)) if spacings else np.nan,
               lo_p50=float(np.median(los)) if los else np.nan,
               ds_max=float(np.nanmax(g[:, 0])), flight=float(np.nanmean(g[:, 1])),
               v=float(np.nanmean(g[:, 2])), anticorr=corr)
    del model, env, raw, w
    return out


def main():
    rows = []
    for name, v in VARIANTS.items():
        for s in SEEDS:
            rows.append(run(name, v, s))
    print(f"{'variant':18s} {'seed':>5} {'ep_len':>7} {'land/s':>7} {'spacing':>8} {'lo':>7} "
          f"{'ds_max':>7} {'flight':>7} {'v':>5} {'anticorr':>9}")
    for r in rows:
        print(f"{r['variant']:18s} {r['seed']:5d} {r['ep_len']:7d} {r['land_per_s']:7.2f} "
              f"{r['spacing_p50']:8.3f} {r['lo_p50']:7.3f} {r['ds_max']:7.3f} {r['flight']:7.3f} "
              f"{r['v']:5.2f} {r['anticorr']:9.2f}")
    ok = [r for r in rows if r["ep_len"] >= 400 and r["land_per_s"] <= 3.0
          and (r["spacing_p50"] or 0) >= 0.10]
    still_tapping = [r for r in rows if r["land_per_s"] > 6.0]
    print(f"\n放行线 ep_len>=400 & 落地/s<=3.0 & spacing>=0.10 : "
          f"{'PASS ' + str(len(ok)) + '/9' if ok else 'FAIL'}")
    print(f"落地/s>6（写死无效）的组数: {len(still_tapping)}/9")


if __name__ == "__main__":
    main()
