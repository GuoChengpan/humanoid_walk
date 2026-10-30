"""D1 闸门：把左右 hip_y 用**绝对反相时钟**写死（去掉 ⑦ 的"已腾空才发力"门控），
在 ⑨ 现成权重上做 det rollout，量它到底能不能站住 + 脚到底有没有前进。

为什么 ⑦ 不能算"写死步态"：train_humanoid_sac.py:1142-1143 只取 left_hip_y 一个关节，
:1155-1158 的触发条件是 `air = not left_contact_prev` —— **左脚已经腾空**才给脉冲。
一个从不抬腿的策略永远拿不到模板。本探针改成：绝对相位时钟 + 双腿 + 摆相内 k=1 全写死。

预注册放行线（跑之前写死，事后不挪）：
  任一 variant 的 2 seed × 1500 帧 det rollout 同时满足
    mean ep_len >= 400
    spacing(|left_x-right_x| 落地时中位) >= 0.10 m
    落地次数/s <= 6.0        （否则模板根本没改变节律）
  否则 D1 判"写死即崩"，不做训练改码。
对照基线 = 同一权重、模板关闭：已知 ep_len 1500 / spacing 0.01-0.073 m / 12 落地每秒。

模板常数来源：人体实测表（LAFAN1+CMU 7 条 clip）换算到本机髋高 0.975 m：
  全周期 2×(1/2.35 步每秒)=0.85 s → 57 帧（dt=0.015）；摆相 = (1-0.63)×0.85 = 0.315 s → 21 帧；
  A = 0.55 沿用 ⑦ 自己实测量级（§30.3），不加大（§39 证据：强推更大 ROM 会掉步幅并在 688 摔）。
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
import mujoco
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

ROOT = "/home/guo/workspace/humanoid_walk/"
POL = ROOT + "best_model_saved/round2/SAC_18/18_8_从头训/best_model_snapshots/best_model_1696000.zip"
HORIZON, SEEDS = 1500, (123, 2026, 7)
DT = 0.015
CYCLE = float(__import__("os").environ.get("D1_CYCLE", 57.0))
SWING = float(__import__("os").environ.get("D1_SWING", 21.0))
A = float(__import__("os").environ.get("D1_A", 0.55))
VARIANTS = {"swing-only k=1": (1.0, "swing"), "swing-only k=0.7": (0.7, "swing"),
            "full-cycle blend k=0.7": (0.7, "always")}


class D1Template(gym.Wrapper):
    """最外层：改写左右 hip_y 的动作，再交给奖励包装器。obs 空间不变。"""

    def __init__(self, env, k, mode):
        super().__init__(env)
        m = env.unwrapped.model
        self.idx = {}
        for side in ("left", "right"):
            self.idx[side] = [i for i in range(m.nu)
                              if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                              == f"{side}_hip_y"][0]
        self.k, self.mode, self.phase = k, mode, 0.0
        self.pulses = []

    def reset(self, **kw):
        self.phase = 0.0
        self.pulses = []
        return self.env.reset(**kw)

    def _pulse(self, side):
        off = 0.0 if side == "left" else 0.5
        p = (self.phase - off) % 1.0
        frac = SWING / CYCLE
        if p < frac:
            return -A * np.sin(np.pi * p / frac)      # 摆相：向前踢
        if self.mode == "swing":
            return None                                # 支撑相：策略全权
        return 0.0                                     # blend 模式：支撑相给中性 0

    def step(self, action):
        self.phase = (self.phase + 1.0 / CYCLE) % 1.0
        a = np.asarray(action, dtype=np.float64).copy()
        for side, i in self.idx.items():
            u = self._pulse(side)
            if u is not None:
                a[i] = float(np.clip((1.0 - self.k) * a[i] + self.k * u, -1.0, 1.0))
        self.pulses.append(a[list(self.idx.values())].copy())
        return self.env.step(a)


def run(model_path, variant, seed, template=True):
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    env = D1Template(w, *variant) if template else w
    model = SAC.load(model_path, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    prev = {"L": True, "R": True}
    lands, spacings, los, lifts, corrs = 0, [], [], [], []
    hip = {s: [] for s in ("left", "right")}
    gauges = []
    steps = 0
    for _ in range(HORIZON):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        d = w.env.unwrapped.data
        for s in ("left", "right"):
            hip[s].append(float(d.qpos[w.env.unwrapped.model.jnt_qposadr[
                mujoco.mj_name2id(w.env.unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_hip_y")]]))
        gl = mujoco.mj_name2id(w.env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
        lifts.append(float(d.geom_xpos[gl][2] - w.env.unwrapped.model.geom_size[gl][0]))
        gauges.append((info.get("double_support_frac", np.nan), info.get("flight_frac", np.nan),
                       info.get("single_frac", np.nan), info.get("speed", np.nan)))
        cL, cR = bool(w.left_contact_prev), bool(w.right_contact_prev)
        for side, c in (("L", cL), ("R", cR)):
            if c and not prev[side]:
                lands += 1
                spacings.append(float(w.foot_diff))
                los.append(min(float(info.get("last_left_stride", 0.0)),
                               float(info.get("last_right_stride", 0.0))))
        prev = {"L": cL, "R": cR}
        if term or trunc:
            break
    h = np.array(hip["left"]), np.array(hip["right"])
    if len(h[0]) > 40:
        corrs.append(float(np.corrcoef(h[0][20:-20], h[1][20:-20])[0, 1]))
    g = np.array(gauges)
    out = dict(variant="+".join(map(str, variant)) if template else "baseline",
               seed=seed, ep_len=steps, landings=lands,
               land_per_s=lands / (steps * DT),
               spacing_p50=float(np.median(spacings)) if spacings else np.nan,
               lo_p50=float(np.median(los)) if los else np.nan,
               lift_p95=float(np.percentile(lifts, 95)),
               ds=float(np.nanmax(g[:, 0])), flight=float(np.nanmax(g[:, 1])),
               v=float(np.nanmean(g[:, 3])), anticorr=corrs[0] if corrs else np.nan)
    del model, env, raw, w
    return out


def main():
    rows = []
    rows.append(run(POL, None, SEEDS[0], template=False))
    for name, v in VARIANTS.items():
        for s in SEEDS:
            rows.append(run(POL, v, s))
    print(f"{'variant':26s} {'seed':>5} {'ep_len':>7} {'land/s':>7} {'spacing':>8} {'lo':>7} "
          f"{'lift95':>7} {'ds':>6} {'flight':>7} {'v':>5} {'anticorr':>9}")
    for r in rows:
        print(f"{r['variant']:26s} {r['seed']:5d} {r['ep_len']:7d} {r['land_per_s']:7.2f} "
              f"{r['spacing_p50']:8.3f} {r['lo_p50']:7.3f} {r['lift_p95']:7.3f} {r['ds']:6.3f} "
              f"{r['flight']:7.3f} {r['v']:5.2f} {r['anticorr']:9.2f}")
    with open("/home/guo/workspace/humanoid_walk/debug_probes/d1_gate_out.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    # 放行判定（预注册线）
    ok = [r for r in rows if r["ep_len"] >= 400 and (r["spacing_p50"] or 0) >= 0.10
          and r["land_per_s"] <= 6.0]
    print(f"\n放行线 ep_len>=400 & spacing>=0.10 & land/s<=6.0 : "
          f"{'PASS ' + str(len(ok)) + ' 个 variant-seed' if ok else 'FAIL（D1 判写死即崩）'}")


if __name__ == "__main__":
    main()
