"""(C) 的定生死闸门：**竖直抗重力辅助**能否让"两脚都往前"的形状活满一集。

为什么是竖直而不是加宽脚掌：`probe_12_fall_mode.py` 实测 ⑫ 终局的终止帧是
z 末 1.05-1.11 m、|roll| 1.3-7.0°、|pitch| 2-12.6°，而 humanoid.xml 的终止条件是
`qpos[2] ∉ (1.0, 2.0)` ⇒ 死法是"越蹲越低、穿过下沿出带"，不是倾覆。所以缺的是支撑期的
**垂直承载**，不是支撑宽度。

实现：每步把 `data.xfrc_applied[torso, 2] = f · mass · |g|`（作用在躯干 free joint 的 body 上，
世界系 z 分量），f 从 0 到 0.6。观测与动作维度**完全不变** ⇒ ⑫ 的权重可以 warm start。

闸门（跑之前钉死，四条同时成立才算 (C) 有料）：
  G1 中位 ep_len ≥ 1500（即一集内根本不终止）
  G2 ≥50% 的集满足 min(左,右 落点间距) > +0.10 且该值的中位 > +0.10
  G3 落地每秒 ≤ 4.0（人体 1.5-2.9，碎步 12）
  G4 f=0 对照必须仍然达不到 G1（否则说明辅助没起作用、是权重自己够了）
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

MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
FS = [0.0, 0.15, 0.30, 0.45, 0.60]
SEEDS = list(range(11, 21))
FR = 1500
DT = 0.015


def run(model, f, n_ep=len(SEEDS)):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    m, d = env.unwrapped.model, env.unwrapped.data
    tid = m.body("torso").id
    mass = float(m.body_subtreemass[tid])
    grav = float(-m.opt.gravity[2])
    rows = []
    for ei in range(n_ep):
        obs, _ = env.reset(seed=SEEDS[ei])
        t = 0
        lo_last = np.nan
        while True:
            if f > 0:                       # 每步重写：reset 会清掉 xfrc_applied
                d.xfrc_applied[:] = 0.0
                d.xfrc_applied[tid, 2] = f * mass * grav
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(np.asarray(act, dtype=np.float64))
            t += 1
            if info["last_left_stride"] or info["last_right_stride"]:
                lo_last = min(info["last_left_stride"], info["last_right_stride"])
            if term or trunc:
                rows.append(dict(len=t, term=bool(term), lo=lo_last,
                                 ds=info["double_support_frac"], fl=info["flight_frac"],
                                 z=float(d.qpos[2]), v=info["speed"],
                                 land_s=env.total_land_count / max(t * DT, 1e-6)))
                break
    return rows, mass, grav


def main():
    model = SAC.load(MODEL)
    print(f"权重 {MODEL}；竖直辅助 f·m·g 作用在 torso（mass/grav 见下表）\n")
    print(f"{'f':>5} {'mass':>6} {'ep_len 中位':>11} {'≥1500 的比例':>12} {'lo 中位':>8} {'lo>0.10 比例':>11} "
          f"{'落地/s':>7} {'ds':>6} {'flight':>7} {'v':>6}")
    out = {}
    for f in FS:
        rows, mass, grav = run(model, f)
        L = np.array([r["len"] for r in rows], dtype=float)
        lo = np.array([r["lo"] for r in rows], dtype=float)
        lo_ok = np.mean(lo > 0.10)
        med_lo = float(np.nanmedian(lo))
        out[f] = dict(len=float(np.median(L)), full=float(np.mean(L >= FR)), lo=med_lo,
                     lo_ok=float(lo_ok), land=float(np.median([r["land_s"] for r in rows])),
                     ds=float(np.median([r["ds"] for r in rows])))
        print(f"{f:5.2f} {mass:6.2f} {out[f]['len']:11.0f} {out[f]['full']:12.2f} {med_lo:8.3f} "
              f"{lo_ok:11.2f} {out[f]['land']:7.2f} {out[f]['ds']:6.3f} "
              f"{np.median([r['fl'] for r in rows]):7.3f} {np.median([r['v'] for r in rows]):6.2f}")
    print("\n=== 闸门判定（跑前钉死的 G1-G4）===")
    best = max((f for f in FS if f > 0), key=lambda f: out[f]["len"])
    g1 = out[best]["len"] >= FR
    g2 = out[best]["lo_ok"] >= 0.5 and out[best]["lo"] > 0.10
    g3 = out[best]["land"] <= 4.0
    g4 = out[0.0]["len"] < FR
    for nm, ok, val in (("G1 中位 ep_len≥1500", g1, f"f={best} ⇒ {out[best]['len']:.0f}"),
                        ("G2 ≥50% 集 min 落点间距>0.10", g2, f"比例={out[best]['lo_ok']:.2f} lo中位={out[best]['lo']:.3f}"),
                        ("G3 落地/s≤4.0", g3, f"{out[best]['land']:.2f}"),
                        ("G4 f=0 对照不满足 G1", g4, f"f=0 ⇒ ep_len={out[0.0]['len']:.0f}")):
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}  实测={val}")
    print(f"\n注：本闸门只回答'辅助能不能把形状托住'。'撤走辅助后是否保留'必须靠训练回答（"
          f"§25/§29/本轮三次都栽在这一条上），不能从这张表推断。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
