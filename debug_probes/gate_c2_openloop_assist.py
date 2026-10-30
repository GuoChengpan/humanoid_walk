"""(C) 的正确预飞测试（第二版）：**开环强制交替步 × 竖直辅助**，问"这个形状到底托不托得住"。

为什么换设计：第一版 `gate_c_vertical_assist.py` 把辅助加在**没用辅助训出来的** ⑫ 策略上，
动力学一变策略就崩（ep_len 346→48、v 翻负）——那是实现必然，不是 (C) 的否证。我撤回那张表作为判据。

本版只测可行性，不测"会不会学"：控制器其余 15 维仍用 ⑫ 权重，但**两条 hip_y 被我按绝对反相时钟
覆盖**（k=1 override，摆相窗口 22/60 帧，幅值 A 可调）——即"把交替步写死"，其余交给网络。
逐档扫 (f=竖直卸载比例, A=摆幅)，看是否存在一档能活满 1500 帧且两脚都真的往前。

跑之前钉死的闸门（四条同时成立才值得为 (C) 花一次训练）：
  H1 ep_len ≥ 1500（= 一集内不终止）
  H2 min(左,右 落地间距) ≥ +0.10 m  且  每只脚自己的步长（起飞→下次落地）≥ +0.05 m
  H3 落地每秒 ≤ 4.0
  H4 同一 A 下 f=0 必须不满足 H1 ⇒ 证明是辅助买到的支撑，不是本来就行
另附实现自检：f=1.0 应当接近失重（z 不再单调下坠）。若不为此，说明 xfrc 用错了。
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
DT, FR = 0.015, 1500
CYCLE, SWING = 60.0, 22.0
SEEDS = (123, 2026, 7, 31)
GRID = [(f, A) for f in (0.0, 0.20, 0.35, 0.50, 0.70, 1.00) for A in (0.35, 0.55, 0.75)]


class ForcedAlternating(gym.Wrapper):
    """把左右 hip_y 的动作**覆盖**成绝对反相时钟脉冲；其余维度原样交给下层（= ⑫ 网络）。"""

    def __init__(self, env, A, f):
        super().__init__(env)
        m = env.unwrapped.model
        self.idx = {s: [i for i in range(m.nu)
                        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == f"{s}_hip_y"][0]
                    for s in ("left", "right")}
        self.tid = m.body("torso").id
        self.mass = float(m.body_subtreemass[self.tid])
        self.grav = float(-m.opt.gravity[2])
        self.A, self.f, self.t = A, f, 0
        self.steps_seen = []

    def reset(self, **kw):
        self.t = 0
        o = self.env.reset(**kw)
        self.env.unwrapped.data.xfrc_applied[:] = 0.0
        return o

    def step(self, a):
        act = np.asarray(a, dtype=np.float64).copy()
        ph = (self.t % CYCLE) / CYCLE
        for side, off in (("left", 0.0), ("right", 0.5)):
            p = (ph - off) % 1.0
            if p < SWING / CYCLE:
                u = p / (SWING / CYCLE)
                act[self.idx[side]] = float(np.clip(-self.A * np.sin(np.pi * u), -1.0, 1.0))
        d = self.env.unwrapped.data
        if self.f > 0:
            d.xfrc_applied[:] = 0.0
            d.xfrc_applied[self.tid, 2] = self.f * self.mass * self.grav
        self.t += 1
        return self.env.step(act)


def run(model, f, A):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    env = ForcedAlternating(w, A, f)
    m, d = w.unwrapped.model, w.unwrapped.data
    lid, rid = m.body("left_foot").id, m.body("right_foot").id
    gl = [g for g in range(m.ngeom) if m.geom_bodyid[g] == lid]
    gr = [g for g in range(m.ngeom) if m.geom_bodyid[g] == rid]
    out = []
    for sd in SEEDS:
        obs, _ = env.reset(seed=sd)
        t = 0
        step_adv = {"L": [], "R": []}
        takeoff = {"L": None, "R": None}
        prev = {"L": True, "R": True}
        lo_vals = []
        zmin = 9.9
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(np.asarray(act, dtype=np.float64))
            t += 1
            lx, rx = d.body("left_foot").xpos[0], d.body("right_foot").xpos[0]
            zmin = min(zmin, float(d.qpos[2]))
            for side, x, ids in (("L", lx, gl), ("R", rx, gr)):
                hit = any((d.geom_xpos[g][2] - m.geom_size[g][0]) < 0.02 for g in ids)
                if hit and not prev[side]:
                    other = rx if side == "L" else lx
                    lo_vals.append(x - other)
                    if takeoff[side] is not None:
                        step_adv[side].append(x - takeoff[side])
                if not hit and prev[side]:
                    takeoff[side] = x
                prev[side] = hit
            if term or trunc:
                lo = min(lo_vals) if lo_vals else np.nan
                sL = np.mean(step_adv["L"]) if step_adv["L"] else np.nan
                sR = np.mean(step_adv["R"]) if step_adv["R"] else np.nan
                out.append(dict(len=t, term=bool(term),
                                land_s=w.total_land_count / max(t * DT, 1e-6),
                                lo=lo, sL=sL, sR=sR, zmin=zmin,
                                ds=info["double_support_frac"], fl=info["flight_frac"],
                                v=info["speed"]))
                break
    return out


def main():
    model = SAC.load(MODEL)
    print(f"权重 {MODEL}（⑫ 4M）；两条 hip_y 被我覆盖成绝对反相时钟（k=1 开环），其余 15 维走网络")
    print(f"周期 {CYCLE:.0f} 帧、摆相 {SWING:.0f} 帧、种 {SEEDS}\n")
    print(f"{'f':>5} {'A':>5} {'ep_len':>7} {'终止?':>6} {'min落地间距':>11} {'步长L':>7} {'步长R':>7} "
          f"{'落地/s':>7} {'ds':>6} {'fl':>6} {'v':>6} {'z 最低':>7}")
    res = {}
    for f, A in GRID:
        o = run(model, f, A)
        med = lambda k: float(np.median([r[k] for r in o if np.isfinite(r[k])]))
        res[(f, A)] = dict(len=med("len"), lo=med("lo"), sL=med("sL"), sR=med("sR"),
                           land=med("land_s"), term=float(np.mean([r["term"] for r in o])),
                           zmin=med("zmin"))
        r = res[(f, A)]
        print(f"{f:5.2f} {A:5.2f} {r['len']:7.0f} {r['term']:6.2f} {r['lo']:11.3f} {r['sL']:7.3f} "
              f"{r['sR']:7.3f} {r['land']:7.2f} {med('ds'):6.3f} {med('fl'):6.3f} {med('v'):6.2f} {r['zmin']:7.2f}")
    print("\n=== 闸门 H1-H4 ===")
    ok = []
    for (f, A), r in res.items():
        if f == 0:
            continue
        h1, h2, h3 = r["len"] >= FR, (np.isfinite(r["lo"]) and r["lo"] >= 0.10
                                      and np.isfinite(r["sL"]) and r["sL"] >= 0.05
                                      and np.isfinite(r["sR"]) and r["sR"] >= 0.05), r["land"] <= 4.0
        base = res.get((0.0, A))
        h4 = base is not None and base["len"] < FR
        if h1 and h2 and h3 and h4:
            ok.append((f, A, h1, h2, h3, h4))
        print(f"  f={f:.2f} A={A:.2f}: H1={h1}(ep_len={r['len']:.0f}) H2={h2}(lo={r['lo']:+.3f} "
              f"步长L={r['sL']:+.3f}/R={r['sR']:+.3f}) H3={h3}(land/s={r['land']:.2f}) H4={h4}")
    print(f"\n⇒ 存在通过 (f,A) 档 {len(ok)} 个" + (f"：{[(f, a) for f, a, *_ in ok]}" if ok else " ⇒ (C) 的这条可行性不成立"))
    r1 = res.get((1.00, 0.55))
    print(f"实现自检 f=1.00/A=0.55：z 最低={r1['zmin']:.2f} m（若辅助写对了，接近失重 ⇒ z 不应一路掉到 1.0 以下）"
          f"，ep_len={r1['len']:.0f}")


if __name__ == "__main__":
    sys.exit(main())
