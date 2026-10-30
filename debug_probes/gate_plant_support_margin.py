"""**植物级支撑裕度闸门**（纯物理、不依赖"会不会学"）：把脚球半径放大，同一套开环强制交替摆腿，
看"两脚都往前 + 有双支撑"这个形状能不能撑住一集。

为什么要这张表：`gate_c2` 的 f=0 行实测到——把 hip_y 写死后**左脚自己的步长能到 +0.20~+0.50 m**
（迈步动作做得出来），但 `min(落地间距)` 恒为 −0.23~−0.62、ds 0.00-0.17、flight 0.19-0.43，
每一集 64-117 帧终止，终止帧 z≈0.98。也就是说瓶颈在**支撑几何**：球脚 r=0.075、髋高 0.98
⇒ 前后/侧向可用倾角只有 atan(r/h)≈4.4°（人脚掌 ≈10°），且 D2b 已证 0/45 组 PD 能站住（无静态平衡）。

规模无关的存活定义（关键，防"脚大了站得更高、离绝对阈值 1.0 更远"这个假便利）：
  **t_drop = 躯干从本集初始高度掉过 0.30 m 所需帧数**（1500 帧内没掉到 ⇒ 记 1500）
  同时报 ds、flight、每脚自己的步长（起飞→下次落地）、min 落地间距、前进速度。

预注册放行线（跑之前钉死）：存在 r > 0.075 的某一档同时满足
  P1 t_drop ≥ 1500（= 全程不掉 0.30 m）
  P2 min(落地间距) ≥ +0.10 m  且  两脚各自步长 ≥ +0.05 m
  P3 ds ≥ 0.15 且 flight ≤ 0.15
  P4 r=0.075 档必须不满足 P1 ⇒ 证明是支撑裕度买到的，不是开环本身本来就够
质量不变（MuJoCo 的 body_mass 在编译期定死，运行时改 geom_size 不改质量）⇒ 只有接触几何变。
观测/动作维度也不变 ⇒ 理论上 ⑫ 权重还能 warm start。
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
A = float(os.environ.get("G_A", "0.55"))
RADII = [float(x) for x in os.environ.get("G_R", "0.075,0.105,0.135,0.165").split(",")]
SEEDS = tuple(int(x) for x in os.environ.get("G_SEEDS", "123,2026,7,31").split(","))
DROP = 0.30


class Forced(gym.Wrapper):
    def __init__(self, env, a_amp):
        super().__init__(env)
        m = env.unwrapped.model
        self.idx = {s: [i for i in range(m.nu)
                        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == f"{s}_hip_y"][0]
                    for s in ("left", "right")}
        self.A, self.t = a_amp, 0

    def reset(self, **kw):
        self.t = 0
        return self.env.reset(**kw)

    def step(self, a):
        act = np.asarray(a, dtype=np.float64).copy()
        ph = (self.t % CYCLE) / CYCLE
        for side, off in (("left", 0.0), ("right", 0.5)):
            p = (ph - off) % 1.0
            if p < SWING / CYCLE:
                act[self.idx[side]] = float(np.clip(-self.A * np.sin(np.pi * p / (SWING / CYCLE)), -1, 1))
        self.t += 1
        return self.env.step(act)


def set_radius(env, r):
    """改脚球半径，并把复位高度按同样的量抬上去（否则大球一上来就插进地板）。
    基准存在函数属性上 ⇒ 每档都从**原始 0.075** 起算，不会随循环累加。"""
    m = env.unwrapped.model
    ids = []
    for bname in ("left_foot", "right_foot"):
        bid = m.body(bname).id
        ids += [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid
                and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)]
    if not hasattr(set_radius, "orig_r"):
        set_radius.orig_r = float(m.geom_size[ids[0]][0])
        set_radius.orig_z = float(m.qpos0[2])
    old, z_base = set_radius.orig_r, set_radius.orig_z
    for g in ids:
        m.geom_size[g][0] = r
    dr = r - old
    # 只把复位高度抬 dr，让**球底离地距离在所有档位一致**（球心太高会自由落体，太低会初始穿透）
    u = env.unwrapped
    m.qpos0[2] = z_base + dr
    if hasattr(u, "init_qpos"):
        u.init_qpos[2] = z_base + dr
    return len(ids), dr


class TimeRemap(gym.Wrapper):
    """把终止条件从"gymnasium 的绝对带 z∉(1.0,2.0)"换成"**相对自身站高塌陷 DROP**"。
    必要原因：脚球放大 → 站高整体上移 → 绝对阈值自动变松，会把"活着"这件事白送给大脚档。"""

    def __init__(self, env, z_ref_fn):
        super().__init__(env)
        self.z_ref_fn, self.z0 = z_ref_fn, None

    def reset(self, **kw):
        o = self.env.reset(**kw)
        self.z0 = self.z_ref_fn()
        return o

    def step(self, a):
        o, r, term, trunc, info = self.env.step(a)
        info["z_rel"] = self.z_ref_fn() - self.z0
        if self.z_ref_fn() < self.z0 - DROP:
            term, info["fell"] = True, True
        return o, r, term, trunc, info


def run(model, r):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=False)
    ngeom, dr = set_radius(w, r)
    zr = lambda: float(w.unwrapped.data.qpos[2])
    env = TimeRemap(Forced(w, A), zr)
    m, d = w.unwrapped.model, w.unwrapped.data
    lid, rid = m.body("left_foot").id, m.body("right_foot").id
    gl = [g for g in range(m.ngeom) if m.geom_bodyid[g] == lid]
    gr = [g for g in range(m.ngeom) if m.geom_bodyid[g] == rid]
    out = []
    for sd in SEEDS:
        obs, _ = env.reset(seed=sd)
        z0 = float(d.qpos[2])
        t = 0
        t_drop = np.nan
        step = {"L": [], "R": []}
        takeoff, prev = {"L": None, "R": None}, {"L": True, "R": True}
        lo_vals = []
        last_info = None
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, rr, term, trunc, info = env.step(np.asarray(act, dtype=np.float64))
            t += 1
            last_info = info
            if np.isnan(t_drop) and d.qpos[2] < z0 - DROP:
                t_drop = t
            lx, rx = d.body("left_foot").xpos[0], d.body("right_foot").xpos[0]
            for side, x, ids in (("L", lx, gl), ("R", rx, gr)):
                hit = any((d.geom_xpos[g][2] - m.geom_size[g][0]) < 0.02 for g in ids)
                if hit and not prev[side]:
                    lo_vals.append(x - (rx if side == "L" else lx))
                    if takeoff[side] is not None:
                        step[side].append(x - takeoff[side])
                if not hit and prev[side]:
                    takeoff[side] = x
                prev[side] = hit
            if term or trunc:
                break   # term 现在由 TimeRemap 的"相对塌陷 DROP"决定，不再是绝对 z<1.0
        if np.isnan(t_drop):
            t_drop = FR
        med = lambda s: float(np.mean(s)) if s else np.nan
        out.append(dict(t_drop=min(t_drop, t), len=t, term=bool(term), z0=z0,
                        lo=min(lo_vals) if lo_vals else np.nan, sL=med(step["L"]), sR=med(step["R"]),
                        ds=last_info["double_support_frac"], fl=last_info["flight_frac"],
                        v=last_info["speed"], lands=w.total_land_count))
    return out, ngeom, dr


def main():
    model = SAC.load(MODEL)
    print(f"权重 {MODEL}；两条 hip_y 开环写死（k=1，周期 {CYCLE:.0f}、摆相 {SWING:.0f}、A={A}）；"
          f"存活判据=躯干相对初始高度掉过 {DROP} m 的帧数")
    print(f"{'r 脚球':>7} {'抬升':>8} {'倾角裕度':>8} {'撑住不塌(=1500)':>15} {'塌陷帧中位':>10} "
          f"{'min落地间距':>10} {'步长L':>7} {'步长R':>7} {'ds':>6} {'fl':>6} {'v':>6}")
    res = {}
    for r in RADII:
        o, ngeom, dr = run(model, r)
        td = np.array([x["t_drop"] for x in o])
        res[r] = dict(td=float(np.median(td)), full=float(np.mean(td >= FR)),
                      lo=float(np.nanmedian([x["lo"] for x in o])),
                      sL=float(np.nanmedian([x["sL"] for x in o])),
                      sR=float(np.nanmedian([x["sR"] for x in o])),
                      ds=float(np.median([x["ds"] for x in o])),
                      fl=float(np.median([x["fl"] for x in o])),
                      v=float(np.median([x["v"] for x in o])),
                      margin=float(np.degrees(np.arctan(r / max(np.nanmedian([x["z0"] for x in o]), 1e-3)))),
                      land=float(np.median([x["lands"] / max(x["len"] * DT, 1e-6) for x in o])))
        R = res[r]
        print(f"{r:7.3f} {dr:+8.3f} {R['margin']:7.1f}° {R['full']:15.2f} {R['td']:10.0f} {R['lo']:10.3f} "
              f"{R['sL']:7.3f} {R['sR']:7.3f} {R['ds']:6.3f} {R['fl']:6.3f} {R['v']:6.2f}")
        print(f"        （单集 len={sorted(x['len'] for x in o)}，z0={sorted(round(x['z0'],3) for x in o)}，"
              f"落地数={sorted(x['lands'] for x in o)}）")
    base = res.get(RADII[0], {})
    print("\n=== 放行线 P1-P4 ===")
    passed = []
    for r, R in res.items():
        if r == RADII[0]:
            continue
        p1 = R["td"] >= FR
        p2 = (np.isfinite(R["lo"]) and R["lo"] >= 0.10 and np.isfinite(R["sL"]) and R["sL"] >= 0.05
              and np.isfinite(R["sR"]) and R["sR"] >= 0.05)
        p3 = R["ds"] >= 0.15 and R["fl"] <= 0.15
        p4 = base.get("td", FR) < FR
        if p1 and p2 and p3 and p4:
            passed.append(r)
        print(f"  r={r:.3f}: P1={p1}(t_drop={R['td']:.0f}) P2={p2}(lo={R['lo']:+.3f} "
              f"步长L={R['sL']:+.3f}/R={R['sR']:+.3f}) P3={p3}(ds={R['ds']:.3f} fl={R['fl']:.3f}) P4={p4}")
    print(f"\n⇒ 通过档 {len(passed)} 个" + (f"：{passed} ⇒ 支撑裕度是真瓶颈，改植物值得做"
          if passed else " ⇒ 放大球脚不足以打开可行性，(C)/(植物改动) 的这条路按 P1-P3 判否"))


if __name__ == "__main__":
    sys.exit(main())
