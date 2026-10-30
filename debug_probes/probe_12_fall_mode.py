"""⑫ 终局盆地（4M）的**摔法归因**：渲染看到的是"一脚在前一脚在后、各自小碎步前进一小段就摔"，
这个脚本把它量化成三件事，为的是决定 (C) 物理辅助该加在哪一维，而不是再猜：

1. 前后**偏移是否恒定**（"剪刀式跟队"）：episode 内 mean(lx−rx) 与其波动。
2. 每只脚**自己有没有前进**：per-foot 的 take-off→下一次落地 位移（真正的 step length），
   以及落地间距 (that foot x − other foot x)。
3. **摔的那一刻**是什么模式：前倾/后仰/侧倒/坐倒，用"重心相对支撑点的倾角"对比
   **本机可用的支撑裕度**（球脚 r=0.075、髋高 0.975 ⇒ 前后倾角裕度只有 atan(r/h)≈4.4°，
   而人脚掌半长 ≈0.16/0.90 ⇒ ≈10°）。这是 D2b"没有静态平衡"的具体数值形态。

只读 rollout，不改任何文件、不训练、不加载除 ⑫ 自己权重之外的东西。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODEL = os.environ.get("PROBE_MODEL", "best_model/best_model.zip")
N_EP = int(os.environ.get("N_EP", "60"))
FR = 1500
DT = 0.015
CONTACT_Z = 0.035          # 球脚 geom 底缘判据（与 _foot_contact_state 同一族，只用于本探针的支撑点统计）


def build(k_forced=None):
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    if k_forced is not None:
        w.set_osc_k(k_forced)
    return w


def run(model, k_forced, n_ep=N_EP):
    env = build(k_forced)
    m, d = env.unwrapped.model, env.unwrapped.data
    lid, rid = m.body("left_foot").id, m.body("right_foot").id
    # 脚是**球** geom（humanoid.xml 里半径 0.075、装在 njoint=0 的 foot body 上）
    # ⇒ 点接触，支撑点的 x 就是球心 x（严格成立），底缘 z = 球心 z − r
    sph = int(getattr(__import__("mujoco"), "mjtGeom", None).mjGEOM_SPHERE) \
        if hasattr(__import__("mujoco"), "mjtGeom") else 2
    lgeom = [g for g in range(m.ngeom) if m.geom_bodyid[g] == lid and int(m.geom_type[g]) == sph]
    rgeom = [g for g in range(m.ngeom) if m.geom_bodyid[g] == rid and int(m.geom_type[g]) == sph]
    assert len(lgeom) == 1 and len(rgeom) == 1, (lgeom, rgeom)
    print(f"  脚球 geom 半径 L={m.geom_size[lgeom[0]][0]:.3f} R={m.geom_size[rgeom[0]][0]:.3f} m")
    rows = []
    for ei in range(n_ep):
        obs, _ = env.reset(seed=7000 + ei)
        hist = dict(off=[], lx=[], rx=[], tx=[], tz=[], step_L=[], step_R=[], land_spac_L=[], land_spac_R=[])
        prev_landed = {"L": None, "R": None}      # 该脚上次落地的 x
        prev_contact = {"L": True, "R": True}
        last_takeoff = {"L": None, "R": None}
        fall = None
        t = 0
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(np.asarray(act, dtype=np.float64))
            t += 1
            tx, ty, tz = d.body("torso").xpos
            lx, rx = d.body("left_foot").xpos[0], d.body("right_foot").xpos[0]
            l_bot = d.geom_xpos[lgeom[0]][2] - m.geom_size[lgeom[0]][0]   # 球：半径在 size[0]，不是 size[2]
            r_bot = d.geom_xpos[rgeom[0]][2] - m.geom_size[rgeom[0]][0]
            lc, rc = l_bot < CONTACT_Z, r_bot < CONTACT_Z
            hist["off"].append(lx - rx)
            hist["lx"].append(lx); hist["rx"].append(rx); hist["tx"].append(tx); hist["tz"].append(tz)
            for side, x, cont in (("L", lx, lc), ("R", rx, rc)):
                if cont and not prev_contact[side]:                    # 落地瞬间
                    if last_takeoff[side] is not None:
                        hist[f"step_{side}"].append(x - last_takeoff[side])
                    other = rx if side == "L" else lx
                    hist[f"land_spac_{side}"].append(x - other)
                    prev_landed[side] = x
                if not cont and prev_contact[side]:
                    last_takeoff[side] = x
                prev_contact[side] = cont
            if term or trunc:
                q = d.qpos[3:7]
                pitch, roll = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[1:3]
                sup = [x for x, c in ((lx, lc), (rx, rc)) if c]
                rows.append(dict(len=t, ret=float(np.sum([])) if False else None, k=k_forced,
                                 off_mean=float(np.mean(hist["off"])), off_std=float(np.std(hist["off"])),
                                 advance=(hist["tx"][-1] - hist["tx"][0]),
                                 stepL=float(np.mean(hist["step_L"])) if hist["step_L"] else np.nan,
                                 stepR=float(np.mean(hist["step_R"])) if hist["step_R"] else np.nan,
                                 spL=float(np.mean(hist["land_spac_L"])) if hist["land_spac_L"] else np.nan,
                                 spR=float(np.mean(hist["land_spac_R"])) if hist["land_spac_R"] else np.nan,
                                 nL=len(hist["land_spac_L"]), nR=len(hist["land_spac_R"]),
                                 pitch=float(np.degrees(pitch)), roll=float(np.degrees(roll)),
                                 z=float(tz), sup_x=float(np.mean(sup)) if sup else np.nan,
                                 torso_x=float(tx), terminated=bool(term)))
                break
    return rows


def describe(rows, lab):
    both = [r for r in rows if np.isfinite(r["spL"]) and np.isfinite(r["spR"])
            and min(r["spL"], r["spR"]) > 0.0]
    tap = [r for r in rows if np.isfinite(r["spL"]) and np.isfinite(r["spR"])
           and min(r["spL"], r["spR"]) <= 0.0]
    print(f"\n### {lab}：{len(rows)} 集，两脚落点都为正 {len(both)} 个（{100*len(both)/max(len(rows),1):.0f}%）")
    for g, nm in ((both, "两脚都往前"), (tap, "剪刀式(一侧为负)")):
        if not g:
            print(f"  {nm:12s} 无样本")
            continue
        f = lambda k: float(np.median([r[k] for r in g if np.isfinite(r[k])]))
        print(f"  {nm:12s} n={len(g):3d} ep_len={f('len'):5.0f} 每步位移 L={f('stepL'):+.3f} R={f('stepR'):+.3f} m"
              f" 落地间距 L={f('spL'):+.3f} R={f('spR'):+.3f}  偏移 mean={f('off_mean'):+.3f}±{f('off_std'):.3f}"
              f"  前进={f('advance'):+.2f} m")
        # 摔法分类（只在 terminated 的集里有意义）
        tm = [r for r in g if r["terminated"]]
        if tm:
            lean = [(r["torso_x"] - r["sup_x"]) / max(r["z"], 1e-3) for r in tm if np.isfinite(r["sup_x"])]
            ang = np.degrees(np.array(lean)) if lean else np.array([np.nan])
            roll_a = np.abs(np.array([r["roll"] for r in tm]))
            pit = np.array([r["pitch"] for r in tm])
            mode = ("前倾主导" if np.nanmedian(pit) < -20 else "侧倾主导" if np.nanmedian(roll_a) > 25
                    else "后仰" if np.nanmedian(pit) > 20 else "坐倒/高度塌")
            print(f"    摔倒帧：pitch 中位={np.nanmedian(pit):+6.1f}° roll={np.nanmedian(roll_a):5.1f}° "
                  f"躯干高={np.nanmedian([r['z'] for r in tm]):.2f} m "
                  f"重心-支撑点倾角={np.nanmedian(ang):+6.1f}°（本机前后裕度只有 ±4.4°） ⇒ 模式判读 **{mode}**")


def main():
    model = SAC.load(MODEL)
    print(f"权重 {MODEL}（⑫ 4M 终局）  渲染看到的盆地在这里量化")
    describe(run(model, None), "部署控制器（模板 k=1，模板开）")
    describe(run(model, 0.0), "同一权重、模板关（k=0）")
    print("\n注：step_X = 该脚 起飞→下一次落地 的位移（真正的自己前进量）；"
          "spX = 落地时该脚相对另一脚的超前量（奖励里 lo 用的就是它的 min）。")


if __name__ == "__main__":
    main()
