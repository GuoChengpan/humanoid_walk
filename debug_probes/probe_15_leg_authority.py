"""腿级权限测量：写死的摆动脉冲，到底是被"承重"吃掉了，还是根本没送到关节上。

背景（用户的提问）：⑪ 那轮是右腿摆、左腿不动；⑬⑭ 这轮反过来变成左腿摆、右腿基本不动。
"已经把左右轮流写死了，为什么还不实现"——有两种完全不同的解释，必须先分开：
  (a) 命令送到了，但那侧脚此时正在承重（球脚+无踝 ⇒ 承重的腿被 hip_y 推只会变成对地面的力矩
      和躯干扰动，不会变成迈步）⇒ 物理层问题，写死关节 ≠ 写死接触时序；
  (b) 我们对某一侧的覆盖其实没生效（索引/缩放/二次裁剪）⇒ 实现层 bug。

每侧各量：
  窗口占比        应 = swing_frac = 0.367（否则模板调度就不对）
  P(脚在地面 | 窗口内)  ← "白打率"，(a) 的直接证据
  关节角位移      窗口内的 |Δhip_y|（deg）vs 窗口外，以及 data.ctrl 实际值 vs 期望脉冲
                  ← 区分 (a)/(b) 的关键：命令对不对得上、关节动不动
  脚位移          每个窗口的 foot_x 前移量（mm）与抬脚高度
  每步前移        起飞→下次落地的位移（真正的"这只脚自己走了多远"）
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

MODEL = os.environ.get("PROBE_MODEL", max([f"best_model/best_model_snapshots/{n}"
        for n in os.listdir("best_model/best_model_snapshots") if n.endswith(".zip")],
        key=os.path.getmtime))
N_EP = int(os.environ.get("G_EP", "6"))
FR = 1500
SWING_FRAC = 22.0 / 60.0


def run(path):
    model = SAC.load(path)
    raw = gym.make("Humanoid-v4", max_episode_steps=FR)
    patch_mirror_symmetry(raw.unwrapped.model)
    w = WalkRewardWrapper(raw, swing_drive_weight=2.0, osc_enable=True)
    m, d = w.unwrapped.model, w.unwrapped.data
    aidx, jidx, foot = {}, {}, {}
    for side in ("left", "right"):
        aidx[side] = [i for i in range(m.nu)
                      if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == f"{side}_hip_y"][0]
        jidx[side] = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_y")])
        bid = m.body(f"{side}_foot").id
        foot[side] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid
                      and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)][0]
    out = {s: dict(win=0, win_contact=0, dwin=[], dnon=[], footadv=[], lift=[], ctrl_err=[],
                   pulse=[], takeoff=None, land=[]
                   ) for s in ("left", "right")}
    prev_c = {s: True for s in ("left", "right")}
    per_leg_steps = {s: [] for s in ("left", "right")}
    n_land = {s: 0 for s in ("left", "right")}
    for i in range(N_EP):
        obs, _ = w.reset(seed=8100 + i)
        for t in range(FR):
            act, _ = model.predict(obs, deterministic=True)
            act_in = np.asarray(act, dtype=np.float64)
            # 期望脉冲（与 wrapper 同一个时钟）：摆相窗口内 = −A·sin(πu)
            ph = (w.total_steps % w.gait_period) / float(w.gait_period)
            inwin = {}
            for s in ("left", "right"):
                off = 0.0 if s == "left" else 0.5
                p = (ph - off) % 1.0
                inwin[s] = p < SWING_FRAC
            obs, r, term, trunc, info = w.step(act_in)
            for s in ("left", "right"):
                o = out[s]
                c = (d.geom_xpos[foot[s]][2] - m.geom_size[foot[s]][0]) < 0.02
                ang = np.degrees(d.qpos[jidx[s]])
                if inwin[s]:
                    o["win"] += 1
                    o["win_contact"] += int(c)
                    o["dwin"].append(d.qpos[jidx[s]])
                    o["ctrl_err"].append(float(d.ctrl[aidx[s]]))
                    o["pulse"].append(float(act_in[aidx[s]]))
                else:
                    o["dnon"].append(d.qpos[jidx[s]])
                if c and not prev_c[s]:
                    n_land[s] += 1
                    if o["takeoff"] is not None:
                        o["footadv"].append(d.body(f"{s}_foot").xpos[0] - o["takeoff"])
                if (not c) and prev_c[s]:
                    o["takeoff"] = d.body(f"{s}_foot").xpos[0]
                    o["lift"] = o.get("lift", [])
                prev_c[s] = c
            if term or trunc:
                break
    return out, n_land, m, aidx


def main():
    print(f"权重（只读）: {MODEL}")
    out, n_land, m, aidx = run(MODEL)
    rng = {s: (np.ptp(out[s]["dwin"]) if out[s]["dwin"] else 0) for s in out}
    print(f"\n{'侧':6s} {'窗口占比':>9} {'P(踩地|窗口)':>13} {'窗内髋角范围':>13} {'窗外髋角范围':>13} "
          f"{'ctrl 范围':>10} {'输入动作范围':>13} {'落地次数':>9} {'自己前移/步':>12}")
    for s in ("left", "right"):
        o = out[s]
        dw = np.array(o["dwin"]) if o["dwin"] else np.zeros(2)
        dn = np.array(o["dnon"]) if o["dnon"] else np.zeros(2)
        ct = np.array(o["ctrl_err"]) if o["ctrl_err"] else np.zeros(2)
        pu = np.array(o["pulse"]) if o["pulse"] else np.zeros(2)
        adv = np.mean(o["footadv"]) if o["footadv"] else np.nan
        tot = max(sum(o["win"] for _ in [0]), 1)
        print(f"{s:6s} {o['win'] / max(len(dw) + len(dn), 1):9.3f} "
              f"{o['win_contact'] / max(o['win'], 1):13.3f} "
              f"{np.degrees(np.ptp(dw)):13.1f} {np.degrees(np.ptp(dn)):13.1f} "
              f"{ct.min():+6.2f}..{ct.max():+.2f} {pu.min():+7.2f}..{pu.max():+.2f} "
              f"{n_land[s]:9d} {adv * 1000:+12.0f} mm")
    print(f"\n  髋 y 关节限位: {np.degrees(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,'left_hip_y')])} deg")
    print("  读法：`P(踩地|窗口)` 高 ⇒ 脉冲落在承重期（物理层）；`窗内髋角范围`≈0 而 `输入动作范围`≈±0.15 ⇒ "
          "命令没变成关节运动；若两者都正常但脚不前移 ⇒ 是触地/负载问题而不是接线问题。")


if __name__ == "__main__":
    main()
