"""左腿到底卡在哪：命令的相位分配 + 真·执行天花板。

上一支探针（probe_left_leg_authority）给出：左髋 |ctrl| 均 0.331、0% 饱和、力矩远未到上限，
但摆动期 |角速度| p95 只有 3.0 rad/s（右 12.6），命令 ×2 后 rom 只从 4.3°→5.6°，
且"让左腿复制右腿动作"每种符号/滞后都在 40-56 步内摔 → 平衡策略是绕着这个非对称弹跳建的。

这里要分清三种可能：
 (1) 命令没打在摆动期（相位错配）→ 要改的是相位奖励；
 (2) 命令打在摆动期但方向与运动相反（刹车）→ 协调问题；
 (3) 全力命令也只能摆这么快（真天花板）→ 再改奖励也没用，只能降低对左腿摆幅的需求。
做法：基线 rollout 按接触相位统计 ctrl；然后做两组"开环天花板"实验——
 在左腿摆动期把 ctrl 钉成 ±1（屈髋方向两个符号都试），量 0.2s 内能达到多少角行程与角速度。
只读，不动模型/进程。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODEL = os.environ.get("PROBE_MODEL", "/home/guo/workspace/humanoid_walk/best_model/best_model.zip")
SEED = 123
HORIZON = 1500


def make():
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    return raw, env


def rollout(mode="base", ctrl_side=0.0, n=600):
    raw, env = make()
    m = raw.unwrapped.model
    data = raw.unwrapped.data
    ai = {m.joint(m.actuator_trnid[a][0]).name: a for a in range(m.nu)}
    A_LY, A_RY = ai["left_hip_y"], ai["right_hip_y"]
    qL, qR = int(m.jnt_qposadr[m.joint("left_hip_y").id]), int(m.jnt_qposadr[m.joint("right_hip_y").id])
    vL, vR = int(m.jnt_dofadr[m.joint("left_hip_y").id]), int(m.jnt_dofadr[m.joint("right_hip_y").id])
    model = SAC.load(MODEL, env=env, device="cpu")
    obs, _ = env.reset(seed=SEED)
    R = {k: [] for k in ("aL", "aR", "wL", "wR", "qL", "qR", "swingL", "stanceL",
                         "swingR", "fall", "t")}
    for t in range(n):
        act, _ = model.predict(obs, deterministic=True)
        a = np.asarray(act, dtype=np.float64).copy()
        if mode == "maxcmd":
            # 只在左腿摆动期把它的命令钉死；摆动判据用上一帧的接触状态
            if not getattr(env, "left_contact_prev", True):
                a[A_LY] = ctrl_side
        a = np.clip(a, -1.0, 1.0)
        obs, rew, term, trunc, info = env.step(a)
        swingL = not bool(getattr(env, "left_contact_prev", True))
        swingR = not bool(getattr(env, "right_contact_prev", True))
        R["aL"].append(float(a[A_LY])); R["aR"].append(float(a[A_RY]))
        R["wL"].append(float(data.qvel[vL])); R["wR"].append(float(data.qvel[vR]))
        R["qL"].append(float(data.qpos[qL])); R["qR"].append(float(data.qpos[qR]))
        R["swingL"].append(swingL); R["stanceL"].append(not swingL)
        R["swingR"].append(swingR); R["t"].append(t)
        if term:
            R["fall"].append(t)
            break
    del model, env, raw
    return {k: np.array(v) for k, v in R.items()}, (len(R["fall"]) == 0), t


def s(v, mask, f=np.mean):
    return f(v[mask]) if mask.any() else float("nan")


def main():
    rec, alive, t = rollout("base")
    sw, st = rec["swingL"], rec["stanceL"]
    print(f"=== (1) 基线：命令的相位分配（{t+1} 步，摔={'否' if alive else '是'}）===")
    print(f"  左髋 |ctrl|：摆动期均={np.abs(rec['aL'][sw]).mean():.3f} p95={np.percentile(np.abs(rec['aL'][sw]),95):.3f}"
          f" | 支撑期均={np.abs(rec['aL'][st]).mean():.3f}")
    print(f"  右髋 |ctrl|：摆动期均={np.abs(rec['aR'][sw]).mean():.3f} | 支撑期均={np.abs(rec['aR'][st]).mean():.3f}")
    sameL = np.mean(np.sign(rec["aL"][sw]) == np.sign(rec["wL"][sw]))
    sameR = np.mean(np.sign(rec["aR"][sw]) == np.sign(rec["wR"][sw]))
    print(f"  摆动期命令与角速度同号(=驱动)占比：左={100*sameL:.0f}% 右={100*sameR:.0f}%"
          f"（<50% 说明一半命令在踩刹车）")
    print(f"  摆动期 |角速度| p95：左={np.percentile(np.abs(rec['wL'][sw]),95):.2f} "
          f"右={np.percentile(np.abs(rec['wR'][sw]),95):.2f} rad/s")
    print(f"  摆动期角度行程 p50：左={np.percentile(np.abs(np.diff(rec['qL'][sw].astype(float))),95):.2f} rad"
          f"（同一相位里最大的单步变化）")

    print("\n=== (2) 开环天花板：左腿摆动期把 ctrl 钉成 ±1（其余照旧）===")
    for side in (+1.0, -1.0):
        r2, alive2, t2 = rollout("maxcmd", ctrl_side=side)
        sw2 = r2["swingL"]
        print(f"  ctrl={side:+.0f}: 撑到 {t2+1} 步 摔={'否' if alive2 else '是'} | "
              f"摆动期 |w| p95={np.percentile(np.abs(r2['wL'][sw2]),95):.2f} rad/s | "
              f"左髋角范围=[{r2['qL'].min():.2f},{r2['qL'].max():.2f}] rad "
              f"({np.degrees(r2['qL'].max()-r2['qL'].min()):.0f}°) | "
              f"右髋 |w| p95={np.percentile(np.abs(r2['wR']),95):.2f}")
    print("\n  对照：humanoid.xml hip_y ctrlrange=-1..1，gear 上限对应 ±120 N·m；"
          "若钉满命令后左腿摆速仍远低于右腿，则是**质量/惯量+协调**限制而非指令限制")


if __name__ == "__main__":
    main()
