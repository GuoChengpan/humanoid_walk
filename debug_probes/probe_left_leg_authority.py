"""左腿到底是"没激励"还是"执行不了"——执行余量 + 反事实驱动探针。

背景：708K 的判决说 alternate 给左腿落地间距留了 **32 分/集 per cm** 的活胡萝卜，
但 `ev_spacing_L` 700K 死在 0.076、`ev_rom_L` 从 12.2° 掉到 6.6°。
要么奖励改法无效（能力受限），要么策略根本没去拿（局部最优/探索受限）。

三段测量（全部只读，不动模型文件、不动训练进程）：
 A 执行余量：左右髋 y / 膝的 ctrl 是否逼近 ctrlrange 边界、actuator_force 是否到力矩上限、
   摆动期实测角速度上限（L vs R），以及"每单位命令换到多少角度行程"（增益）。
 B 放大命令：把左腿 hip_y/knee 的动作幅度 ×2（仍 clip 在范围内），看 reach/间距/腾空是否响应
   —— 如果响应且没摔，说明"没命令"是策略选择，不是物理限制。
 C 反事实驱动：让左腿**复制右腿的动作、滞后半个腿周期**（±两个符号都试），
   看能否得到 duty 上升/腾空下降的节律，以及会不会立刻摔。
"""
import gc
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

MODEL = os.environ.get("PROBE_MODEL", "/home/guo/workspace/humanoid_walk/best_model/best_model.zip")
SEEDS = (123, 2026, 7)
HORIZON = 1500


def build():
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    model = raw.unwrapped.model
    patch_mirror_symmetry(model)
    return raw


def idx_of(m):
    """actuator 索引：按关节名找 hip_y / knee（左右各一）。"""
    out = {}
    for a in range(m.nu):
        jid = m.actuator_trnid[a][0]
        name = m.joint(jid).name
        out[name] = a
    return out


def run(path, seed, mode="base", lag=13, sign=1.0, amp=2.0):
    raw = build()
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    m = raw.unwrapped.model
    ai = idx_of(m)
    A_LY, A_RY = ai["left_hip_y"], ai["right_hip_y"]
    A_LK, A_RK = ai["left_knee"], ai["right_knee"]
    A_LX, A_RX = ai["left_hip_x"], ai["right_hip_x"]
    A_LZ, A_RZ = ai["left_hip_z"], ai["right_hip_z"]
    qadr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in
            ["left_hip_y", "right_hip_y", "left_knee", "right_knee"]}
    vadr = {n: int(m.jnt_dofadr[m.joint(n).id]) for n in
            ["left_hip_y", "right_hip_y", "left_knee", "right_knee"]}
    gear = {n: float(m.actuator_gear[idx_of(m)[n]][0]) for n in
            ["left_hip_y", "right_hip_y", "left_knee", "right_knee"]}
    crange = {n: tuple(m.actuator_ctrlrange[idx_of(m)[n]]) for n in
              ["left_hip_y", "right_hip_y", "left_knee", "right_knee"]}

    model = SAC.load(path, env=env, device="cpu")
    obs, _ = env.reset(seed=seed)
    data = raw.unwrapped.data
    rec, act_hist = [], []
    ret = 0.0
    fell = False
    n = 0
    for t in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        a = np.asarray(act, dtype=np.float64).copy()
        if mode == "amp":
            a[[A_LY, A_LK]] = np.clip(a[[A_LY, A_LK]] * amp, -1.0, 1.0)
        elif mode == "copy" and len(act_hist) > lag:
            src = act_hist[-lag]
            a[[A_LY, A_LK, A_LX, A_LZ]] = sign * src[[A_RY, A_RK, A_RX, A_RZ]]
        act_hist.append(np.asarray(act, dtype=np.float64).copy())
        obs, rew, term, trunc, info = env.step(a)
        n += 1
        ret += float(rew)
        rec.append(dict(
            aLY=float(a[A_LY]), aRY=float(a[A_RY]), aLK=float(a[A_LK]), aRK=float(a[A_RK]),
            fLY=float(data.actuator_force[A_LY]), fRY=float(data.actuator_force[A_RY]),
            vLY=float(data.qvel[vadr["left_hip_y"]]), vRY=float(data.qvel[vadr["right_hip_y"]]),
            qLY=float(data.qpos[qadr["left_hip_y"]]), qRY=float(data.qpos[qadr["right_hip_y"]]),
            vLK=float(data.qvel[vadr["left_knee"]]), vRK=float(data.qvel[vadr["right_knee"]]),
            flight=float(info["flight_frac"]), ds=float(info["double_support_frac"]),
            single=float(info["single_frac"]), duty=float(info["duty_sum"]),
            v=float(info["speed"]), gate=float(info["walk_gate"]),
            reachL=float(info.get("ev_reach_L_mean", np.nan)),
            reachR=float(info.get("ev_reach_R_mean", np.nan)),
            spL=float(info.get("ev_spacing_L_mean", np.nan)),
            spR=float(info.get("ev_spacing_R_mean", np.nan)),
            romL=float(info.get("ev_rom_L_mean", np.nan)), romR=float(info.get("ev_rom_R_mean", np.nan)),
            liftL=float(info.get("ev_lift_L_mean", np.nan)),
            swL=float(info.get("mean_left_swing", np.nan)), swR=float(info.get("mean_right_swing", np.nan)),
            term=bool(term)))
        if term:
            fell = True
            break
    del model, env, raw
    gc.collect()
    return rec, ret, fell, n, gear, crange


def g(rec, k):
    return np.array([r[k] for r in rec], dtype=np.float64)


def main():
    print(f"模型: {MODEL}\n")
    rec, ret, fell, n, gear, crange = run(MODEL, SEEDS[0], "base")
    print("=== A 执行余量（左 vs 右）===")
    for nm, ak, fk, vk in (("hip_y", "aLY", "fLY", "vLY"),):
        a, f, v = g(rec, ak), g(rec, fk), g(rec, vk)
        ar, fr, vr = g(rec, "aRY"), g(rec, "fRY"), g(rec, "vRY")
        lo, hi = crange["left_hip_y"]
        lim = gear["left_hip_y"]
        print(f"  {nm} 左: |ctrl| 均={np.abs(a).mean():.3f} p95={np.percentile(np.abs(a),95):.3f} "
              f"贴边(|a|>0.95)占{100*np.mean(np.abs(a)>0.95):.0f}% | 力矩 均={np.abs(f).mean():.1f} "
              f"p99={np.percentile(np.abs(f),99):.1f} 上限≈{lim:.0f} → 贴力矩上限占{100*np.mean(np.abs(f)>0.9*lim):.0f}%")
        print(f"  {nm} 右: |ctrl| 均={np.abs(ar).mean():.3f} p95={np.percentile(np.abs(ar),95):.3f} "
              f"贴边占{100*np.mean(np.abs(ar)>0.95):.0f}% | 力矩 均={np.abs(fr).mean():.1f} "
              f"p99={np.percentile(np.abs(fr),99):.1f} 上限≈{gear['right_hip_y']:.0f}")
        print(f"  摆动期 |角速度| p95: 左={np.percentile(np.abs(v),95):.2f} rad/s  右={np.percentile(np.abs(vr),95):.2f} rad/s"
              f"  (比值 右/左={np.percentile(np.abs(vr),95)/max(np.percentile(np.abs(v),95),1e-9):.2f})")
    romL, romR = np.nanmean(g(rec, "romL")), np.nanmean(g(rec, "romR"))
    aL, aR = np.abs(g(rec, "aLY")).mean(), np.abs(g(rec, "aRY")).mean()
    print(f"  行程/命令增益: 左 {romL:.1f}° / |a|={aL:.3f} → {romL/max(aL,1e-9):.1f}°每单位命令 | "
          f"右 {romR:.1f}° / |a|={aR:.3f} → {romR/max(aR,1e-9):.1f}°每单位命令")
    print(f"  基线: ret={ret:.0f} 步数={n} 摔={fell} flight={np.mean(g(rec,'flight')):.3f} "
          f"duty_sum={np.mean(g(rec,'duty')):.3f} v={np.mean(g(rec,'v')):.2f} "
          f"reachL={np.nanmean(g(rec,'reachL')):+.3f} spL={np.nanmean(g(rec,'spL')):+.3f}")

    print("\n=== B 把左腿命令幅度 ×2（其他不变）===")
    for seed in SEEDS:
        rec2, ret2, fell2, n2, _, _ = run(MODEL, seed, "amp")
        print(f"  seed{seed}: 步数={n2} 摔={'是' if fell2 else '否'} ret={ret2:.0f} "
              f"flight={np.mean(g(rec2,'flight')):.3f} duty={np.mean(g(rec2,'duty')):.3f} "
              f"v={np.mean(g(rec2,'v')):.2f} reachL={np.nanmean(g(rec2,'reachL')):+.3f} "
              f"spL={np.nanmean(g(rec2,'spL')):+.3f} romL={np.nanmean(g(rec2,'romL')):.1f}° "
              f"liftL={np.nanmean(g(rec2,'liftL')):.3f}")

    print("\n=== C 左腿复制右腿动作（滞后半周期）===")
    for sign in (1.0, -1.0):
        for lag in (13, 25):
            r = []
            for seed in SEEDS[:2]:
                rec3, ret3, fell3, n3, _, _ = run(MODEL, seed, "copy", lag=lag, sign=sign)
                r.append((seed, n3, fell3, ret3, np.mean(g(rec3, "flight")), np.mean(g(rec3, "duty")),
                          np.mean(g(rec3, "v")), np.nanmean(g(rec3, "reachL")), np.mean(g(rec3, "swL")),
                          np.mean(g(rec3, "swR"))))
            print(f"  sign={sign:+.0f} lag={lag}: " + " | ".join(
                f"s{sd} {n}步{'摔' if f else ''} ret{rt:.0f} 腾空{fl:.2f} duty{du:.2f} v{v:.2f} "
                f"reachL{reL:+.2f} 摆L{sl:.0f}/摆R{sr:.0f}" for sd, n, f, rt, fl, du, v, reL, sl, sr in r))


if __name__ == "__main__":
    main()
