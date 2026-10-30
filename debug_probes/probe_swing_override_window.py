"""(丁) 的第二块试金石：**覆盖式**驱动左髋（网络对该关节只有部分或零控制权），而不是加性残差。

为什么从"加"改成"覆盖"：`probe_swing_residual_window.py` 实测 A≤0.25 的加性脉冲被策略
**完全抵消**（rom_L p90 2.94→3.69°，而分/步从 1.24 **涨到** 1.30 —— 它把注入变现了却没迈步），
A=0.40 才出现 ds 0.069 / rom 9.4° 但 812 步就摔。⇒ 只要网络还能在这一维上"往回扳"，
任何结构先验都会被那个平衡解吸收。真正的先验必须**剥夺网络对该关节的部分控制权**。

三种模式（全部用**未训练的现有 752K 策略**跑 det，只问"物理上有没有活路"）：
  PURE  ：a_L = −A·sin(πφ)      网络对左髋零控制权（φ 由接触时钟推进，落地即归零）
  MIX   ：a_L = (1−k)·a_net + k·脉冲   部分让渡
  CONST ：a_L = −A 常数前屈      对照——区分"是振荡杀它"还是"是前屈姿态杀它"
判据（跑之前写死）：存在配置使 **两种子 ep_len ≥ 1200** 且 **rom_L p90 ≥ 8°** 且 **det ds ≥ 0.05**
⇒ (丁) 可行、值得开一轮。若全部摔，那就如实报"接口层同样推不动"，不再往这个项目上加机制。
"""
import math
import sys

import mujoco
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

PATH = "/home/guo/workspace/humanoid_walk/best_model/tmp_model_data/last_TMP/best_model.zip"
HORIZON = 1500
SEEDS = (123, 2026)
CYCLE = 17               # 帧：一次摆动 ≈0.255s
CONFIGS = [("MIX", 0.30, 0.55), ("MIX", 0.35, 0.55), ("MIX", 0.40, 0.55),
           ("MIX", 0.45, 0.55), ("MIX", 0.50, 0.55), ("MIX", 0.60, 0.55),
           ("PURE", 0.0, 0.55), ("CONST", 0.0, 0.20)]


def left_down(u, env, gl):
    """**必须与 wrapper 同一判据**：左脚任意 geom × 与 world(body 0) 接触。
    早先这里只取一个 geom 且不限对身，导致"空中"窗口偏长、脉冲打得偏久，
    量出的 rom_L p90 18.3° 比线上 wrapper 的 11.2° 虚高（§30.3 已按此更正）。"""
    m, d = u.model, u.data
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if m.geom_bodyid[g1] != 0 and m.geom_bodyid[g2] != 0:
            continue
        if g1 in gl or g2 in gl:
            return True
    return False


def one(model, env, u, aidx, gl, mode, k, A, seed):
    obs, _ = env.reset(seed=seed)
    phase, n, tot = 0.0, 0, 0.0
    rom = []
    ds = fl = sg = rl = 0.0
    for _ in range(HORIZON):
        act, _ = model.predict(obs, deterministic=True)
        act = np.asarray(act, dtype=np.float64).copy()
        air = not left_down(u, env, gl)
        if air:
            phase = (phase + 1.0 / CYCLE) % 1.0
        pulse = -A * math.sin(math.pi * phase) if air else 0.0
        if mode == "NONE":
            new = act[aidx]                              # 真基线：完全不介入
        elif mode == "CONST":
            new = -A
        elif mode == "PURE":
            new = pulse
        else:
            new = (1 - k) * act[aidx] + k * pulse
        act[aidx] = float(np.clip(new, -1.0, 1.0))
        obs, r, term, trunc, info = env.step(act)
        n += 1
        tot += float(r)
        rom.append(float(info.get("last_rom_L_deg", 0.0)))
        ds = float(info.get("double_support_frac", ds))
        fl = float(info.get("flight_frac", fl))
        sg = float(info.get("single_frac", sg))
        rl = float(info.get("ev_reach_L_mean", rl))
        if term or trunc:
            break
    return (n, tot / max(n, 1), float(np.percentile(rom, 90)), ds, fl, sg, rl)


def main():
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(PATH, env=env, device="cpu")
    u = env.unwrapped
    aidx = [i for i in range(u.model.nu)
            if mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == "left_hip_y"][0]
    obs, _ = env.reset(seed=1)
    for _ in range(5):
        env.step(np.zeros(u.model.nu))
    lfid = u.model.body("left_foot").id
    gl = frozenset(g for g in range(u.model.nu * 0 + u.model.ngeom)
                   if u.model.geom_bodyid[g] == lfid)      # 与 wrapper 一致：左脚全部 geom
    print("权重 752K ｜ left_hip_y 动作 idx=%d ｜ 左脚 geom 集=%s ｜ 判据 ep_len≥1200 且 rom_p90≥8° 且 ds≥0.05\n"
          % (aidx, sorted(gl)))
    print("%-6s %5s %5s %8s %9s %8s %8s %8s %8s %8s" % (
        "mode", "k", "A", "ep_len", "rom_p90", "分/步", "ds", "flight", "single", "reachL"))
    base = [one(model, env, u, aidx, gl, "NONE", 0.0, 0.0, s) for s in SEEDS]
    print("%-6s %5s %5s %8.0f %9.2f %8.2f %8.3f %8.3f %8.3f %8.3f   ← 基线(不加任何驱动)" % (
        "none", 0, 0, float(np.mean([b[0] for b in base])), float(np.mean([b[2] for b in base])),
        float(np.mean([b[1] for b in base])), float(np.mean([b[3] for b in base])),
        float(np.mean([b[4] for b in base])), float(np.mean([b[5] for b in base])),
        float(np.mean([b[6] for b in base]))))
    for mode, k, A in CONFIGS:
        rs = [one(model, env, u, aidx, gl, mode, k, A, s) for s in SEEDS]
        m = lambda i: float(np.mean([x[i] for x in rs]))
        ok = "  ← 可行窗口" if (min(x[0] for x in rs) >= 1200 and m(2) >= 8.0
                                and m(3) >= 0.05) else ""
        print("%-6s %5.2f %5.2f %8.0f %9.2f %8.2f %8.3f %8.3f %8.3f %8.3f%s" % (
            mode, k, A, m(0), m(2), m(1), m(3), m(4), m(5), m(6), ok))
    del model, env, raw


if __name__ == "__main__":
    main()
