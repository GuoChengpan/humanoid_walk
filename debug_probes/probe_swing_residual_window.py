"""(丁) 的立项实测：**左髋摆动期加一个前屈脉冲残差**，扫幅度看有没有"A 既甩得开腿、又摔不倒"的窗口。

为什么走这条路（证据链，全部本项目自测）：
 · §17/§25：左髋不缺执行器（pin ctrl=−1 → 114°@10.5rad/s），不缺探索（逐维 std 0.5498），
   现在也不缺钱（§29：走路相每步收入已从 −66% 收到 ≈0/−14%）——**三轮都推不动的是"摔"**；
 · §25.1：det 左髋命令是常数 +0.350±0.042、饱和 0% ⇒ 它被征用成平衡配重，
   策略自己的输出分布里根本没有"前摆"这个动作，所以学不到；
 · ⇒ 结构先验的正确形式不是改奖励、也不是改观测，而是**把左腿的摆动直接塞进动作接口**，
   让策略在"腿真的会摆"的世界里学其余的自由度（经典 hybrid oscillator/残差控制）。

关键问题只有一个：**存活的 A 区间**。det 策略在 A=0 时 rom_L≈2.3°、跑满 1500；
扫上去若立刻摔（像 §25 的 pin/×2 实验那样 32-56 步），(丁A) 就同样不可行，
我必须如实报"没有窗口"而不是硬调一个勉强不倒的 A 上去。

判据（跑之前写死）：可行 = 存在 A 使 `ep_len ≥ 1200/1500`（两种子都）且 `rom_L p90 ≥ 8°`
且 `det ds ≥ 0.05`——三个同时满足才叫"能在此基础上学"。
注意：这个残差会同时加在训练和 eval env（否则评的就不是部署的策略），
所以**eval 分数与历史轮次不可比**，这在本轮预算里已经算过代价。
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
AMP = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
SWING_CYCLE = 17          # 帧：一次摆动 ~0.255s，与实测摆动时长 0.20-0.26s 同量级


def main():
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(PATH, env=env, device="cpu")
    u = env.unwrapped
    gl, gr = None, None
    aidx = [i for i in range(u.model.nu)
            if mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == "left_hip_y"][0]
    print("权重 752K（md5 238fea20）｜left_hip_y 动作索引=%d｜摆动周期=%d 帧\n" % (aidx, SWING_CYCLE))
    print("%-6s %8s %8s %9s %9s %8s %8s %8s %8s" % (
        "A", "ep_len", "分/步", "romL_p50", "romL_p90", "ds", "flight", "single", "reachL"))
    for a0 in AMP:
        rows = []
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            rom, phase, n, tot = [], 0.0, 0, 0.0
            for _ in range(HORIZON):
                act, _ = model.predict(obs, deterministic=True)
                act = np.asarray(act, dtype=np.float64).copy()
                # 该腿在空中 ⇒ 注入半正弦前屈脉冲（负=前屈，实测 pin −1 → 114°）；落地相位归零
                left_air = not _left_down(u, env)
                if left_air and a0 > 0:
                    phase = (phase + 1.0 / SWING_CYCLE) % 1.0
                    act[aidx] = float(np.clip(act[aidx] - a0 * math.sin(math.pi * phase), -1.0, 1.0))
                elif not left_air:
                    phase = 0.0
                obs, r, term, trunc, info = env.step(act)
                n += 1
                tot += float(r)
                rom.append(float(info.get("last_rom_L_deg", 0.0)))
                if term or trunc:
                    break
            rows.append((n, tot / max(n, 1), float(np.percentile(rom, 50)), float(np.percentile(rom, 90)),
                         float(info.get("double_support_frac", np.nan)),
                         float(info.get("flight_frac", np.nan)), float(info.get("single_frac", np.nan)),
                         float(info.get("ev_reach_L_mean", np.nan))))
        m = lambda i: float(np.mean([r[i] for r in rows]))
        ok = "  ← 可行窗口" if (min(r[0] for r in rows) >= 1200 and m(3) >= 8.0 and m(4) >= 0.05) else ""
        print("%-6.2f %8.0f %8.2f %9.2f %9.2f %8.3f %8.3f %8.3f %8.3f%s" % (
            a0, m(0), m(1), m(2), m(3), m(4), m(5), m(6), m(7), ok))
    del model, env, raw


def _left_down(u, env):
    """左脚是否在地面（与 wrapper 同一判据：foot geom 参与任何 contact）。"""
    if env._foot_geom_one is None:
        return False
    gl = env._foot_geom_one[0]
    c = u.data.contact
    for i in range(u.data.ncon):
        if gl in c[i].geom:
            return True
    return False


if __name__ == "__main__":
    main()
