"""(e) 非对称初始状态的**存活网格**：把左髋在 reset 后强行搬到"已经迈在前面"的状态，
然后让**当前 det 策略**从这个状态跑，看它能不能活着、能不能真甩出行程、脚有没有穿地。

为什么这是 (甲) 唯一剩下的候选：
 · 逐维 std 实测左髋 0.5498（全体 0.448-0.635）→ **不是探索塌了**，定向加噪无从下手；
 · 强行抬/压左髋 std 都能甩出 p90=14-22° 行程，但**摔率 100%**（`probe_left_hip_exploration.py`）；
 · 终止判据是 `is_healthy`（根高 < 初始−1.0m）→ 已经宽松到"真摔才终止"，没有放松余量；
 · §17 的 pin/copy 与上面同向：**左髋被征用为平衡稳定翼，问题在状态空间覆盖，不在奖励也不在执行器**。
    → 唯一没试过的杠杆是**让经验分布里出现"左腿在前面"的状态**，而不是逼策略自己撞进去。

判读标准（三个都要，缺一个就是废的）：
 1) 存活：det 从该初始状态跑，ep_len 不能塌到几百以下——否则灌进 buffer 的是崩盘数据
    （§3.1 死亡螺旋的老配方，也是 1472K 那份 warm-start 现在不能用的原因）；
 2) 有效行程：左腿摆动行程 p90 要显著高于基线 2.9°；
 3) 状态物理合法：初始那只脚不能穿地（geom 底面 z − 半径 ≥ 0），否则求解器会把它弹飞。
"""
import sys

import mujoco
import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

PATH = "/home/guo/workspace/humanoid_walk/best_model/best_model.zip"
HORIZON = 1500
SEEDS = (123, 2026)
DELTAS = [0.0, -0.10, -0.20, -0.30, -0.45, 0.20]     # 左髋 qpos 增量(rad)，负=往前迈


def main():
    raw = gym.make("Humanoid-v4", max_episode_steps=HORIZON)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0)
    model = SAC.load(PATH, env=env, device="cpu")
    u = env.unwrapped
    jid = u.model.joint("left_hip_y").id
    adr = u.model.jnt_qposadr[jid]
    # 左脚接触 geom：body('left_foot') 下 z 向半径最大的那个 geom（球状接触体）
    lfid = u.model.body("left_foot").id
    cands = [g for g in range(u.model.ngeom) if u.model.geom_bodyid[g] == lfid]
    left_foot_gid = max(cands, key=lambda g: u.model.geom_size[g][2])
    left_foot_r = float(u.model.geom_size[left_foot_gid][2])
    print("权重 %s ｜ 左髋 qpos 地址=%d ｜ 左脚 geom=%s r=%.3f" % (
        PATH.split("/")[-1], adr, mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_GEOM, left_foot_gid),
        left_foot_r))
    print("\n%-8s %6s %6s %8s %8s %8s %9s %9s" % (
        "Δ左髋", "集1长", "集2长", "存活率", "romL_p90", "reachL", "脚底z(min)", "穿地?"))
    for d in DELTAS:
        lens, roms, reachs, footz = [], [], [], []
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            u.data.qpos[adr] += d
            u.data.qvel[u.model.jnt_dofadr[jid]] = 0.0     # 抹掉该关节速度，别让注入变成一脚踹
            mujoco.mj_forward(u.model, u.data)
            # 注入后的状态合法性：左脚底离地高度（geom 心 z − 半径）。负=穿地=非法状态
            # （不借 wrapper 的 _foot_geom_one——它是首次 step 之后才懒初始化的）
            fz0 = float(u.data.geom_xpos[left_foot_gid][2] - left_foot_r)
            footz.append(fz0)
            obs = u._get_obs()                      # 用改完的状态重算观测，策略才看见新状态
            rom, rch = [], []
            n = 0
            for _ in range(HORIZON):
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                n += 1
                rom.append(float(info.get("last_rom_L_deg", 0.0)))
                if "ev_reach_L_mean" in info:
                    rch.append(float(info["ev_reach_L_mean"]))
                if term or trunc:
                    break
            lens.append(n)
            roms.append(np.percentile(rom, 90) if rom else 0.0)
            reachs.append(rch[-1] if rch else float("nan"))
        survive = np.mean([l >= HORIZON * 0.9 for l in lens])
        print("%-8.2f %6d %6d %8.0f%% %8.1f %9.3f %9.3f %9s" % (
            d, lens[0], lens[1] if len(lens) > 1 else 0, 100 * survive,
            float(np.mean(roms)), float(np.mean(reachs)), float(np.mean(footz)),
            "是" if np.mean(footz) < -0.001 else "否"))
    del model, env, raw


if __name__ == "__main__":
    main()
