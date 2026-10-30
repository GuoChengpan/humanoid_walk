"""摆动驱动奖励 A/B 干跑：SWING_DRIVE_WEIGHT = 0 vs 2.0
同一 warm-start(608K)、同种子，唯一变量 = 摆动驱动奖励开关。
测：左髋摆动期角速度是否开始爬（机制响应性）——这是新奖励的直接目标参数。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, gymnasium as gym
import torch.nn as nn
from stable_baselines3 import SAC
from mirror_map import MirrorMap, patch_mirror_symmetry
from train_humanoid_sac import WalkRewardWrapper, MirrorEqSAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_env(w):
    e = gym.make("Humanoid-v4", max_episode_steps=1500)
    patch_mirror_symmetry(e.unwrapped.model)
    return WalkRewardWrapper(e, swing_drive_weight=w)

def swing_vel(model, seed=1):
    """确定性一集：左/右髋摆动期平均|hip_y vel|（rad/s）"""
    e = build_env(0.0)
    mm = e.unwrapped.model
    ji = {mm.joint(i).name: i for i in range(mm.njnt)}
    vd = {n: int(mm.jnt_dofadr[ji[n]]) for n in ("left_hip_y","right_hip_y")}
    obs, _ = e.reset(seed=seed)
    stats = {"L": [], "R": []}
    cur = {"L": None, "R": None}
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = e.step(a)
        d = e.unwrapped.data
        lz, rz = d.body("left_foot").xpos[2], d.body("right_foot").xpos[2]
        cL, cR = lz >= 0.08, rz >= 0.08
        vL, vR = abs(d.qvel[vd["left_hip_y"]]), abs(d.qvel[vd["right_hip_y"]])
        if not cL and cR:
            if cur["L"] is None: cur["L"] = []
            cur["L"].append(vL)
        elif cur["L"] is not None:
            stats["L"].append(np.mean(cur["L"])); cur["L"] = None
        if not cR and cL:
            if cur["R"] is None: cur["R"] = []
            cur["R"].append(vR)
        elif cur["R"] is not None:
            stats["R"].append(np.mean(cur["R"])); cur["R"] = None
        done = term or trunc
    e.close()
    return float(np.median(stats["L"])), float(np.median(stats["R"]))

old = SAC.load(os.path.join(ROOT, "best_model/best_model.zip"), device="cpu")
print(f"warm-start num_timesteps={old.num_timesteps}")
vL0, vR0 = swing_vel(old)
print(f"基线摆动期髋角速度: L={vL0:.2f} / R={vR0:.2f} rad/s")

for w in (0.0, 2.0):
    env = build_env(w)
    mm = MirrorMap(env.unwrapped.model)
    m = MirrorEqSAC('MlpPolicy', env, buffer_size=30_000, batch_size=256, train_freq=256,
                    gradient_steps=8, learning_starts=256, ent_coef=0.05, target_entropy=-8.5,
                    policy_kwargs={'net_arch': [512, 256], 'activation_fn': nn.Tanh},
                    seed=0, verbose=0, mirror_map=mm, eq_mix_lambda=0.0)
    m.policy.load_state_dict(old.policy.state_dict())
    m.learning_starts = 256
    m.learn(total_timesteps=4000, reset_num_timesteps=False, log_interval=None)
    vL, vR = swing_vel(m)
    sd = m.logger.name_to_value.get("gait/swing_drive_reward")
    print(f"W={w}: 训后 L={vL:.2f} / R={vR:.2f} rad/s (摆动驱动奖励最新 {sd if sd is None else round(float(sd),4)})")
    del m

print("\n判读: W=2.0 若 L 明显高于 W=0 与基线 → 机制对奖励响应（可行性 ✓）")
