"""按训练真实路径冒烟：8 路 SubprocVecEnv + VecMonitor + warm-start 模型，跑 400 步。"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from train_humanoid_sac import make_env


def main():
    env_id, n_train_envs, seed = "Humanoid-v4", 8, 42
    train_vec_env = VecMonitor(
        SubprocVecEnv([make_env(env_id, i, seed) for i in range(n_train_envs)]))
    policy_kwargs = {'net_arch': [512, 256], 'activation_fn': nn.Tanh}

    model = SAC("MlpPolicy", train_vec_env, learning_rate=1e-5,
                buffer_size=1_500_000, batch_size=1024, tau=0.01, gamma=0.99,
                train_freq=512, gradient_steps=512, ent_coef=0.1, target_entropy=-8.5,
                policy_kwargs=policy_kwargs, verbose=0)
    print("SAC 构造完成 obs=%s act=%s" % (model.observation_space, model.action_space), flush=True)

    old = SAC.load("best_model/best_model.zip", env=train_vec_env)
    sd = old.policy.state_dict()
    del old.replay_buffer
    del old
    gc.collect()
    model.policy.load_state_dict(sd)
    del sd
    gc.collect()
    print("warm-start 权重已加载", flush=True)

    # 预填充路径（训练 1077 行同款）
    obs = train_vec_env.reset()
    stall_sum, nonfinite, stall_frames = 0.0, 0, 0
    for i in range(400):
        acts = np.array([model.predict(obs[k], deterministic=False)[0]
                         for k in range(n_train_envs)])
        obs, rewards, dones, infos = train_vec_env.step(acts)
        if not np.all(np.isfinite(rewards)):
            nonfinite += 1
        for inf in infos:
            stall_sum += float(inf.get("stall_penalty", 0.0))
            if float(inf.get("stall_penalty", 0.0)) > 0:
                stall_frames += 1

    print()
    print("400 步 × 8 env = %d 次步进，无异常" % (400 * n_train_envs), flush=True)
    print("reward 非有限帧数 = %d（应为 0）" % nonfinite, flush=True)
    print("stall_penalty 累计 = %.1f  stall>0 帧数 = %d" % (stall_sum, stall_frames), flush=True)
    print("info 含 stall 字段 = stall_penalty:%s stall_max:%s"
          % ("stall_penalty" in infos[0], "stall_max" in infos[0]), flush=True)
    print("stall_max 各 env 当前值 = %s" % [inf.get("stall_max") for inf in infos], flush=True)
    train_vec_env.close()
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
