"""固定 seed 42/43/44 逐集对比：416K 原始 vs 硬压std，确定性+随机"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import train_humanoid_sac as T

env_id = "Humanoid-v4"
n_train_envs = 8
seed = 42


def run_eval_seeded(model, det, seeds):
    out = []
    for s in seeds:
        env = gym.make(env_id, max_episode_steps=1500)
        env = T.WalkRewardWrapper(env)
        obs, _ = env.reset(seed=s)
        done=False; n=0; rsum=0.0
        while not done and n<1500:
            a,_ = model.predict(obs, deterministic=det)
            obs,r,term,trunc,info = env.step(a)
            rsum+=r; n+=1; done = term or trunc
        out.append((s, n, round(rsum,1)))
    return out


def main():
    train_vec_env = SubprocVecEnv([T.make_env(env_id, i, seed) for i in range(n_train_envs)])
    train_vec_env = VecMonitor(train_vec_env)

    old_model = SAC.load("./best_model/tmp_model_data/best_model_416000.zip", env=train_vec_env)
    model = SAC('MlpPolicy', train_vec_env, learning_rate=1e-4, buffer_size=1_500_000,
                batch_size=1024, tau=0.005, gamma=0.99, train_freq=512, gradient_steps=1024,
                ent_coef=0.05, target_entropy=-8.5, policy_kwargs={'net_arch':[512,256], 'activation_fn': nn.Tanh}, verbose=0)
    model.policy.load_state_dict(old_model.policy.state_dict())

    seeds = [42, 43, 44]
    print('== 416K 原始模型 ==')
    print('  确定性:', run_eval_seeded(model, True, seeds))
    print('  随机:  ', run_eval_seeded(model, False, seeds))

    with th.no_grad():
        model.policy.actor.log_std.weight.data.zero_()
        model.policy.actor.log_std.bias.data.fill_(th.log(th.tensor(0.08 / 0.4)))

    print('== 硬压后（统一 std=0.08 env）==')
    print('  确定性:', run_eval_seeded(model, True, seeds))
    print('  随机:  ', run_eval_seeded(model, False, seeds))

    del old_model, model


if __name__ == '__main__':
    main()
