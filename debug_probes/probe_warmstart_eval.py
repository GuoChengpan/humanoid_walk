"""评估 warm-start 模型在当前 reward 函数下的表现（确定性 + 随机）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

import train_humanoid_sac as T

def run_eval(model, deterministic, n_eps=3, seed=42):
    env = gym.make("Humanoid-v4", max_episode_steps=1500)
    env = T.WalkRewardWrapper(env)
    env.reset(seed=seed)
    results = []
    for ep in range(n_eps):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(action)
            ep_rew += rew
            steps += 1
            done = term or trunc
            if steps >= 1500:
                break
        results.append((ep_rew, steps))
        key = ['phase_reward','position_swap_reward','stride_bonus','lead_credit','alternate_penalty',
               'last_left_stride','last_right_stride','support_sym_penalty','stride_balance_penalty']
        vals = {k: round(info.get(k, 0.0), 4) for k in key}
        print(f'  ep{ep}: rew={ep_rew:9.1f}  len={steps:5d}  {vals}')
    return results

if __name__ == '__main__':
    model = SAC.load("./best_model/best_model.zip")
    print('模型 num_timesteps =', model.num_timesteps)
    print('\n=== 确定性 eval（当前 reward 函数）===')
    run_eval(model, deterministic=True, n_eps=3)
    print('\n=== 随机 eval（当前 reward 函数）===')
    run_eval(model, deterministic=False, n_eps=3)
