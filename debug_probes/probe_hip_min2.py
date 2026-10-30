"""用能走的 1472000 模型测行走时的左右 hip_y 最小值，判断 min() 绑定哪条腿"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import train_humanoid_sac as T

env_id = "Humanoid-v4"


def main():
    model = SAC.load("best_model/tmp_model_data/17_6/17_6_d_的最终版_加positions_swap/best_model.zip")
    print('模型 num_timesteps =', model.num_timesteps)

    for ep in range(5):
        env = gym.make(env_id, max_episode_steps=1500)
        env = T.WalkRewardWrapper(env)
        obs, _ = env.reset(seed=50+ep)
        done = False; n = 0; rsum = 0.0
        lvals, rvals = [], []
        while not done and n < 1500:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            q = env.unwrapped.data.qpos.copy()
            lvals.append(q[16]); rvals.append(q[12])
            rsum += r; n += 1; done = term or trunc
        lmin, rmin = min(lvals), min(rvals)
        lf = max(0.0, 0.15-lmin); rf = max(0.0, 0.15-rmin)
        bind = '左' if lf < rf else '右'
        print(f'ep{ep} len={n:4d} rew={rsum:8.1f} | 左hip min={lmin:6.3f} max={max(lvals):6.3f} | '
              f'右hip min={rmin:6.3f} max={max(rvals):6.3f} | left_fw={lf:5.3f} right_fw={rf:5.3f} → min()绑定{bind}腿')


if __name__ == '__main__':
    main()
