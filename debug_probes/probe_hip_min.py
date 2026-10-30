"""实测左右髋 hip_y 的最小值（前屈程度），判断 0.15 阈值下 min() 绑定哪条腿"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import train_humanoid_sac as T

env_id = "Humanoid-v4"


def main():
    model = SAC.load("./best_model/best_model.zip")  # 现在是 17_6_d (1472K)
    print('模型 num_timesteps =', model.num_timesteps)

    # 用训练脚本同款的 joint 索引方式
    env = gym.make(env_id)
    m = env.unwrapped.model
    hip_idx = {}
    for name in ['left_hip_y', 'right_hip_y']:
        jid = m.joint(name).id
        hip_idx[name] = int(m.jnt_qposadr[jid])
    print('hip_y qpos 索引:', hip_idx)

    for ep in range(3):
        env = gym.make(env_id, max_episode_steps=1500)
        env = T.WalkRewardWrapper(env)
        obs, _ = env.reset(seed=42+ep)
        done = False; n = 0
        lmin, lmax, lmean = [], [], []
        rmin, rmax, rmean = [], [], []
        while not done and n < 1500:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            q = env.unwrapped.data.qpos.copy()
            lh = q[hip_idx["left_hip_y"]]
            rh = q[hip_idx["right_hip_y"]]
            lmin.append(lh); rmin.append(rh)
            n += 1; done = term or trunc
        print(f'ep{ep} (len={n}):')
        print(f'  左 hip_y: min={min(lmin):.3f} max={max(lmin):.3f}   → max(0,0.15-min)={max(0.0,0.15-min(lmin)):.3f}')
        print(f'  右 hip_y: min={min(rmin):.3f} max={max(rmin):.3f}   → max(0,0.15-min)={max(0.0,0.15-min(rmin)):.3f}')
        lf = max(0.0, 0.15-min(lmin)); rf = max(0.0, 0.15-min(rmin))
        print(f'  min(left_forward,right_forward) = {min(lf,rf):.3f}  ← 绑定{"左" if lf<rf else "右"}腿')


if __name__ == '__main__':
    main()
