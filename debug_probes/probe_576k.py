"""探测 576000 (best_model.zip) 的随机 rollout 健壮性：若随机死（20-40步），则 prefill 数据必死 → 开局即崩"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch as th
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import train_humanoid_sac as T

env_id = "Humanoid-v4"


def run_eval(model, det, seeds):
    out = []
    for s in seeds:
        env = gym.make(env_id, max_episode_steps=1500)
        env = T.WalkRewardWrapper(env)
        obs, _ = env.reset(seed=s)
        done = False; n = 0; rsum = 0.0
        while not done and n < 1500:
            a, _ = model.predict(obs, deterministic=det)
            obs, r, term, trunc, info = env.step(a)
            rsum += r; n += 1
            done = term or trunc
        out.append((n, round(rsum, 1)))
    return out


def main():
    model = SAC.load("./best_model/best_model.zip")
    print('当前 best_model.zip num_timesteps =', model.num_timesteps)

    env = gym.make(env_id); env = T.WalkRewardWrapper(env)
    obs, _ = env.reset(seed=42)
    with th.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        _, log_std, _ = model.policy.actor.get_action_dist_params(obs_t)
    print(f'探索std(站立) = {float(th.exp(log_std).mean())*0.4:.3f}')
    v = log_std.detach().cpu().numpy().ravel()
    print(f'log_std: min={v.min():.2f} max={v.max():.2f} 顶到+2占比={100*np.mean(v>=1.99):.1f}%')

    print('\n确定性 eval (seed 42/44/50-52):', run_eval(model, True, [42, 44, 50, 51, 52]))
    print('随机 eval (seed 42/43/44):', run_eval(model, False, [42, 43, 44]))
    del model


if __name__ == '__main__':
    main()