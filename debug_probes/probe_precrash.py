"""诊断崩前 checkpoint（best_model.zip=256K，eval 峰 1340）：
1) 确定性/随机 rollout ep_len —— 崩前随机策略有多脆
2) 逐动作维 log_std —— 哪些维度的噪声高（死亡螺旋源头）
3) 与 warm-start 1472000 对比
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch as th
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import train_humanoid_sac as T

env_id = "Humanoid-v4"


def meas_logstd(model, obs):
    with th.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        _, log_std, _ = model.policy.actor.get_action_dist_params(obs_t)
        return log_std.detach().cpu().numpy()  # (N, 17)


def run_ep(model, det, seed):
    env = gym.make(env_id, max_episode_steps=1500); env = T.WalkRewardWrapper(env)
    obs, _ = env.reset(seed=seed); done=False; n=0; r=0.0
    while not done and n < 1500:
        a, _ = model.predict(obs, deterministic=det)
        obs, rr, term, trunc, info = env.step(a); r += rr; n += 1
        done = term or trunc
    return n, round(r, 1)


def diag(model, tag):
    print(f'=== {tag} ===')
    print('num_timesteps =', model.num_timesteps)
    # 站立姿态 log_std
    env0 = gym.make(env_id); env0 = T.WalkRewardWrapper(env0)
    obs0, _ = env0.reset(seed=42)
    ls0 = meas_logstd(model, obs0)
    print(f'站立 log_std: mean={ls0.mean():.3f} min={ls0.min():.3f} max={ls0.max():.3f}')
    print(f'站立 std(env) 逐维:', ' '.join(f'{v:.3f}' for v in np.exp(ls0[0])*0.4))
    # 行走状态 log_std（确定性走 500 步采几个点）
    env = gym.make(env_id, max_episode_steps=1500); env = T.WalkRewardWrapper(env)
    obs, _ = env.reset(seed=50); walk_std = []
    done=False; n=0
    while not done and n < 500:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if n % 100 == 0:
            walk_std.append(meas_logstd(model, obs)[0])
        n += 1; done = term or trunc
    if walk_std:
        W = np.array(walk_std)
        print(f'行走 log_std: mean={W.mean():.3f} 逐维mean:', ' '.join(f'{v:.3f}' for v in W.mean(0)))
        print(f'行走 std(env) 逐维mean:', ' '.join(f'{v:.3f}' for v in np.exp(W.mean(0))*0.4))
    # rollout
    print('确定性 (42/50/52):', [run_ep(model, True, s) for s in [42, 50, 52]])
    print('随机   (42/43/44):', [run_ep(model, False, s) for s in [42, 43, 44]])
    print()


def main():
    crash = SAC.load("./best_model/best_model.zip")       # 256K 崩前
    base = SAC.load("best_model/tmp_model_data/17_6/17_6_d_的最终版_加positions_swap/best_model.zip")  # 1472000 warm-start
    diag(crash, '崩前 checkpoint')
    diag(base, 'warm-start 1472000 参照')
    del crash, base


if __name__ == '__main__':
    main()
