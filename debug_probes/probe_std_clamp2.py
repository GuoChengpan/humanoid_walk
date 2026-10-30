"""硬压 log_std：验证 416K 的 log_std 分布 + 清零 weight/填 bias 后随机 rollout 能否走"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch as th
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import train_humanoid_sac as T

env_id = "Humanoid-v4"
n_train_envs = 8
seed = 42


def main():
    train_vec_env = SubprocVecEnv([T.make_env(env_id, i, seed) for i in range(n_train_envs)])
    train_vec_env = VecMonitor(train_vec_env)

    old_model = SAC.load("./best_model/tmp_model_data/best_model_416000.zip", env=train_vec_env)
    print('416K 模型 num_timesteps =', old_model.num_timesteps)
    model = SAC('MlpPolicy', train_vec_env, learning_rate=1e-4, buffer_size=1_500_000,
                batch_size=1024, tau=0.005, gamma=0.99, train_freq=512, gradient_steps=1024,
                ent_coef=0.05, target_entropy=-8.5, policy_kwargs={'net_arch':[512,256]}, verbose=0)
    model.policy.load_state_dict(old_model.policy.state_dict())

    def log_std_stats(m, tag):
        with th.no_grad():
            obs_t, _ = m.policy.obs_to_tensor(train_vec_env.reset())
            _, log_std, _ = m.policy.actor.get_action_dist_params(obs_t)
            v = log_std.detach().cpu().numpy().ravel()
            print(f'  {tag}: mean={v.mean():7.2f} min={v.min():7.2f} max={v.max():7.2f} '
                  f'顶到+2占比={100*np.mean(v>=1.99):5.1f}% exp均值={np.exp(v).mean():8.3f} → env探索std={np.exp(v).mean()*0.4:.3f}')

    print('压低前 log_std 分布（站立初始姿态）:')
    log_std_stats(model, 'before')

    # 硬压：log_std 变为常数 ln(0.08/0.4)，与状态无关，所有维度统一
    with th.no_grad():
        model.policy.actor.log_std.weight.data.zero_()
        model.policy.actor.log_std.bias.data.fill_(th.log(th.tensor(0.08 / 0.4)))

    print('硬压后 log_std 分布:')
    log_std_stats(model, 'after')

    for det in [True, False]:
        lens = []
        for ep in range(4):
            env = gym.make(env_id, max_episode_steps=1500)
            env = T.WalkRewardWrapper(env)
            obs, _ = env.reset(seed=100+ep)
            done=False; n=0; rsum=0.0
            while not done and n<1500:
                a,_ = model.predict(obs, deterministic=det)
                obs,r,term,trunc,info = env.step(a)
                rsum+=r; n+=1; done = term or trunc
            lens.append((n, round(rsum,1)))
        print(f'{"确定性" if det else "随机"} eval（硬压后）ep_len/rew:', lens)

    del old_model, model


if __name__ == '__main__':
    main()
