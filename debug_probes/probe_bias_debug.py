"""调试 log_std bias 是否真正生效"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch as th
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

    lin = model.policy.actor.log_std
    print('log_std 模块 type:', type(lin).__name__, '| bias shape:', tuple(lin.bias.shape))
    print('bias 前3个值:', lin.bias.detach().cpu().numpy()[:3])

    with th.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(train_vec_env.reset())
        _, log_std, _ = model.policy.actor.get_action_dist_params(obs_t)
        cur = float(th.exp(log_std).mean())
        print('原始 log_std mean =', float(log_std.mean()), '| 归一化std =', cur)
        target = 0.08 / 0.4
        print('target归一化std =', target)
        delta = th.log(th.tensor(target / cur))
        print('delta =', float(delta))
        lin.bias.data.add_(delta)

    print('改后 bias 前3个值:', lin.bias.detach().cpu().numpy()[:3])
    with th.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(train_vec_env.reset())
        _, log_std2, _ = model.policy.actor.get_action_dist_params(obs_t)
        print('改后 log_std mean =', float(log_std2.mean()), '| 归一化std =', float(th.exp(log_std2).mean()), '| env探索std =', float(th.exp(log_std2).mean())*0.4)

    del old_model, model


if __name__ == '__main__':
    main()
