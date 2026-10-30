"""确认探索强度的正确口径，并量化 axis=0 跨环境 std 的虚高幅度。

问题：train/action_std 用 np.std(actions, axis=0) 量「探索噪声」，
但 actions 是 8 个 env 在 8 个不同步态相位上的采样——
std = sqrt(Var(采样噪声) + Var(mu_跨相位))，后者不是探索而是动作本身的相位变化。

正确量：actor.get_action_dist_params(obs)[1] = log_std，每维每状态的采样 std 本身。
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

PATH = "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"


def main():
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    model = SAC.load(PATH, env=env, buffer_size=20000)
    del model.replay_buffer
    gc.collect()
    device = model.policy.device

    # --- 单环境走 5 集，逐状态记录 log_std ---
    n_dim = model.action_space.shape[0]
    logstd_rows = []      # (T, 17) 每个状态的每维 std
    obs_rows = []         # (T, obs_dim)
    for ep in range(1, 6):
        obs, _ = env.reset(seed=2000 + ep)
        while True:
            o = th.from_numpy(np.asarray(obs)).float().to(device)
            o = o.unsqueeze(0)                      # (1, obs_dim)
            with th.no_grad():
                mean_a, log_std, _ = model.policy.actor.get_action_dist_params(o)
            logstd_rows.append(log_std.squeeze(0).cpu().numpy())
            obs_rows.append(np.asarray(obs, dtype=np.float32))
            a, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            if te or tr:
                break
        print("  集%d 结束" % ep)
    env.close()

    L = th.tensor(np.asarray(logstd_rows), dtype=th.float32)   # (T, 17)
    std_per_dim = th.exp(L)                                   # 每维每状态 std
    per_dim_mean = std_per_dim.mean(dim=0)                    # (17,)
    per_state_mean = std_per_dim.mean(dim=1)                  # (T,)

    print("\n=== 正确口径：exp(log_std) 采样分布 std（560K 快照，%.0f 个状态）===" % L.shape[0])
    print("  每维跨状态均值: %8.4f   最小 %8.4f   最大 %8.4f"
          % (per_dim_mean.mean(), per_dim_mean.min(), per_dim_mean.max()))
    print("  按状态看均值:   %8.4f   最小 %8.4f   最大 %8.4f"
          % (per_state_mean.mean(), per_state_mean.min(), per_state_mean.max()))

    # --- 反推：每维 std 的平方和开根 = 动作向量的总抖动幅度 ---
    total_jitter = th.sqrt((std_per_dim.pow(2)).sum(dim=1)).mean()
    print("  动作向量总抖动 sqrt(sum(std^2)) 跨状态均值: %.4f" % total_jitter)

    # --- 对照：axis=0 跨环境 std 会量到多少 ---
    print("\n=== 对照：axis=0 跨环境 std（8 个不同相位 env）===")
    T = len(obs_rows)
    np.random.seed(0)
    picks = np.random.choice(T, size=40, replace=False)
    fake_xenv = []
    for _ in range(20):                       # 20 次重复抽 8 个状态
        idx = np.random.choice(T, size=8, replace=False)
        obsb = th.from_numpy(np.asarray([obs_rows[i] for i in idx], dtype=np.float32)).to(device)
        with th.no_grad():
            samp = model.policy.actor(obsb, deterministic=False)   # 训练时同一条路径：squashed 采样
        fake_xenv.append(float(np.std(samp.cpu().numpy(), axis=0).mean()))
    print("  axis=0 跨环境 std 抽样 20 次: 均值 %.4f  范围 %.4f~%.4f"
          % (np.mean(fake_xenv), np.min(fake_xenv), np.max(fake_xenv)))
    print("  → 虚高倍数 = %.2f×" % (np.mean(fake_xenv) / per_state_mean.mean()))

    # --- ent_coef 0.05 vs 0.1 能否降 std：看 log_std 与 ent_coef 的耦合点 ---
    print("\n=== ent_coef 作用路径 ===")
    print("  ent_coef=%.2f  当前值(快照) log_std 逐维范围: %s"
          % (model.ent_coef, np.round(np.asarray([per_dim_mean.min(), per_dim_mean.mean(),
                                                  per_dim_mean.max()]), 3).tolist()))
    print("  actor_loss 中含 + ent_coef * log_std.mean()（SAC 目标），ent_coef 越大 log_std 越被推高")
    print("  → train/action_std 下降 = log_std 下降 = ent_coef 生效的唯一可观测证据")


if __name__ == "__main__":
    main()
