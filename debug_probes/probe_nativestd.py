"""量随机 rollout 与确定性轨迹上策略实际使用的探索 std（exp(log_std)，env 单位）。

上一轮 σ=0.18 是沿确定性轨迹量的。但随机 rollout 走的是另一批状态
（探索会离开确定性轨道、跌倒、重置）。这两条轨迹上的 log_std 分布若不同，
那才是 train/action_std 真实会读到的量，也解释了「0.18 却 319/集」的矛盾。

只比 exp(log_std)：它是策略自己的探索幅度，单位明确（env 动作空间），不与 tanh 混淆。
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
N_EP = 5


def walk(model, env, seed0, det, scale):
    """走 N_EP 集，返回 (每步的 17 维 exp(log_std)*scale 数组, 每集步数列表)"""
    rows, lens = [], []
    device = model.policy.device
    for ep in range(1, N_EP + 1):
        obs, _ = env.reset(seed=seed0 + ep)
        n = 0
        while True:
            o = th.from_numpy(np.asarray(obs)).float().to(device).unsqueeze(0)
            with th.no_grad():
                _, ls, _ = model.policy.actor.get_action_dist_params(o)
            rows.append(th.exp(ls).squeeze(0).cpu().numpy() * scale)
            a, _ = model.predict(obs, deterministic=det)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
            n += 1
            if te or tr:
                break
        lens.append(n)
        print("  集%d 结束: %d 步" % (ep, n))
    return np.asarray(rows), lens


def rep(name, L):
    per_state = L.mean(axis=1)
    print("\n=== %s  (%d 个状态) ===" % (name, L.shape[0]))
    print("  每维 exp(log_std)*0.4  总体均值: %.4f" % L.mean())
    print("  按状态均值: %.4f  (跨状态范围 %.4f ~ %.4f)"
          % (per_state.mean(), per_state.min(), per_state.max()))
    print("  17 维逐维均值: 最小 %.4f  最大 %.4f"
          % (L.mean(axis=0).min(), L.mean(axis=0).max()))
    return per_state


def main():
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    model = SAC.load(PATH, env=env, buffer_size=20000)
    del model.replay_buffer
    gc.collect()
    scale = float(model.action_space.high[0])   # 0.4: actor 原始空间 → env 空间

    print("动作空间 ±%.2f；exp(log_std)*%.2f 即 env 单位探索 std" % (scale, scale))
    print("\n--- 随机 rollout（策略自己探索）---")
    Ls, ns = walk(model, env, 9000, False, scale)
    print("\n--- 确定性轨迹（对照）---")
    Ld, nd = walk(model, env, 9500, True, scale)
    env.close()

    ps = rep("随机 rollout", Ls)
    pd = rep("确定性轨迹", Ld)

    print("\n=== 两者差异 ===")
    print("  按状态 std 均值: 随机 %.4f  vs 确定性 %.4f  →  差 %.4f (%.1f%%)"
          % (ps.mean(), pd.mean(), ps.mean() - pd.mean(), 100 * (ps.mean() - pd.mean()) / pd.mean()))

    print("\n=== 对照噪声悬崖表（env 单位，560K）===")
    print("  σ=0.05 → 99.6%%   σ=0.10 → 25%%   σ=0.18 → -8%%   σ=0.25 → -14.6%%")
    print("  原生随机 rollout 实测 std 落在哪一档 = train/action_std 的判定基准。")


if __name__ == "__main__":
    main()
