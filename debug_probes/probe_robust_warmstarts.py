"""当前 reward 下测候选模型的随机 rollout 健壮性。
判定：warm-start 能活的必要条件 = 随机(原生std) rollout ep_len >= 200。
同时量确定性 ep_len 和原生探索 std，判断每个候选的噪声水平。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch as th
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import train_humanoid_sac as T

env_id = "Humanoid-v4"

CANDIDATES = [
    ("17_6_交替惩罚2.3 (稳定run) ", "best_model/tmp_model_data/17_6/17_6_交替惩罚2.3/best_model.zip"),
    ("V9/14_2 1472K (峰值1786)  ", "best_model_saved/V9/14_2/best_model_snapshots_14_2/best_model_1472000.zip"),
    ("V8 best (3872K)          ", "best_model_saved/V8/best_model.zip"),
    ("V7 best (4192K)          ", "best_model_saved/V7/best_model.zip"),
    ("V9/14_5 best (4896K)     ", "best_model_saved/V9/14_5/best_model.zip"),
    ("tmp 944K                 ", "best_model/tmp_model_data/best_model_944000.zip"),
    ("tmp 800K                 ", "best_model/tmp_model_data/best_model_800000.zip"),
    ("V10 15_1高速惩罚 (2112K)   ", "best_model_saved/V10/15_1/增加高速惩罚/best_model.zip"),
    ("tmp last_TMP             ", "best_model/tmp_model_data/last_TMP/best_model.zip"),
    ("17_5 best (2352K)        ", "best_model/tmp_model_data/17_5/best_model.zip"),
]


def meas_std(model):
    env = gym.make(env_id)
    env = T.WalkRewardWrapper(env)
    obs, _ = env.reset(seed=42)
    with th.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        _, log_std, _ = model.policy.actor.get_action_dist_params(obs_t)
        return float(th.exp(log_std).mean()) * 0.4


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
    for name, path in CANDIDATES:
        if not os.path.exists(path):
            print(f'{name}  ✗ 文件不存在: {path}')
            continue
        try:
            model = SAC.load(path)
            nts = model.num_timesteps
            std = meas_std(model)
            det = run_eval(model, True, [42, 44])
            stoc = run_eval(model, False, [42, 43, 44])
            lens = [x[0] for x in stoc]
            maxl = max(lens)
            alive = '✓ 健壮' if maxl >= 200 else '✗ 边缘'
            print(f'{name} steps={nts:>8d} 探索std={std:.3f} | 确定性={det} | 随机={stoc} → max={maxl} {alive}')
            del model
        except Exception as e:
            print(f'{name}  ✗ 错误: {type(e).__name__}: {e}')
        print(flush=True)


if __name__ == '__main__':
    main()