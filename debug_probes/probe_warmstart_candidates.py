"""warm-start 候选鲁棒性探针（修补后环境 + 当前奖励）
历史最强成功相关性：warm-start 模型随机 rollout（原生 std）能活 200+ 步 → 训练稳定；
随机 rollout 全死（如 576K）→ 预填充全是死亡轨迹 → 立即崩。
对每个候选：确定性 3 seed + 随机 3 seed，记录 ep_len / 总奖励 / 左右摆动幅度。
每模型加载后立即 del+gc（SAC.load 会按保存的 buffer_size 重建数 GB 的 replay buffer）。
"""
import os, sys, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from mirror_map import patch_mirror_symmetry
from train_humanoid_sac import WalkRewardWrapper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = [
    ("560K_17_7门控0.15膝折2.0(eval1766,左步幅0.121)",
     "best_model/tmp_model_data/17_7/17_7_门控0.15_膝深折惩罚2.0/best_model_snapshots/best_model_560000.zip"),
    ("1472K_14_2(eval峰值1786)",
     "best_model_saved/V9/14_2/best_model_snapshots_14_2/best_model_1472000.zip"),
    ("896K_近轮lineage",
     "best_model/best_model_snapshots/best_model_896000.zip"),
    ("1680K_近轮(=当前best_model.zip)",
     "best_model/best_model.zip"),
]


def rollout(model, seed, deterministic):
    e = gym.make("Humanoid-v4", max_episode_steps=1500)
    patch_mirror_symmetry(e.unwrapped.model)
    e = WalkRewardWrapper(e)
    obs, _ = e.reset(seed=seed)
    tot, n, done = 0.0, 0, False
    info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = e.step(a)
        tot += float(r)
        n += 1
        done = term or trunc
    e.close()
    return dict(ep_len=n, rew=tot,
                left_sw=info.get("mean_left_swing"), right_sw=info.get("mean_right_swing"))


def fmt(rs):
    lens = [r["ep_len"] for r in rs]
    rews = [f'{r["rew"]:.0f}' for r in rs]
    sw = [f'({r["left_sw"]:.2f}/{r["right_sw"]:.2f})' if r["left_sw"] is not None else "(-/-)" for r in rs]
    return f'步数{str(lens):24s} 奖励{str(rews):20s} 左/右摆幅{sw}'


print(f'{"候选":46s} | 随机rollout鲁棒线: 全seed ≥200 步为佳')
for name, path in CANDIDATES:
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        print(f'\n== {name}\n   ✗ 文件不存在: {path}')
        continue
    model = SAC.load(p, device="cpu")
    det = [rollout(model, s, True) for s in (1, 2, 3)]
    sto = [rollout(model, s, False) for s in (1, 2, 3)]
    print(f'\n== {name}  [num_timesteps={model.num_timesteps}]')
    print(f'   确定性: {fmt(det)}')
    print(f'   随机std: {fmt(sto)}')
    del model
    gc.collect()
