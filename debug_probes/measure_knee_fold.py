"""只读探针：同一仪器下比较各权重膝盖屈曲深度（不训练、不写模型）。

用法：
    python3 debug_probes/measure_knee_fold.py [模型路径 ...]

不传路径则跑默认的对照集（形态 2 / 3 / 4 / 5）。每个模型跑确定性策略，
种子 123 / 2026 各至多 1500 步（摔倒提前结束）。指标是膝关节角（rad，
负=屈曲）的逐帧最小值、中位数，以及屈曲超过 1.5 rad 的帧占比。
"""

import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

MAX_STEPS = 1500
SEEDS = (123, 2026)
FOLD_THRESH = -1.5  # rad，超过即视为"深折叠"

DEFAULT = [
    ("形态2 V9-实现交替慢跑", "best_model_saved/round1/V9/V9-实现交替慢跑/best_model.zip"),
    ("形态3 17_6_交替惩罚2.3", "best_model_saved/round2/17_6/17_6_交替惩罚2.3/best_model.zip"),
    ("形态3 17_6_d_最终版", "best_model_saved/round2/17_6/17_6_d_的最终版_加positions_swap/best_model.zip"),
    ("形态3 17_7_门控0.15_膝深折2.0", "best_model_saved/round2/17_7/17_7_门控0.15_膝深折惩罚2.0/best_model.zip"),
    ("形态4 18_加重腾空惩罚", "best_model_saved/round2/SAC_18/18_加重腾空惩罚/best_model.zip"),
    ("形态4 18_6", "best_model_saved/round2/SAC_18/18_6/best_model.zip"),
    ("形态5 19_4", "best_model_saved/round2/SAC_19/19_4_调权重_结果还是原地踏步速度慢了/best_model.zip"),
]


def measure(model_path):
    model = SAC.load(model_path, device="cpu")
    env = gym.make("Humanoid-v4", max_episode_steps=MAX_STEPS)
    m = env.unwrapped.model
    kr = m.jnt_qposadr[m.joint("right_knee").id]
    kl = m.jnt_qposadr[m.joint("left_knee").id]
    rows = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        R, L = [], []
        for _ in range(MAX_STEPS):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            R.append(float(env.unwrapped.data.qpos[kr]))
            L.append(float(env.unwrapped.data.qpos[kl]))
            if term or trunc:
                break
        R, L = np.array(R), np.array(L)
        rows.append(dict(seed=seed, steps=len(R),
                         rmin=R.min(), rmed=np.median(R), rfold=(R < FOLD_THRESH).mean(),
                         lmin=L.min(), lmed=np.median(L), lfold=(L < FOLD_THRESH).mean()))
    env.close()
    return rows


def main(paths):
    print(f"{'模型':34s} {'seed':>5s} {'步数':>5s} {'右膝min':>8s} {'右膝med':>8s} "
          f"{'右深折%':>8s} {'左膝min':>8s} {'左膝med':>8s} {'左深折%':>8s}")
    for name, path in paths:
        for r in measure(path):
            print(f"{name:34s} {r['seed']:5d} {r['steps']:5d} {r['rmin']:8.2f} {r['rmed']:8.2f} "
                  f"{100*r['rfold']:7.1f}% {r['lmin']:8.2f} {r['lmed']:8.2f} {100*r['lfold']:7.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main([(p.split("/")[-2] if p.endswith("best_model.zip") else p, p) for p in sys.argv[1:]])
    else:
        main(DEFAULT)
