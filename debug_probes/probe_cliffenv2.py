"""隔离「噪声分布形状」这一个变量。

上一轮结果里，560K 策略的真实采样幅度 ≈0.18(env)，落在悬崖死亡区，
但原生随机 rollout 得 319/集。中心(μ)相同、std 相近，唯一差别是形状：
  - 原生: scale * tanh(mu + std*eps)      → 大 std 下近均匀分布于 [-0.4,0.4]
  - 注入: clip(det_action + N(0,sigma))    → 以 det 动作为中心的截断高斯
同种子、同环境、8 集，确认这个差不是种子假象。
"""
import os, sys, gc
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper

PATH = "best_model/tmp_model_data/17_6/17_6_2.3+门控0.5/best_model_snapshots/best_model_576000.zip"
SEEDS = list(range(20000, 20010))   # 10 集，所有条件共用
N = len(SEEDS)

# 条件：("名称", 采样方式)
CONDS = [
    ("det 无噪声",            "det"),
    ("det + N(0,0.05) 截断",  0.05),
    ("det + N(0,0.10) 截断",  0.10),
    ("det + N(0,0.18) 截断",  0.18),
    ("原生策略采样(tanh)",     "nat"),
]


def run(model, env, cond, seed):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    tot, n = 0.0, 0
    while True:
        if cond == "det":
            a, _ = model.predict(obs, deterministic=True)
        elif cond == "nat":
            a, _ = model.predict(obs, deterministic=False)
        else:
            a, _ = model.predict(obs, deterministic=True)
            a = np.asarray(a, dtype=np.float32)
            a = np.clip(a + rng.normal(0.0, cond, a.shape),
                        model.action_space.low, model.action_space.high)
        obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
        tot += float(r)
        n += 1
        if te or tr:
            break
    return tot, n


def main():
    env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    model = SAC.load(PATH, env=env, buffer_size=20000)
    del model.replay_buffer
    gc.collect()

    print("动作空间 ±%.2f    每条件 %d 集（同种子 20000..20009）\n"
          % (float(model.action_space.high[0]), N))
    print("%-26s %10s %8s %9s %9s" % ("条件", "reward/集", "ep_len", "活满率", "波动sd"))
    rows = []
    for name, cond in CONDS:
        rw, ln, sv = [], [], 0
        for s in SEEDS:
            r, l = run(model, env, cond, s)
            rw.append(r); ln.append(l); sv += 1 if l >= 1450 else 0
        rw = np.array(rw)
        rows.append((name, rw.mean(), np.mean(ln), 100 * sv / N, rw.std()))
        print("%-26s %10.1f %8.0f %8.0f%% %9.1f" % (name, rw.mean(), np.mean(ln), 100 * sv / N, rw.std()))
    env.close()

    base = rows[0][1]
    print("\n相对 det 保持率:")
    for name, r, l, sv, sd in rows:
        print("  %-26s %8.1f%%   (%.0f/集)" % (name, 100 * r / base, r))

    det18 = [x for x in rows if x[0].startswith("det + N(0,0.18)")][0]
    nat = [x for x in rows if x[0].startswith("原生")][0]
    print("\n=== 形状差异的净效应 ===")
    print("  同中心(μ)、同幅度(≈0.18) 下:")
    print("    tanh 原生采样 : %7.1f/集" % nat[1])
    print("    截断高斯注入   : %7.1f/集" % det18[1])
    print("    形状净效应     : %+7.1f/集" % (nat[1] - det18[1]))
    print("  → 若为正且显著大于两者波动 sd(%.0f)，则分布形状本身是保护因素，" % max(nat[4], det18[4]))
    print("    train/action_std 不能直接对照悬崖表的 σ 判定。")


if __name__ == "__main__":
    main()
