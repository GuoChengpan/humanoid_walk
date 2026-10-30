"""端到端冒烟：SubprocVecEnv(2 worker, with_mirror) + 小 SAC 短程 learn
验证：worker 内录制/重放、info 经管道序列化、MirrorAugCallback 注入 buffer 全链路。
不加载/不保存任何用户模型，不产生 checkpoint。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from train_humanoid_sac import make_env, MirrorAugCallback, MirrorInjector

if __name__ == "__main__":
    N_ENVS = 2
    STEPS = 3000

    vec = SubprocVecEnv([make_env("Humanoid-v4", i, 0, patch=True, with_mirror=True)
                         for i in range(N_ENVS)])
    injector = MirrorInjector(n_envs=N_ENVS)
    model = SAC('MlpPolicy', vec, buffer_size=50_000, batch_size=64, train_freq=64,
                gradient_steps=2, seed=0, verbose=0)
    model.learn(total_timesteps=STEPS, callback=[MirrorAugCallback(injector)])

    buf_n = model.replay_buffer.pos * N_ENVS
    real_n = model.num_timesteps
    print(f"num_timesteps={real_n}, buffer={buf_n} 条, 镜像注入={injector.transitions} 条 "
          f"({injector.episodes} 集)")
    ratio = injector.transitions / max(real_n, 1)
    print(f"镜像/真实 比 = {ratio:.2f}（期望 ≈ 1.0）")

    # 抽样健康检查
    s = model.replay_buffer.sample(64)
    finite = all(np.isfinite(x.numpy()).all() for x in [s.observations, s.actions, s.rewards])
    print(f"sample 有限={finite}")

    ok = injector.transitions > 0 and 0.5 < ratio < 1.6 and finite
    print("✓ 冒烟通过" if ok else "✗ 冒烟失败")
    sys.exit(0 if ok else 1)
