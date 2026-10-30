"""镜像增广到底有没有真的进 buffer？——迷你复现（不训练，只跑 collect 路径）。

动机：Q(Ms,Ma) − Q(s,a) = −75.4±1.9（相对 Q0=18.1 是 −417%）说明 **critic 完全不等变**。
如果镜像整集真的按 1:1 进了 replay buffer，一个 512×256 的 MLP 面对一半严格镜像的数据
不可能差到这个量级 → 怀疑数据通路本身。这里绕开训练，直接跑 SB3 的 collect+注入路径：
 8 个 DummyVecEnv（与 make_env 完全一致的 with_mirror 包装）+ MirrorEqSAC 的真实 buffer，
 随机动作走 3 个 episode 长度，统计 injector.episodes / buffer.size() / 镜像集长度。
只读，不动模型文件、不动训练进程。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from mirror_map import MirrorMap
from train_humanoid_sac import (EpisodeRecorder, MirrorReplayer, WalkRewardWrapper,
                                patch_mirror_symmetry)

N_ENV = 8
STEPS = 3000


def main():
    # 与 train_humanoid_sac.make_env(with_mirror=True) 同构
    def _init(rank):
        env = gym.make("Humanoid-v4", max_episode_steps=1500)
        patch_mirror_symmetry(env.unwrapped.model)
        env = WalkRewardWrapper(env, swing_drive_weight=2.0)
        r_env = gym.make("Humanoid-v4", max_episode_steps=1500)
        patch_mirror_symmetry(r_env.unwrapped.model)
        r_env = WalkRewardWrapper(r_env, swing_drive_weight=2.0)
        env = EpisodeRecorder(env, MirrorReplayer(r_env, MirrorMap(r_env.unwrapped.model)))
        env.reset(seed=42 + rank)
        return env

    vec = DummyVecEnv([(lambda i=i: _init(i)) for i in range(N_ENV)])
    probe = SAC("MlpPolicy", vec, policy_kwargs=dict(net_arch=[16, 16]),
                buffer_size=200000, learning_starts=10 ** 9, seed=1, device="cpu")
    buf = probe.replay_buffer
    # 注入器：与 MirrorAugCallback 走的同一条路
    from train_humanoid_sac import MirrorInjector
    inj = MirrorInjector(n_envs=N_ENV)

    obs = vec.reset()
    n_ep = 0
    err = None
    for t in range(STEPS):
        a = np.stack([vec.action_space.sample() for _ in range(N_ENV)])
        obs, r, d, info = vec.step(a)
        try:
            inj.drain_infos(info, buf)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
        n_ep += int(np.sum(d))
        if d.any():
            obs = vec.reset()
        if t > 400 and inj.episodes >= 3:
            break
    print(f"跑了 {t+1} vec-step（每组 {N_ENV} env）→ 结束集数≈{n_ep}")
    print(f"MirrorInjector.episodes = {inj.episodes}   pending 余 = {len(inj.pending)}")
    print(f"buffer.size() = {getattr(buf, 'size', lambda: -1)()}  (采集真实转移 {t+1}*{N_ENV}={(t+1)*N_ENV})")
    print(f"drain 异常: {err or '无'}")
    if inj.episodes == 0:
        print("\n✗ 一个镜像整集都没产生 → 通路断了（这就是『镜像不起作用』的真正原因）")
    else:
        print(f"\n✓ 镜像集确实在产生；每集约 {int(buf.size()/inj.episodes)} 条进 buffer"
              f"（若远小于真实轨迹占比，说明注入率不是 1:1）")
    vec.close()


if __name__ == "__main__":
    main()
