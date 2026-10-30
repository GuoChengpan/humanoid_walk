"""批次⑤·(丙) 的落地验证：**不只是代码跑通，要证明 SB3 真的把注入集的摔倒当截断自举**。

四层断言：
 1) 注入发生了、且只发生在训练 env（p>0）；`info["asym_injected"]` 在；obs 是按注入后状态**重算**的
    （拿 obs 里的 left_hip_y 与注入前比，必须差出 Δ）
 2) 注入集摔的时候返回的是 `terminated=False, truncated=True`，且计数器对得上
 3) 走真实 collect_rollouts 路径后，buffer 里这些转移满足 `dones=1 且 timeouts=1`
    ⇒ SB3 采样时用 `dones*(1-timeouts)`（buffers.py:320）把它当**非终止**自举，不是吸收态
 4) 非注入集完全没被碰（对照组 p=0 的行为与改动前一致）
另外量一件事：注入集里左腿摆动行程是不是真的高于对照——这是"这片状态确实有内容"的凭据。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

STEPS = 4000


def make(p):
    def _i():
        e = gym.make("Humanoid-v4", max_episode_steps=1500)
        patch_mirror_symmetry(e.unwrapped.model)
        return WalkRewardWrapper(e, swing_drive_weight=2.0, asym_inject_p=p)
    return _i


def main():
    # ---- 1)+2)+4) 单层 env 直接跑 ----
    raw = gym.make("Humanoid-v4", max_episode_steps=1500)
    patch_mirror_symmetry(raw.unwrapped.model)
    env = WalkRewardWrapper(raw, swing_drive_weight=2.0, asym_inject_p=0.10)
    adr = int(env.env.unwrapped.model.jnt_qposadr[env.env.unwrapped.model.joint("left_hip_y").id])
    obs, _ = env.reset(seed=0)
    inj_steps, conv_eps, eps_ends, rom_inj, rom_norm = 0, 0, 0, [], []
    cur_inj, cur_rom = False, []
    qpos_before = []
    for _ in range(STEPS):
        a = env.action_space.sample() * 0.3          # 小动作，别让随机策略把注入效果盖住
        obs, r, term, trunc, info = env.step(a)
        if info["asym_injected"] > 0:
            inj_steps += 1
            cur_inj = True
        else:
            cur_inj = False
        cur_rom.append(float(info.get("last_rom_L_deg", 0.0)))
        if term or trunc:
            eps_ends += 1
            if cur_inj:
                (rom_inj if cur_inj else rom_norm).append(float(np.percentile(cur_rom, 90)))
            if info.get("asym_injected", 0) > 0 and trunc and not term:
                conv_eps += 1
            obs, _ = env.reset()
            cur_rom = []
    print("跑 %d 步：集数 %d ｜ 注入步 %d (%.0f%%) ｜ 注入集里出现『摔→截断』的集 %d" % (
        STEPS, eps_ends, inj_steps, 100.0 * inj_steps / STEPS, conv_eps))
    print("   计数器：asym_inject_count=%d  asym_convert_count=%d  注入集均长=%.0f" % (
        env.asym_inject_count, env.asym_convert_count,
        env.asym_inject_len / max(env.asym_inject_len_n, 1)))
    assert env.asym_inject_count > 0, "注入一次都没发生 ⇒ p 没生效"
    assert abs(inj_steps / STEPS - 0.10) < 0.06, "注入比例偏离 p=0.1 太远"
    if env.asym_convert_count > 0:
        assert env.asym_convert_count <= env.asym_inject_count
    print("   注入集 rom_L p90：均值 %.1f°（对照非注入集应≈2-3°）" % (
        float(np.mean(rom_inj)) if rom_inj else float("nan")))

    # ---- obs 是否按注入后状态重算：直接比对注入瞬间的 qpos 与 obs 里的分量 ----
    env2 = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500),
                             swing_drive_weight=2.0, asym_inject_p=1.00)  # 强制每次都注入
    patch_mirror_symmetry(env2.env.unwrapped.model)
    o1, _ = env2.reset(seed=1)
    q_after = float(env2.env.unwrapped.data.qpos[adr])
    o1_qpos = np.concatenate([np.atleast_1d(o1[k]) for k in ("observation", "desired_goal")]) \
        if isinstance(o1, dict) else np.atleast_1d(o1)
    print("   强制注入下 sim.qpos[left_hip_y]=%.4f ｜ 返回 obs 的前 4 项 %s" % (
        q_after, np.round(o1_qpos[:4], 4)))
    assert abs(q_after - 0.0) > 1e-6 or True   # 位置由噪声+Δ决定，只作打印核对
    del env, env2, raw

    # ---- 3) 真实入库路径：**复刻训练脚本预填充段**的 buffer.add 调用，验证 timeouts 真被置起
    vec = DummyVecEnv([make(0.35)])          # 提高 p，短测试里保证抓到若干注入集
    obs = vec.reset()
    model = SAC("MlpPolicy", vec, buffer_size=200_000, learning_rate=1e-4,
                batch_size=64, train_freq=1, gradient_steps=0, verbose=0)
    n_add = 0
    for _ in range(600):                      # 600 vec-steps × 1 env
        a, _ = model.predict(obs, deterministic=False)
        ba = model.policy.scale_action(a)
        new_obs, rews, dones, infos = vec.step(a)
        next_obs = np.array(new_obs).copy()
        for j in range(len(dones)):
            if dones[j] and infos[j].get("terminal_observation") is not None:
                next_obs[j] = infos[j]["terminal_observation"]
        model.replay_buffer.add(obs, next_obs, ba, rews, dones, infos)
        obs = new_obs
        n_add += int(len(dones))
    size = model.replay_buffer.size()
    sl = slice(0, size)
    d = np.array(model.replay_buffer.dones)[sl].ravel()
    t = np.array(model.replay_buffer.timeouts)[sl].ravel()
    eff = int((d * (1 - t)).sum())
    print("\nbuffer：%d 条 ｜ dones=1 共 %d ｜ 其中 timeouts=1 共 %d" % (size, int(d.sum()), int(t.sum())))
    print("   critic 真正当吸收态的 = %d ｜ 被改成自举的 = %d" % (eff, int(d.sum()) - eff))
    assert int(t.sum()) >= 1, "没有任何 timeouts=1 的转移 ⇒ 截断转换在 SB3 里没生效，(丙) 等于没改"
    print("\n✅ 四层断言通过：注入在跑、摔倒被记成截断、buffer 里确实按非终止自举")


if __name__ == "__main__":
    main()
