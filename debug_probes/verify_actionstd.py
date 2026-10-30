"""两件事：
1) 验证 train/action_std 真的落进 tensorboard（logger 要 close 才会 flush，events 在 exp-*/ 子目录）。
2) 重新量策略的真实探索 std —— 上轮 0.0817 是 3 样本含确定性中心的严重小样本偏差，
   用 E|ε1-ε2| = σ√(2/π) 反推真实 σ 约 0.18，跟 0.08 差一倍多。这决定判定线画在哪。
SubprocVecEnv 走 forkserver 会重跑 __main__，所以全包在 if __name__ == "__main__" 里。
不碰 best_model.zip（只挂 LegActivityCallback，无 EvalCallback/SaveBest）。
"""
import os, sys, glob, shutil, gc, tempfile
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from train_humanoid_sac import WalkRewardWrapper, LegActivityCallback, make_env


def check_tag():
    tmp = tempfile.mkdtemp(prefix="tb_actionstd_")
    envs = VecMonitor(SubprocVecEnv([make_env("Humanoid-v4", i, 42) for i in range(4)]))
    m = SAC("MlpPolicy", envs, learning_rate=1e-5, buffer_size=4000, batch_size=16,
            train_freq=4, gradient_steps=4, ent_coef=0.05, target_entropy=-8.5,
            policy_kwargs={"net_arch": [512, 256]}, verbose=0, tensorboard_log=tmp)
    m.learn(total_timesteps=400, callback=[LegActivityCallback()], log_interval=5, progress_bar=False)
    m.logger.close()
    m.env.close()

    files = glob.glob(os.path.join(tmp, "**", "events.out.tfevents.*"), recursive=True)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(files[0], size_guidance={"scalars": 0}) if files else EventAccumulator(tmp)
    acc.Reload()
    tags = sorted(acc.Tags()["scalars"])
    tr = [t for t in tags if t.startswith("train")]
    print("events 文件 %d 个, tags %d 个" % (len(files), len(tags)))
    print("train/* tags:", tr)
    if "train/action_std" in tags:
        v = np.array([x.value for x in acc.Scalars("train/action_std")])
        print("OK train/action_std 写入成功: %d 点, 范围 %.4f~%.4f 均值 %.4f"
              % (len(v), v.min(), v.max(), v.mean()))
        print("   前 12 点:", np.round(v[:12], 4))
    else:
        print("FAIL train/action_std 未写入")
    m.replay_buffer = None
    del m
    shutil.rmtree(tmp, ignore_errors=True)
    gc.collect()


def measure_true_sigma():
    print("\n=== 策略真实探索 std（560K 快照，沿确定性轨迹采样 40 个状态）===")
    e = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
    m = SAC.load("best_model/best_model.zip", env=e, buffer_size=20000)
    K = 8
    o, _ = e.reset(seed=7)
    per_step = []
    for _ in range(40):
        mu, _ = m.predict(o, deterministic=True)
        samp = np.stack([np.asarray(m.predict(o, deterministic=False)[0], dtype=np.float32) for _ in range(K)])
        s = samp.std(axis=0)
        per_step.append(s)
        o, _, te, tr, _ = e.step(np.asarray(mu, dtype=np.float32))
        if te or tr:
            o, _ = e.reset(seed=11)
    P = np.array(per_step)
    print("  每维 std: 跨状态均值 %.4f  最小 %.4f  最大 %.4f" % (P.mean(), P.min(), P.max()))
    print("  按状态看：均值 %.4f  最小 %.4f  最大 %.4f" % (P.mean(axis=1).mean(), P.mean(axis=1).min(), P.mean(axis=1).max()))
    diffs = []
    for _ in range(200):
        a, b = m.predict(o, deterministic=False)[0], m.predict(o, deterministic=False)[0]
        diffs.append(np.abs(np.asarray(a) - np.asarray(b)).mean())
    d = float(np.mean(diffs))
    print("  E|a1-a2| 均值 %.4f  →  反推 σ = %.4f  (σ = E|diff| / √(2/π))" % (d, d / np.sqrt(2 / np.pi)))
    m.replay_buffer = None
    del m
    gc.collect()


if __name__ == "__main__":
    check_tag()
    measure_true_sigma()
