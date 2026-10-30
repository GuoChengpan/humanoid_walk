"""端到端 smoke：确认 train/action_std 与 train/exploration_std 都真写入 tensorboard。

随机初始化策略（快速跌倒 → 集短 → dump_logs 能触发），2000 步 / 4 env = 500 批次，
门控 step%100==0 命中 0/100/200/300/400 共 5 次，dump 每 5 集一次。
SubprocVecEnv 走 forkserver 会重跑 __main__，全包在 if __name__ == "__main__" 里。
不碰 best_model.zip（只挂 LegActivityCallback，无 EvalCallback/SaveBest）。
"""
import os, sys, glob, shutil, gc, tempfile
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from train_humanoid_sac import LegActivityCallback, make_env


def check_tags():
    tmp = tempfile.mkdtemp(prefix="tb_tagcheck_")
    envs = VecMonitor(SubprocVecEnv([make_env("Humanoid-v4", i, 42) for i in range(4)]))
    m = SAC("MlpPolicy", envs, learning_rate=1e-5, buffer_size=4000, batch_size=16,
            train_freq=4, gradient_steps=4, ent_coef=0.05, target_entropy=-8.5,
            policy_kwargs={"net_arch": [512, 256]}, verbose=0, tensorboard_log=tmp)
    m.learn(total_timesteps=2000, callback=[LegActivityCallback()], log_interval=5,
            progress_bar=False)
    m.logger.close()
    m.env.close()

    files = glob.glob(os.path.join(tmp, "**", "events.out.tfevents.*"), recursive=True)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(files[0], size_guidance={"scalars": 0}) if files else EventAccumulator(tmp)
    acc.Reload()
    tags = sorted(acc.Tags()["scalars"])
    print("events 文件 %d 个, tags %d 个" % (len(files), len(tags)))
    print("train/* tags:", [t for t in tags if t.startswith("train")])

    ok = True
    for tag in ["train/action_std", "train/exploration_std"]:
        if tag in tags:
            v = [x.value for x in acc.Scalars(tag)]
            import numpy as np
            v = np.array(v)
            print("OK %-22s 写入成功: %2d 点, 范围 %.4f~%.4f 均值 %.4f"
                  % (tag, len(v), v.min(), v.max(), v.mean()))
            if len(v) < 2:
                ok = False
                print("   ⚠ 点数不足，门控可能太密")
        else:
            ok = False
            print("FAIL %-22s 未写入" % tag)

    m.replay_buffer = None
    del m
    shutil.rmtree(tmp, ignore_errors=True)
    gc.collect()
    print("\n结论: %s" % ("✅ 两个 tag 都真写入 tensorboard" if ok else "❌ 有问题"))


if __name__ == "__main__":
    check_tags()
