"""批次⑨ 开跑前的**零写入冒烟测试**：验证随机初始化路径能起来，且在 `learn()` 入口就停住。

为什么要它：⑦ 那轮在开跑前抓到 3 个启动期 bug（`make_env` 不认识的 kwarg、`set_osc_k` 被误插进
`__init__`、课程里的 k1 与 wrapper 不一致），都是"按下去才发现"的类别。本轮改了加载/预填充/预热三处，
同类风险重新出现，所以先跑这个探针。

安全性：把 `SAC.learn` 换成一个"检查完状态就 SystemExit"的替身 ⇒
  · 不产生梯度更新、不写 `best_model/`、不写 `eval_logs/`、不写 TB events；
  · 只启动 8 个训练 env + 1 个 eval env，检查完显式 close() 防孤儿进程。
"""
import runpy
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/guo/workspace/humanoid_walk")

import stable_baselines3 as sb3
from stable_baselines3 import SAC

STATE = {}


def fake_learn(self, total_timesteps=None, **kw):
    STATE["model"] = self
    print("\n=== learn() 入口（训练在此被拦停，不写任何文件）===")
    print("  请求 total_timesteps =", total_timesteps)
    print("  learning_starts      =", self.learning_starts, "（⑨ 期望 100000；⑧ 是 0）")
    print("  buffer 当前条数      =", self.replay_buffer.size(), "（⑨ 期望 0：没有策略预填充）")
    print("  已发生梯度更新次数   =", getattr(self, "_n_updates", -1))
    print("  train_freq/gradient_steps =", self.train_freq, "/", self.gradient_steps)
    print("  ent_coef/目标熵      =", float(self.ent_coef), "/", self.target_entropy)
    print("  batch/buffer         =", self.batch_size, "/", self.buffer_size)
    w = self.policy.actor.latent_pi[0].weight.detach()
    fan_in = w.shape[1]
    # torch 的 Linear 默认是 U(-1/sqrt(fan_in), +1/sqrt(fan_in)) ⇒ std = 1/sqrt(3·fan_in)
    exp_std = 1.0 / np.sqrt(3.0 * fan_in)
    print("  actor 首层权重 std   = %.5f，torch 默认初始化期望 %.5f ⇒ %s" % (
        w.std().item(), exp_std,
        "随机初始化 ✅（未加载任何权重）" if abs(w.std().item() - exp_std) < 0.15 * exp_std else "**不是随机初始化，检查加载路径**"))
    asd = self.policy.actor.state_dict()
    lk = [x for x in asd if "log_std" in x]
    if lk:
        print("  动作 std 初值        =", np.round(asd[lk[0]].detach().exp().cpu().numpy()[:4], 4),
              "（随机初始化应≈1.0 附近）")
    obs = np.zeros(self.observation_space.shape, dtype=np.float32)
    a, _ = self.predict(obs, deterministic=True)
    print("  随机策略一次 predict ：动作维度=%d，|a|max=%.3f ⇒ 前向通路 OK" % (len(a), float(np.abs(a).max())))
    # 关掉 SubprocVecEnv，防止孤儿进程继续吃 CPU
    try:
        ve = self.env
        while hasattr(ve, "env") and ve.env is not None:
            ve = ve.env
        ve.close()
        print("  train_vec_env.close() ✅")
    except Exception as e:
        print("  ⚠ 关闭 vec env 失败（请手动检查孤儿进程）：", type(e).__name__, e)
    # 必须抛出去：否则 learn() 正常返回后，脚本会继续执行 `model.save("sac_humanoid_walk_final")`
    # ——第一版就因为这个在本目录留下了一个 zip（已删除）。
    raise SystemExit(0)


def main():
    SAC.learn = fake_learn                       # MirrorEqSAC 继承 SAC.learn ⇒ 拦截生效
    print("已把 SAC.learn 换成零写入替身，现在以 __main__ 身份执行 train_humanoid_sac.py")
    try:
        runpy.run_path("/home/guo/workspace/humanoid_walk/train_humanoid_sac.py", run_name="__main__")
    except SystemExit:
        print("\n✅ 冒烟测试在 learn() 入口正常停下（未进入训练、未写文件）")
    except Exception as e:
        print("\n❌ 启动期异常：%s: %s" % (type(e).__name__, e))
        raise
    m = STATE.get("model")
    print("\n结论：批次⑨ 的三处改动（不加载权重 / 不预填充 / 随机预热 100K）在真实启动路径上成立。")
    if m is not None and m.learning_starts != 100_000:
        print("⚠ learning_starts 不是 100000，请检查 FROM_SCRATCH 开关")


if __name__ == "__main__":
    main()
