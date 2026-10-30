"""端到端核验：真 Logger + 真 callback + 真 env + 真 infos，确认 gait/stride_balance_penalty 真被写入"""
import os, sys, tempfile, shutil
sys.path.insert(0, "/home/guo/workspace/humanoid_walk")
os.chdir("/home/guo/workspace/humanoid_walk")
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.logger import Logger
from train_humanoid_sac import WalkRewardWrapper, LegActivityCallback

USER = "best_model/tmp_model_data/best_model.zip"

m = SAC.load(USER, buffer_size=20000)
tmp = tempfile.mkdtemp()
m._logger = Logger(tmp, [])          # SAC.load 不建 logger，手动注入
cb = LegActivityCallback()
cb.init_callback(m)

env = WalkRewardWrapper(gym.make("Humanoid-v4", max_episode_steps=1500))
obs, _ = env.reset(seed=23)
n_step = 0
max_pen = 0.0
for _ in range(1500):
    a, _ = m.predict(obs, deterministic=True)
    obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.float32))
    n_step += 1
    max_pen = max(max_pen, float(info.get("stride_balance_penalty", 0.0)))
    cb.locals = {"infos": [info], "actions": [a]}
    cb.n_calls = 1
    if not cb.on_step():
        raise SystemExit("on_step 返回假值，训练会中断")
    if te or tr:
        break

env.close()
vals = dict(m.logger.name_to_value)
if hasattr(m.logger, "exp_writer") and hasattr(m.logger.exp_writer, "close"):
    m.logger.exp_writer.close()
shutil.rmtree(tmp, ignore_errors=True)

want = ["gait/stride_balance_penalty", "gait/last_left_stride", "gait/last_right_stride"]
print("跑完 %d 步，env 侧观测到的最大 penalty = %.4f" % (n_step, max_pen))
for k in want:
    v = vals.get(k, "未记录")
    flag = "✓" if k in vals else "✗"
    print("  %s %s = %s" % (flag, k, v))
print("\n对照（既有 tag，验证测试本身有效）: gait/activity_difference = %s" % vals.get("gait/activity_difference"))
