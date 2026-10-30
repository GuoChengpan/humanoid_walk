import imageio
import importlib.util
import gymnasium as gym
from stable_baselines3 import SAC
from train_humanoid_sac import WalkRewardWrapper, patch_mirror_symmetry

"""加载一个已训练好的策略，开窗口直接渲染看它走成什么样；也可以录一段 GIF。

    python3 test_humanoid.py

- 鼠标左键旋转 / 右键平移 / 滚轮缩放；空格暂停；ESC 退出。
- 换权重只改下面的 `MODEL_PATH`（归档位置见 README 的「项目结构」；
  注意 `best_model/best_model.zip` 会被每次训练覆盖）。
- 摔倒或到 1500 步上限会自动 reset，可以连续看多集；每集结束打印长度与累积 reward。
- 录 GIF 把 `EXPORT_GIF` 改成 True，并换一个不覆盖 README 素材的 `GIF_PATH`。
"""

MODEL_PATH = "./best_model_saved/round2/17_4/best_model.zip"

MAX_STEPS = 1500          # 与训练时的 max_episode_steps 一致
TOTAL_STEPS = 10000       # 实时渲染的总步数上限（≈ 6-7 集），Ctrl+C 可随时退出
EXPORT_GIF = False        # True = 录一段 GIF；False = 开窗口
GIF_PATH = "./humanoid_walk_gif/test.gif"
GIF_FRAMES = 600

model = SAC.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# 环境：先定渲染方式，再按权重所属世代选择接口（三选一，其余保持注释）
# ---------------------------------------------------------------------------
env = gym.make(
    "Humanoid-v4",
    render_mode="rgb_array" if EXPORT_GIF else "human",
    max_episode_steps=MAX_STEPS,
)

# ① 裸环境：era-1（V*）与 17_x 用这一个，不需要任何 wrapper。

# ② 18_x（含 18_8）：只加镜像 patch。
# patch_mirror_symmetry(env.unwrapped.model)

# ③ 19_1（T1 模板，训练摆幅 A=0.25）：patch + 当前 wrapper，构造后把 osc_A 改回 0.25。
# patch_mirror_symmetry(env.unwrapped.model)
# env = WalkRewardWrapper(env, swing_drive_weight=2.0, osc_enable=True)
# env.osc_A = 0.25

# ④ 19_2 / 19_3 / 19_4（T1 模板，摆幅 A=0.15）：patch + 当前 wrapper。
#   19_3 无归档源码，按同系列推定。
# patch_mirror_symmetry(env.unwrapped.model)
# env = WalkRewardWrapper(env, swing_drive_weight=2.0, osc_enable=True)

# ⑤ 19_5（T2 模板）：patch + 它自己归档的源码。
# patch_mirror_symmetry(env.unwrapped.model)
# spec = importlib.util.spec_from_file_location(
#     "m19_5", "best_model_saved/round2/SAC_19/19_5_厥着腚高速原地踏步/train_humanoid_sac.py")
# m = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(m)
# env = m.WalkRewardWrapper(env, swing_drive_weight=2.0, osc_enable=True)

# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
obs, _ = env.reset()

if EXPORT_GIF:
    frames = []
    for _ in range(GIF_FRAMES):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    imageio.mimsave(GIF_PATH, frames, fps=50, loop=0)
    print(f"已写入 {GIF_PATH}（{len(frames)} 帧）")
else:
    ep_len, ep_ret, ep_n = 0, 0.0, 0
    for _ in range(TOTAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_len += 1
        ep_ret += float(reward)
        if terminated or truncated:
            ep_n += 1
            cause = "摔倒(terminated)" if terminated else "到步数上限(truncated)"
            print(f"[ep {ep_n}] 长度={ep_len} 步, return={ep_ret:.0f}, {cause}")
            obs, _ = env.reset()
            ep_len, ep_ret = 0, 0.0
    env.close()
