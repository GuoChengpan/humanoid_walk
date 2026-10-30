import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import Wrapper
from collections import deque
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC,PPO  # 改为 SAC
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor,DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor


class WalkRewardWrapper(Wrapper):
    """
    重新设计Humanoid的奖励函数,强调前进速度,降低存活奖励。
    解决“原地站立”局部最优问题
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_x = None  # 用于计算位移
        self.step_count = 0
        self.forward_weight = 1.5
        self.healthy_weight = 1.0
        self.ctrl_cost_weight = 0.005
        self.fall_penalty_weight = -1000.0
        self.displacement_weight = 0.5

        self.gait_weight = 0.3       # gait reward  
        self.stride_weight = 1.0
        self.clearance_weight = 1.5    # 抬脚奖励
        self.drag_penalty_weight = 0.5 # 脚拖地惩罚权重
        self.left_contact_prev = True
        self.right_contact_prev = True
        self.left_takeoff_x = 0
        self.right_takeoff_x = 0
        self.balance_weight = 0.5    # 支撑稳定
        # 保存脚位置
        self.prev_left_x = None
        self.prev_right_x = None
        # gait phase
        self.phase = 0
        self.phase_frequency = 0.08
        # 左右脚接触相位差参数初始化
        self.phase_weight = 3.0
        self.contact_buffer_size = 60
        self.left_contact_buffer = deque(maxlen=self.contact_buffer_size)
        self.right_contact_buffer = deque(maxlen=self.contact_buffer_size)
            
    def compute_phase_reward(self):
        if len(self.left_contact_buffer) < self.contact_buffer_size:
            return 0

        left = np.array(self.left_contact_buffer)
        right = np.array(self.right_contact_buffer)
        # 去中心化
        left = left - left.mean()
        right = right - right.mean()
        best_score = -1
        # 搜索可能的相位差
        for lag in range(5, self.contact_buffer_size//2):
            shifted_right = np.roll(right,lag)
            # 相关性
            corr = np.mean(left * shifted_right)
            # 我们希望反相
            score = -corr
            best_score = max(best_score,score)
        # 映射到0~1
        reward = np.clip(best_score,0,1)

        return reward

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.left_contact_buffer.clear()
        self.right_contact_buffer.clear()
        self.prev_x = self.env.unwrapped.data.qpos[0]
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        mujoco_env = self.env.unwrapped
        x_velocity = mujoco_env.data.qvel[0]
        torso_height = mujoco_env.data.qpos[2]   # 使用 qpos
        # 获取双脚状态
        left_foot = mujoco_env.data.body("left_foot")
        right_foot = mujoco_env.data.body("right_foot")
        left_z = left_foot.xpos[2]
        right_z = right_foot.xpos[2]
        left_x = left_foot.xpos[0]
        right_x = right_foot.xpos[0]


        # 速度奖励
        forward_reward = self.forward_weight * max(0, x_velocity)

        # 高度奖励（维持0.5~1.8）
        height_reward = self.healthy_weight * max(0, min(torso_height - 1.0, 0.5))

        # 生存奖励（固定低值）
        survival_bonus = 1.0

        # 控制成本
        ctrl_cost = self.ctrl_cost_weight * np.square(action).sum()

        # 摔倒惩罚
        fall_penalty = self.fall_penalty_weight if terminated else 0.0

        # #左右脚交替奖励/单脚支撑奖励
        # left_contact = (left_z < 0.08)
        # right_contact = (right_z < 0.08)
        # if left_contact and right_contact:
        #     gait_reward = 0.3
        # elif left_contact != right_contact:
        #     gait_reward = 1.0
        # else:
        #     gait_reward = 0
        # gait_reward *= self.gait_weight

        # # 抬腿奖励
        # if left_contact and not right_contact:
        #     swing_height = right_z
        # elif right_contact and not left_contact:
        #     swing_height = left_z
        # else:
        #     swing_height = 0
        # clearance_reward = np.clip(swing_height / 0.12,0,1)
        # clearance_reward *= self.clearance_weight

        # # 步幅奖励
        # stride_reward = 0
        # # 左脚事件
        # if self.left_contact_prev and not left_contact:
        #     # 左脚离地
        #     self.left_takeoff_x = left_x
        # if not self.left_contact_prev and left_contact:
        #     # 左脚重新落地
        #     left_stride = (left_x -self.left_takeoff_x)
        #     if left_stride>0:
        #         stride_reward += np.clip(left_stride / 0.5,0,1)
        # # 右脚事件
        # if self.right_contact_prev and not right_contact:
        #     self.right_takeoff_x = right_x
        # if not self.right_contact_prev and right_contact:
        #     right_stride = (right_x -self.right_takeoff_x)
        #     if right_stride>0:
        #         stride_reward += np.clip(right_stride / 0.5,0,1)
        # self.left_contact_prev = left_contact
        # self.right_contact_prev = right_contact
        # stride_reward *= self.stride_weight

        # # 身体稳定奖励
        # quat = mujoco_env.data.qpos[3:7]
        # r = Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]])
        # roll,pitch,yaw = r.as_euler("xyz")
        # balance_reward = np.exp(-(abs(pitch)+abs(roll)))
        # orientation_reward=np.exp(-abs(yaw))
        # balance_reward = (self.balance_weight*balance_reward + orientation_reward*0.2)

        # # 左右脚接触相位差
        # self.left_contact_buffer.append(int(left_contact))
        # self.right_contact_buffer.append(int(right_contact))
        # phase_reward = self.compute_phase_reward()*self.phase_weight

        # # 脚拖地惩罚
        # drag_penalty=0
        # if not left_contact:
        #     drag_penalty += abs(mujoco_env.data.body("left_foot").cvel[0])
        # if not right_contact:
        #     drag_penalty += abs(mujoco_env.data.body("right_foot").cvel[0])
        # drag_penalty *= self.drag_penalty_weight 

        reward = (forward_reward + 
                # gait_reward +
                # clearance_reward +
                # stride_reward +
                # balance_reward +
                height_reward + 
                fall_penalty +
                # phase_reward +
                survival_bonus -
                # drag_penalty -
                ctrl_cost)

        return obs, reward, terminated, truncated, info


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = WalkRewardWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == "__main__":
    env_id = "Humanoid-v4"
    n_train_envs = 8    # 训练环境数量
    n_eval_envs = 1     # 评估环境数量（通常 1 个即可）
    total_timesteps = 6_000_000
    seed = 42

    # ---- 训练环境 ----
    train_vec_env = SubprocVecEnv(
        [make_env(env_id, i, seed) for i in range(n_train_envs)]
    )
    # 如果需要，可以包装 VecMonitor，同时用于训练日志
    train_vec_env = VecMonitor(train_vec_env)

    # ---- 评估环境（独立） ----
    eval_vec_env = DummyVecEnv([make_env(env_id, 100, seed)])
    # 评估环境也建议用 VecMonitor 包装，方便 EvalCallback 记录
    eval_vec_env = VecMonitor(eval_vec_env)

    # 策略网络结构（与PPO相同，SAC也支持）
    policy_kwargs = {
        'net_arch': [512, 256],
        'activation_fn': nn.Tanh
    }

    # ============ PPO 模型 ============
    model = PPO(
        'MlpPolicy',
        train_vec_env,
        learning_rate=lambda progress: 5e-4 * progress + 1e-5 * (1 - progress),
        n_steps=4096,
        batch_size=512,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_humanoid_walk/"
    )

    # ---------- 加载之前保存的模型 ----------
    # model = PPO.load("./best_model/best_model", env=train_vec_env)
    # model.replay_buffer.reset()


    # 回调：达到阈值时停止训练
    callback = StopTrainingOnRewardThreshold(
        reward_threshold=8000,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_vec_env,
        callback_on_new_best=callback,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=20000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=total_timesteps,
        # reset_num_timesteps=False,          # 关键：保持总步数计数，不从头算
        callback=eval_callback
    )
    model.save("ppo_humanoid_walk_final")

    print("训练完成！模型已保存。")