import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import Wrapper
from stable_baselines3 import SAC  # 改为 SAC
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
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
        self.survival_base = 0.5          # 基础存活奖励
        self.survival_increment = 0.05    # 每步额外增加
        self.forward_weight = 3.0
        self.healthy_weight = 1.0
        self.ctrl_cost_weight = 0.005
        self.fall_penalty_weight = -50.0
        self.displacement_weight = 0.5

        # ---- 新加入：鼓励双腿活跃 ----
        self.limb_activity_weight = 0.2      # 双腿总活跃度奖励
        self.lazy_leg_bonus_weight = 10.0     # 额外鼓励“懒腿”的奖励
        
        # ---- 新加入：惩罚脚不动 ----
        self.foot_stuck_penalty_weight = 10.0 # 脚不动惩罚系数
        
        # 预存关节执行器索引，避免每步都查找
        self._actuator_indices = None

    def _get_actuator_indices(self):
        """获取左右腿关键关节的执行器索引"""
        if self._actuator_indices is not None:
            return self._actuator_indices
        model = self.env.unwrapped.model
        actuator_names = [model.actuator(i).name for i in range(model.nu)]
        indices = {}
        # 根据 Humanoid-v4 的常见执行器名称，请根据实际环境微调
        for side in ['left', 'right']:
            try:
                indices[f'{side}_knee'] = actuator_names.index(f'{side}_knee')
                indices[f'{side}_hip_x'] = actuator_names.index(f'{side}_hip_x')  # 代表性的髋关节
            except ValueError:
                # 如果名称不匹配，静默处理，不影响主逻辑
                pass
        self._actuator_indices = indices
        return indices        


    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_x = self.env.unwrapped.data.qpos[0]
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        mujoco_env = self.env.unwrapped
        x_pos = mujoco_env.data.qpos[0]
        x_velocity = mujoco_env.data.qvel[0]
        torso_height = mujoco_env.data.qpos[2]   # 使用 qpos


        # 位移奖励（取消，或者改为小权重并限制累积）
        # 如果你仍想保留一点位移激励，可以改为：只奖励正向位移增量，但限制每次不超过0.5
        displacement = x_pos - self.prev_x
        self.prev_x = x_pos
        displacement_reward = min(0.5, max(0, displacement)) * self.displacement_weight  # 有界

        # 指数速度奖励，但将速度截断在 [-0.5, 2.0] 之间，避免爆炸
        # v_clipped = np.clip(x_velocity, -0.5, 2.0)
        # forward_reward = self.forward_weight * (np.exp(max(0, v_clipped)) - 1.0)  # 指数但有限
        forward_reward = self.forward_weight * max(0, x_velocity)

        # 高度奖励（维持0.8~1.8）
        height_reward = self.healthy_weight * max(0, min(torso_height - 0.7, 0.8))

        # 生存奖励（固定低值）
        # self.step_count += 1
        # survival_bonus = min(1.0, self.survival_base + self.survival_increment * self.step_count)
        survival_bonus = 0.5

        # 控制成本
        ctrl_cost = self.ctrl_cost_weight * np.square(action).sum()

        # 摔倒惩罚
        fall_penalty = self.fall_penalty_weight if terminated else 0.0

# ---- 2. 新奖励：双腿活跃度 + 懒腿额外鼓励 ----
        activity_reward = 0.0
        lazy_leg_bonus = 0.0
        indices = self._get_actuator_indices()
        if indices:
            try:
                left_knee = action[indices['left_knee']]
                left_hip = action[indices['left_hip_x']]
                right_knee = action[indices['right_knee']]
                right_hip = action[indices['right_hip_x']]
                
                left_activity = abs(left_knee) + abs(left_hip)
                right_activity = abs(right_knee) + abs(right_hip)
                
                # 总活跃度奖励
                activity_reward = self.limb_activity_weight * (left_activity + right_activity)
                # 懒腿额外鼓励：min(左活跃, 右活跃) 越大，说明两条腿越均衡，给更多奖励
                lazy_leg_bonus = self.lazy_leg_bonus_weight * min(left_activity, right_activity)
            except KeyError:
                pass  # 索引不存在则跳过
        
        # ---- 3. 新惩罚：脚相对于身体的速度惩罚 ----
        body_vel = mujoco_env.data.qvel[0:3].copy()  # 身体线速度
        foot_vel_right = mujoco_env.data.body('right_foot').subtree_linvel.copy()
        foot_vel_left = mujoco_env.data.body('left_foot').subtree_linvel.copy()
        
        rel_vel_right = np.linalg.norm(foot_vel_right - body_vel)
        
        rel_vel_left = np.linalg.norm(foot_vel_left - body_vel)
        
        # 我们希望脚相对于身体的速度至少 > 0.5 m/s，否则视为“不动”
        stuck_threshold = 0.5
        foot_stuck_right = max(0.0, stuck_threshold - rel_vel_right)
        foot_stuck_left = max(0.0, stuck_threshold - rel_vel_left)
        foot_stuck_penalty = self.foot_stuck_penalty_weight * (foot_stuck_right + foot_stuck_left)
        

        reward = (forward_reward + 
                  displacement_reward + 
                  height_reward + 
                  fall_penalty +
                  activity_reward + 
                  lazy_leg_bonus+
                  survival_bonus -
                  foot_stuck_penalty -
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
    total_timesteps = 1_500_000
    seed = 42

    # ---- 训练环境 ----
    train_vec_env = SubprocVecEnv(
        [make_env(env_id, i, seed) for i in range(n_train_envs)]
    )
    # 如果需要，可以包装 VecMonitor，同时用于训练日志
    # from stable_baselines3.common.vec_env import VecMonitor
    train_vec_env = VecMonitor(train_vec_env)

    # ---- 评估环境（独立） ----
    eval_vec_env = SubprocVecEnv(
        [make_env(env_id, n_train_envs + i, seed) for i in range(n_eval_envs)]
    )
    # 评估环境也建议用 VecMonitor 包装，方便 EvalCallback 记录
    eval_vec_env = VecMonitor(eval_vec_env)

    # 策略网络结构（与PPO相同，SAC也支持）
    policy_kwargs = {
        'net_arch': [512, 256],
        'activation_fn': nn.Tanh
    }

    # ============ SAC 模型 ============
    model = SAC(
        'MlpPolicy',
        train_vec_env,
        learning_rate=lambda prog: 5e-4 * prog + 1e-5 * (1 - prog),  # 线性衰减
        buffer_size=1_500_000,
        batch_size=1024,
        tau=0.01,
        gamma=0.99,
        train_freq=512,
        gradient_steps=512,
        ent_coef="auto",              # 固定高熵探索
        target_entropy="auto",      # 动作维度17
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./sac_humanoid_walk/"
    )

    # ---------- 加载之前保存的模型 ----------
    model = SAC.load("./best_model/best_model", env=train_vec_env)
    # model.replay_buffer.reset()


    # 回调：达到阈值时停止训练
    callback = StopTrainingOnRewardThreshold(
        reward_threshold=10000,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_vec_env,
        callback_on_new_best=callback,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=2000,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,          # 关键：保持总步数计数，不从头算
        callback=eval_callback
    )
    model.save("sac_humanoid_walk_final")

    print("训练完成！模型已保存。")