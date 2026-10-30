import os,json
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import time as _time
import gc
from gymnasium import Wrapper
from collections import deque
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC,PPO  # 改为 SAC
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold,BaseCallback,CallbackList
from stable_baselines3.common.utils import polyak_update
from mirror_map import MirrorMap, patch_mirror_symmetry


class LegActivityCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_gait_data = {}  # 新增：保存最近记录的 gait 指标

    def _on_step(self):
        infos = self.locals["infos"]
        # 探索强度监控，两个口径都放 for 循环外——Logger.record 是覆盖不是平均，
        # 循环内会被 8 个 env 各记一次只剩最后一个。
        #
        # train/action_std: (n_envs,17) env 空间动作的跨环境 std，看趋势。
        #   经 unscale_action 已在 env 空间(±0.4)；560K 基线 ≈0.21。
        #   被 tanh 压缩、天花板 0.4，动态范围有限，别当绝对阈值。
        #
        # train/exploration_std: exp(log_std) 折回 env 空间，未经 tanh 压缩，
        #   是 ent_coef 的直接作用量（actor_loss 含 +ent_coef*log_std.mean()）。
        #   560K 基线 ≈0.267（同策略 realized 采样 std ≈0.181）。
        #   对照注入截断高斯的悬崖表 σ=0.05→89.8% / 0.10→24.4% / 0.18→-13.3%；
        #   但原生 tanh 采样同幅度下比注入高斯好 ~367/集（分布近均匀、非高斯），
        #   所以此值只能看升降趋势，不能直接当 σ 判生死。
        #   门控用 self.n_calls（on_step 单调递增的批次计数），不能用 num_collected_steps——
        #   后者是 collect_rollouts 的局部变量，每次调用都重置为 0、只跑 1..train_freq，
        #   train_freq < 100 时永远命中不了；也不能用 model.num_timesteps % 500，
        #   它是 env 数的倍数（8 envs → 8,16,24...），500 不是 8 的倍数同样永远命中不了。
        #   record 是覆盖，抽样与每步量输出相同，省掉 99% 的前向开销。
        actions = self.locals.get("actions")
        if actions is not None:
            self.logger.record("train/action_std", float(np.std(np.asarray(actions), axis=0).mean()))
        if self.n_calls % 100 == 0:
            obs = self.model._last_obs   # 本轮 actions 对应的 obs；是 numpy（off_policy:509 赋的是 env.step 输出）
            if obs is not None:
                space = self.model.action_space
                scale = float((space.high[0] - space.low[0]) / 2.0)  # 0.4: actor 空间 → env 空间
                obs_t, _ = self.model.policy.obs_to_tensor(obs)        # 走 SB3 自己的转换（device/dtype/dict space）
                with th.no_grad():
                    _, log_std, _ = self.model.policy.actor.get_action_dist_params(obs_t)
                self.logger.record("train/exploration_std", float(th.exp(log_std).mean()) * scale)
        for info in infos:
            if "mean_left_swing" in info:
                self.logger.record("gait/mean_left_swing", info["mean_left_swing"])
                self.logger.record("gait/mean_right_swing", info["mean_right_swing"])
            self.logger.record("gait/flight_phase_penalty", info["flight_phase_penalty"])
            self.logger.record("gait/activity_difference", info.get("activity_difference",0))
            self.logger.record("gait/phase_reward", info.get("phase_reward",0))
            self.logger.record("gait/ctrl_cost", info.get("ctrl_cost",0))

            self.logger.record("gait/position_swap_reward", info.get("position_swap_reward",0))
            self.logger.record("gait/contact_phase_reward", info.get("contact_phase_reward",0))
            self.logger.record("gait/last_contact_phase_reward", info.get("last_contact_phase_reward",0.0))
            self.logger.record("gait/stride_bonus", info.get("stride_bonus",0))
            self.logger.record("gait/stride_event_count", info.get("stride_event_count",0))
            self.logger.record("gait/stride_bonus_event_count", info.get("stride_bonus_event_count",0))
            self.logger.record("gait/stride_length", info.get("stride_length",0.0))
            self.logger.record("gait/last_stride_bonus", info.get("last_stride_bonus",0.0))
            self.logger.record("gait/stride_bonus_rate", info.get("stride_bonus_rate",0.0))
            self.logger.record("gait/hip_xz_angle_penalty", info.get("hip_xz_angle_penalty",0))
            self.logger.record("gait/short_stride_penalty", info.get("short_stride_penalty",0))
            self.logger.record("gait/last_short_stride_penalty", info.get("last_short_stride_penalty",0.0))
            self.logger.record("gait/short_stride_event_rate", info.get("short_stride_event_rate",0))
            self.logger.record("gait/negative_stride_penalty", info.get("negative_stride_penalty",0))
            self.logger.record("gait/last_negative_stride_penalty", info.get("last_negative_stride_penalty",0.0))
            self.logger.record("gait/negative_stride_event_count", info.get("negative_stride_event_count",0))
            self.logger.record("gait/negative_stride_rate", info.get("negative_stride_rate",0.0))
            self.logger.record("gait/stride_balance_penalty", info.get("stride_balance_penalty",0.0))
            self.logger.record("gait/last_left_stride", info.get("last_left_stride",0.0))
            self.logger.record("gait/last_right_stride", info.get("last_right_stride",0.0))
            
            self.logger.record("gait/support_sym_penalty", info.get("support_sym_penalty",0))
            self.logger.record("gait/lateral_penalty", info.get("lateral_penalty",0))
            self.logger.record("gait/alternate_penalty", info.get("alternate_penalty",0))
            self.logger.record("gait/stall_penalty", info.get("stall_penalty",0))
            self.logger.record("gait/stall_max", info.get("stall_max",0))
            self.logger.record("gait/body_advance", info.get("body_advance", 0.0))
            self.logger.record("gait/speed", info.get("speed", 0))
            self.logger.record("gait/position_swap_rate", info.get("position_swap_rate", 0))
            self.logger.record("gait/alternate_reward_rate", info.get("alternate_reward_rate", 0))
            self.logger.record("gait/alternate_penalty_rate", info.get("alternate_penalty_rate", 0))
            self.logger.record("gait/forward_reward", info.get("forward_reward", 0))
            self.logger.record("gait/forward_gate", info.get("forward_gate", 1.0))
            self.logger.record("gait/height_reward", info.get("height_reward", 0))
            self.logger.record("gait/net_balance", info.get("net_balance", 0))
            self.logger.record("gait/pitch_angle", info.get("pitch_angle", 0))
            self.logger.record("gait/roll_angle", info.get("roll_angle", 0))
            self.logger.record("gait/swing_forward_reward", info.get("swing_forward_reward", 0))
            self.logger.record("gait/last_swing_forward_reward", info.get("last_swing_forward_reward", 0.0))
                

            # 新增：保存到实例属性，供 SaveExtraParamsOnBest 使用
            self.last_gait_data.update({
                "activity_difference": info.get("activity_difference",0),
                "phase_reward": info.get("phase_reward",0),
                "mean_left_swing": info.get("mean_left_swing",0),
                "mean_right_swing": info.get("mean_right_swing",0),
                "position_swap_reward": info.get("position_swap_reward",0),
                "contact_phase_reward": info.get("contact_phase_reward",0),
                "last_contact_phase_reward": info.get("last_contact_phase_reward",0.0),
                "stride_bonus": info.get("stride_bonus",0),
                "stride_event_count": info.get("stride_event_count",0),
                "stride_bonus_event_count": info.get("stride_bonus_event_count",0),
                "stride_length": info.get("stride_length",0.0),
                "last_stride_bonus": info.get("last_stride_bonus",0.0),
                "stride_bonus_rate": info.get("stride_bonus_rate",0.0),
                "hip_xz_angle_penalty": info.get("hip_xz_angle_penalty",0),
                "support_sym_penalty":info.get("support_sym_penalty",0),
                "lateral_penalty": info.get("lateral_penalty",0),
                "alternate_penalty": info.get("alternate_penalty",0),
                "ctrl_cost": info.get("ctrl_cost",0),
                "forward_reward": info.get("forward_reward", 0),
                "height_reward": info.get("height_reward", 0),
                "net_balance": info.get("net_balance", 0),
                "pitch_angle": info.get("pitch_angle", 0),
                "roll_angle": info.get("roll_angle", 0),
                "swing_forward_reward": info.get("swing_forward_reward", 0),
                "last_swing_forward_reward": info.get("last_swing_forward_reward", 0.0),
                "body_advance": info.get("body_advance", 0.0),
                "speed": info.get("speed", 0.0),
                "position_swap_rate": info.get("position_swap_rate", 0.0),
                "alternate_reward_rate": info.get("alternate_reward_rate", 0.0),
                "alternate_penalty_rate": info.get("alternate_penalty_rate", 0.0),     
                "short_stride_penalty": info.get("short_stride_penalty", 0.0), 
                "last_short_stride_penalty": info.get("last_short_stride_penalty", 0.0), 
                "short_stride_event_rate": info.get("short_stride_event_rate", 0.0), 
                "negative_stride_penalty": info.get("negative_stride_penalty", 0.0), 
                "last_negative_stride_penalty": info.get("last_negative_stride_penalty", 0.0), 
                "negative_stride_event_count": info.get("negative_stride_event_count", 0.0), 
                "negative_stride_rate": info.get("negative_stride_rate", 0.0),
                "stride_balance_penalty": info.get("stride_balance_penalty", 0.0),
                "last_left_stride": info.get("last_left_stride", 0.0),
                "last_right_stride": info.get("last_right_stride", 0.0),
                "flight_phase_penalty": info.get("flight_phase_penalty", 0.0),
                "forward_gate": info.get("forward_gate", 1.0)
            })
        return True


class SaveExtraParamsOnBest(BaseCallback):
    def __init__(self, save_path, leg_callback, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.leg_callback = leg_callback  # 引用 LegActivityCallback 实例
        os.makedirs(self.save_path, exist_ok=True)
        self.log_file = os.path.join(self.save_path, "best_extra_logs.jsonl")

    def _on_step(self) -> bool:
        if hasattr(self.parent, 'best_mean_reward'):
            # 1. 从 EvalCallback 获取当前评估的结果
            eval_mean_reward = getattr(self.parent, 'last_mean_reward', None)
            if hasattr(self.parent, 'evaluations_length') and self.parent.evaluations_length:
                eval_mean_ep_length = float(np.mean(self.parent.evaluations_length[-1]))
            else:
                eval_mean_ep_length = None
            # 2. 从 LegActivityCallback 获取最近记录的 gait 数据
            gait_data = getattr(self.leg_callback, 'last_gait_data', {})

            # 3. 尝试从 logger 获取训练损失（可能已被 dump 清空，作为备选）
            log_dict = self.model.logger.name_to_value

            def _json_safe(value):
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, dict):
                    return {str(k): _json_safe(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_json_safe(v) for v in value]
                return value

            extra_data = {
                "timestep": int(self.num_timesteps),
                "eval/mean_ep_length": eval_mean_ep_length,
                "eval/mean_reward": eval_mean_reward,
                "gait/activity_difference": gait_data.get("activity_difference"),
                "gait/phase_reward": gait_data.get("phase_reward"),
                "gait/mean_left_swing": gait_data.get("mean_left_swing"),
                "gait/mean_right_swing": gait_data.get("mean_right_swing"),
                "gait/position_swap_reward": gait_data.get("position_swap_reward"),
                "gait/contact_phase_reward": gait_data.get("contact_phase_reward"),
                "gait/last_contact_phase_reward": gait_data.get("last_contact_phase_reward"),
                "gait/stride_bonus": gait_data.get("stride_bonus"),
                "gait/stride_event_count": gait_data.get("stride_event_count"),
                "gait/stride_bonus_event_count": gait_data.get("stride_bonus_event_count"),
                "gait/stride_length": gait_data.get("stride_length"),
                "gait/last_stride_bonus": gait_data.get("last_stride_bonus"),
                "gait/stride_bonus_rate": gait_data.get("stride_bonus_rate"),
                "gait/hip_xz_angle_penalty": gait_data.get("hip_xz_angle_penalty"),
                "gait/support_sym_penalty": gait_data.get("support_sym_penalty"),
                "gait/lateral_penalty": gait_data.get("lateral_penalty"),
                "gait/alternate_penalty": gait_data.get("alternate_penalty"),
                "gait/forward_reward": gait_data.get("forward_reward"),
                "gait/height_reward": gait_data.get("height_reward"),
                "gait/net_balance": gait_data.get("net_balance"),
                "gait/pitch_angle": gait_data.get("pitch_angle"),
                "gait/roll_angle": gait_data.get("roll_angle"),
                "gait/swing_forward_reward": gait_data.get("swing_forward_reward"),
                "gait/last_swing_forward_reward": gait_data.get("last_swing_forward_reward"),
                "gait/body_advance": gait_data.get("body_advance"),
                "gait/speed": gait_data.get("speed"),
                "gait/position_swap_rate": gait_data.get("position_swap_rate"),
                "gait/alternate_reward_rate": gait_data.get("alternate_reward_rate"),
                "gait/alternate_penalty_rate": gait_data.get("alternate_penalty_rate"),  
                "gait/ctrl_cost":gait_data.get("ctrl_cost"),
                "gait/short_stride_penalty": gait_data.get("short_stride_penalty"),
                "gait/last_short_stride_penalty": gait_data.get("last_short_stride_penalty"),
                "gait/short_stride_event_rate": gait_data.get("short_stride_event_rate"),
                "gait/negative_stride_penalty": gait_data.get("negative_stride_penalty"),
                "gait/last_negative_stride_penalty": gait_data.get("last_negative_stride_penalty"),
                "gait/negative_stride_event_count": gait_data.get("negative_stride_event_count"),
                "gait/negative_stride_rate": gait_data.get("negative_stride_rate"),
                "gait/last_left_stride": gait_data.get("last_left_stride"),
                "gait/last_right_stride": gait_data.get("last_right_stride"),
                "gait/flight_phase_penalty": gait_data.get("flight_phase_penalty"),
                "gait/forward_gate": gait_data.get("forward_gate")
            }

            extra_data = {key: _json_safe(value) for key, value in extra_data.items()}

            # 追加写入 jsonl
            with open(self.log_file, "a") as f:
                f.write(json.dumps(extra_data) + "\n\n")

            if self.verbose > 0:
                print(f"\n✅ 新最佳模型日志已追加至 {self.log_file}")
        return True

class SaveBestModelOnNewBest(BaseCallback):
    """
    当 EvalCallback 检测到新的最佳模型时，保存一份带时间步的副本。
    不会覆盖已有文件。
    """
    def __init__(self, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 此回调只在 EvalCallback 发现新最佳时被调用
        if self.model is not None:
            timestep = self.num_timesteps
            path = os.path.join(self.save_dir, f"best_model_{timestep}.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"✅ 新最佳模型已保存至 {path}")
        return True


class WalkRewardWrapper(Wrapper):
    """
    重新设计Humanoid的奖励函数,强调前进速度,降低存活奖励。
    解决“原地站立”局部最优问题
    """
    def __init__(self, env):
        super().__init__(env)
        self.initial_y = None
        self.forward_weight = 0.4
        self.forward_gate_ref = 0.15   # 【对称门控·终线】0.12→0.15：增量冲击仅-0.05/步（gate 0.67→0.53），
                                       # 填谷可吸收；拉力-20%。左步幅均值目标 0.12→0.15
        self.healthy_weight = 1.0   # 【填谷】0.5→1.0：抬高过渡期（半速弹跳）的地板，Q 不塌、螺旋不触发；站立仍被移动门挡住
        self.ctrl_cost_weight = 0.005
        self.fall_penalty_weight = -50.0
        self.stride_weight = 1.0
        self.balance_weight = 0.5
        self.phase_weight = 1.0
        self.activity_rom_weight = 0.6
        self.activity_vel_weight = 0.4
        self.activity_diff_penalty_weight = 0.5
        self.support_sym_weight = 2.0  # 回退：原值为2.0，过高惩罚导致静止策略
        self.diff_scale = 0.18          # 【新增】diff归一化缩放因子
        self.prev_diff = None          # 【新增】上次触地时的diff值
        self.alternate_penalty_weight = 2.3  # 【新增】无交替惩罚权重
        # 【新增】换脚停滞惩罚：连续 N 次落地没换脚就罚，超出部分封顶防无界
        self.stall_threshold = 4
        self.stall_cap = 4
        self.stall_weight = 0.5
        self.lateral_weight = 0.1
        self.stride_balance_weight = 0.5   # 左右步幅对称惩罚。560K 上 0.5 → 27.0/集（3.6% reward）；2.0 在 flail 期实测爆到基线 9 倍并反向奖励摔倒
        self.stride_event_count = 0
        self.stride_bonus_event_count = 0
        self.last_stride_length = 0.0
        self.last_left_stride = None   # 左脚落地时 left_x - right_x，与右脚配对
        self.last_right_stride = None  # 右脚落地时 right_x - left_x
        self.last_contact_phase_reward = 0.0
        self.last_swing_forward_reward = 0.0
        self.last_stride_bonus = 0.0
        self.short_stride_event_count = 0
        self.last_short_stride_penalty = 0.0
        self.negative_stride_event_count = 0
        self.last_negative_stride_penalty = 0.0
        self.prev_body_x_per_frame = None
        self.speed = 0.0

        self.left_contact_prev = True
        self.right_contact_prev = True
        self.left_takeoff_x = None
        self.right_takeoff_x = None

        # 【新增】触地计数器
        self.total_land_count = 0
        self.position_swap_land_count = 0
        self.alternate_reward_count = 0
        self.alternate_penalty_count = 0
        self.landings_since_swap = 0
        self.flight_time_penalty_weight = 0.5
        self.flight_time_frames = 0
        self.prev_left_hip_angle = 0.0
        self.prev_right_hip_angle = 0.0
        self.leg_activity_buffer_size = 50
        self.left_hip_angle_buffer = deque(maxlen=self.leg_activity_buffer_size)
        self.right_hip_angle_buffer = deque(maxlen=self.leg_activity_buffer_size)
        self.left_leg_vel_buffer = deque(maxlen=self.leg_activity_buffer_size)
        self.right_leg_vel_buffer = deque(maxlen=self.leg_activity_buffer_size)

        self.foot_diff_buffer = deque(maxlen=120)
        self.left_contact_hist = deque(maxlen=50)
        self.right_contact_hist = deque(maxlen=50)

        self.left_contact_count = 0
        self.right_contact_count = 0
        # 脚底拖行惩罚追踪
        self.drag_threshold = 0.1
        self.drag_weight = 0.2
        self.total_steps = 0
        self.left_swing_steps = 0
        self.right_swing_steps = 0
        self.left_swing_total_steps = 0
        self.right_swing_total_steps = 0
        self.left_swing_history = []
        self.right_swing_history = []
        self.left_stride_history = deque(maxlen=20)
        self.right_stride_history = deque(maxlen=20)

        self.leg_joint_ids = None

        # 位置交换追踪
        self.prev_position_state = None
        self.speed = 0.0
        self.curr_position_state = 'centered'
        self.left_foot_drag_start_x = None
        self.right_foot_drag_start_x = None
        self.prev_body_x = None  # ⚠️ 之前漏了这个

        self.position_swap_weight = 1.0  # 【稠密化降权】3.0→1.0：事件→每步稠密后平均贡献膨胀 3-5 倍，
                                         # 降至 1.0 保持与旧事件版同等的平均占比（~0.4/步），避免压制其他项
        self.gait_period = 60       # 【相位时钟】步态周期（步）≈0.9s：前半左脚在前、后半右脚在前；
                                    # 与实际步频不合拍时调此值
        self.contact_phase_weight = 0.3
        # 位置交换调试日志
        self.foot_diff = 0.0
        self.body_advance = 0.0

        self.left_swing_hip_total = 0.0
        self.left_swing_time = 0
        self.left_swing_speed_total = 0.0

        self.right_swing_hip_total = 0.0
        self.right_swing_time = 0
        self.right_swing_speed_total = 0.0

        self.left_stance_time = 0
        self.left_stance_speed_total = 0.0

        self.right_stance_time = 0
        self.right_stance_speed_total = 0.0



    def compute_position_swap_reward(self, left_x, right_x, body_x, left_contact, right_contact):
        """检测左右脚前后位置交换、奖励左脚领先落地、惩罚连续不换脚"""
        swap_reward = 0.0
        stall_penalty = 0.0
        self.last_landing_swapped = False   # 非落地帧恒 False，供步幅门控读取

        # 计算前后脚位置差（用于调试）
        self.foot_diff = abs(left_x - right_x)  # 正数表示左脚在前

        # 只在脚触地时检查交替
        left_just_landed = left_contact and not self.left_contact_prev
        right_just_landed = right_contact and not self.right_contact_prev

        # 计算身体前进量（用于调试）
        if self.prev_body_x is not None:
            self.body_advance = body_x - self.prev_body_x
        else:
            self.body_advance = 0

        if left_just_landed or right_just_landed:
            # 位置状态检测（保留：last_landing_swapped 供 stride_bonus 门控）
            if left_x > right_x + 0.05:
                self.curr_position_state = 'left_ahead'
            elif right_x > left_x + 0.05:
                self.curr_position_state = 'right_ahead'
            else:
                self.curr_position_state = 'centered'

            swapped = False
            if ((self.prev_position_state == 'left_ahead' and self.curr_position_state == 'right_ahead') or
                (self.prev_position_state == 'right_ahead' and self.curr_position_state == 'left_ahead')):
                self.position_swap_land_count += 1
                swapped = True
            self.last_landing_swapped = swapped

            if swapped:
                self.landings_since_swap = 0
            else:
                self.landings_since_swap += 1
            over = max(0, self.landings_since_swap - self.stall_threshold)
            stall_penalty = self.stall_weight * min(over, self.stall_cap)

            self.prev_body_x = body_x
            self.prev_position_state = self.curr_position_state

        # 【稠密相位对齐】时钟 φ：前半周期左脚应在前(+)、后半周期右脚应在前(-)，
        # 每步按脚位置差与节奏目标的对齐度给分——时序模板显式化，不再依赖事件
        self.gait_phase = (self.total_steps % self.gait_period) / self.gait_period
        target = 1.0 if self.gait_phase < 0.5 else -1.0
        gap_norm = (left_x - right_x) / self.diff_scale          # 左前为正
        align = max(0.0, 1.0 - abs(gap_norm - target))            # 完全对齐=1，反向=0
        swap_reward = self.position_swap_weight * align

        return swap_reward, stall_penalty
    
    
    def compute_contact_phase_consistency(self, left_contact, right_contact,
                                        left_foot_vel, right_foot_vel):
        """检查支撑相和摆动相的合理性
        left_foot_vel: 左脚水平线速度（m/s），来自 body("left_foot").xvel[:2] 的模长
        """
        reward = 0.0
        
        if left_contact and not right_contact:
            # 左脚支撑：希望脚几乎不动；右腿摆动：希望脚快速前移
            stance_quality = 1.0 - np.clip(left_foot_vel / 1.0, 0, 1)
            swing_quality  = np.clip(right_foot_vel / 2.0, 0, 1)
            reward = (stance_quality + swing_quality) / 2
        elif right_contact and not left_contact:
            stance_quality = 1.0 - np.clip(right_foot_vel / 1.0, 0, 1)
            swing_quality  = np.clip(left_foot_vel / 2.0, 0, 1)
            reward = (stance_quality + swing_quality) / 2
        return reward

            
    def compute_hip_phase_reward(self):
        if len(self.left_hip_angle_buffer) < self.leg_activity_buffer_size:
            return 0
        left = np.array(self.left_hip_angle_buffer)
        right = np.array(self.right_hip_angle_buffer)

        # 前摆幅度（用原始信号，非去中心化）
        # 【改】用前摆幅度 max(0, -min) 替代 std。std 对位置盲目——左腿永远伸展
        # （hip_y 全正 0.04~0.42）的 std 和正常摆动一样大，phase 分白送。
        # 前摆幅度直接测"髋弯到多前"：左腿从未向前弯 → forward=0 → activity=0
        # → 当前 reward 直接归零，任何前弯立刻产生奖励。min 确保弱腿限幅，
        # 策略无法靠单腿刷分——只有弱腿前摆才能增加 reward。
        left_forward = max(0.0, -np.min(left))
        right_forward = max(0.0, -np.min(right))

        # 去中心化（仅用于反相关系数）
        left = left - left.mean()
        right = right - right.mean()
        # 反相关系数
        corr = np.corrcoef(left, right)[0,1]
        anti_phase = (1 - corr) / 2.0

        # 混合活动度：前摆为主（权重 1），振荡为底分（权重 0.1）
        # 纯前摆公式下 warm-start policy 左腿 forward=0 → activity=0 → reward 完全归零，
        # 模型失去 phase 奖励来源导致 actor 爆炸。加 0.1×std 底分防止完全归零，
        # 同时保持前摆作为主要梯度方向。
        activity = min(left_forward, right_forward) + 0.1 * min(np.std(left), np.std(right))
        activity = np.clip(activity / 0.5, 0, 1)  # 0.5 rad (~28°) 为典型前摆幅度

        reward = anti_phase * activity
        return reward

    def compute_leg_activity(self,hip_buffer,vel_buffer):
        if len(hip_buffer) < 40:
            return 0.0
        # 1. ROM activity
        hip_rom = (max(hip_buffer)-min(hip_buffer))
        rom_activity = hip_rom
        # 2. velocity activity
        vel_activity = np.mean(np.abs(np.array(vel_buffer)))
        # normalize
        rom_activity = np.clip(rom_activity / 3.0,0,1)
        vel_activity = np.clip(vel_activity / 10.0,0,1)
        # fusion
        activity = (self.activity_rom_weight *rom_activity+self.activity_vel_weight *vel_activity)
        return activity
    
    def compute_alternate_penalty(self, left_x, right_x, left_contact, right_contact):
        # [新] 检测相邻两次触地时前后关系是否发生翻转
        diff = (right_x - left_x) / self.diff_scale
        if abs(diff) < 0.01:return 0
        res = 0
        if left_contact and not self.left_contact_prev:
            res = self._check_cycle(diff)
        elif right_contact and not self.right_contact_prev:
            res = self._check_cycle(diff)

        # 统计触发次数，用于日志监控
        if res < 0:
            self.alternate_reward_count += 1
        elif res > 0:
            self.alternate_penalty_count += 1

        return res

    def _check_cycle(self, current_diff):
        # [新] 检查相邻两次触地的diff符号是否翻转
        if self.prev_diff is None:
            self.prev_diff = current_diff
            return 0.0
        product = self.prev_diff * current_diff
        self.prev_diff = current_diff
        # 完成交替，奖励，没完成，惩罚，且交替步伐越大越接近，奖励越多
        
        return product

    def compute_foot_drag_penalty(self, left_contact, right_contact, left_x, right_x):
        """惩罚触地期间脚随身体滑动（拖行 vs 正常支撑）"""
        drag_penalty = 0.0
        if left_contact:
            if self.left_foot_drag_start_x is None:
                self.left_foot_drag_start_x = left_x
            drag = abs(left_x - self.left_foot_drag_start_x)
            if drag > self.drag_threshold:
                drag_penalty += (drag - self.drag_threshold)
        else:
            self.left_foot_drag_start_x = None
        if right_contact:
            if self.right_foot_drag_start_x is None:
                self.right_foot_drag_start_x = right_x
            drag = abs(right_x - self.right_foot_drag_start_x)
            if drag > self.drag_threshold:
                drag_penalty += (drag - self.drag_threshold)
        else:
            self.right_foot_drag_start_x = None
        return drag_penalty

    def _get_leg_joint_ids(self):
        if self.leg_joint_ids is not None:
            return self.leg_joint_ids
        
        model = self.env.unwrapped.model
        joint_names = ["left_hip_x","left_hip_y","left_hip_z",
                        "right_hip_x","right_hip_y","right_hip_z"]
        ids={}
        for name in joint_names:
            jid=model.joint(name).id
            ids[name]={
                    "qpos":model.jnt_qposadr[jid],
                    "qvel":model.jnt_dofadr[jid]
                    }
        return ids

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.left_hip_angle_buffer.clear()
        self.right_hip_angle_buffer.clear()
        self.left_leg_vel_buffer.clear()
        self.right_leg_vel_buffer.clear()
        self.foot_diff_buffer.clear()
        self.left_contact_hist.clear()
        self.right_contact_hist.clear()
        self.left_stride_history.clear()
        self.right_stride_history.clear()
        self.stride_event_count = 0
        self.stride_bonus_event_count = 0
        self.last_stride_length = 0.0
        self.last_left_stride = None
        self.last_right_stride = None
        self.last_contact_phase_reward = 0.0
        self.last_swing_forward_reward = 0.0
        self.last_stride_bonus = 0.0
        self.short_stride_event_count = 0
        self.last_short_stride_penalty = 0.0
        self.negative_stride_event_count = 0
        self.last_negative_stride_penalty = 0.0

        self.initial_y = self.env.unwrapped.data.qpos[1]
        self.prev_body_x = self.env.unwrapped.data.qpos[0]  # ⚠️ 新增
        self.prev_body_x_per_frame = self.env.unwrapped.data.qpos[0]
        self.prev_left_hip_angle = 0.0
        self.prev_right_hip_angle = 0.0
        self.prev_diff = None          # 【新增】重置diff追踪
        self.left_contact_count = 0
        self.right_contact_count = 0
        # 脚底拖行惩罚追踪
        self.left_foot_drag_start_x = None
        self.right_foot_drag_start_x = None
        self.total_steps = 0
        self.left_swing_steps = 0
        self.right_swing_steps = 0
        self.left_swing_total_steps = 0
        self.right_swing_total_steps = 0
        self.left_swing_history = []
        self.right_swing_history = []
        self.left_contact_prev = True
        self.right_contact_prev = True
        self.prev_position_state = None
        self.speed = 0.0
        self.curr_position_state = 'centered'

        # 【新增】触地计数器
        self.total_land_count = 0
        self.position_swap_land_count = 0
        self.alternate_reward_count = 0
        self.alternate_penalty_count = 0
        self.landings_since_swap = 0
        self.flight_time_frames = 0

        self.left_swing_hip_total = 0.0
        self.left_swing_time = 0
        self.left_swing_speed_total = 0.0

        self.right_swing_hip_total = 0.0
        self.right_swing_time = 0
        self.right_swing_speed_total = 0.0

        self.left_stance_time = 0
        self.left_stance_speed_total = 0.0

        self.right_stance_time = 0
        self.right_stance_speed_total = 0.0

        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        mujoco_env = self.env.unwrapped
        data = mujoco_env.data
        x_velocity = data.qvel[0]
        body_x = data.qpos[0]
        advance = body_x - self.prev_body_x_per_frame
        self.prev_body_x_per_frame = body_x
        torso_height = data.qpos[2]   # 使用 qpos
        # 获取双脚状态
        left_foot = data.body("left_foot")
        right_foot = data.body("right_foot")
        left_z = left_foot.xpos[2]
        right_z = right_foot.xpos[2]
        left_x = left_foot.xpos[0]
        right_x = right_foot.xpos[0]
        left_contact = (left_z < 0.08)
        right_contact = (right_z < 0.08)
        joint_ids=self._get_leg_joint_ids()
        qvel=data.qvel
        left_hip_angle = data.qpos[joint_ids["left_hip_y"]["qpos"]]
        right_hip_angle = data.qpos[joint_ids["right_hip_y"]["qpos"]]
        left_hip_vel = qvel[joint_ids["left_hip_y"]["qvel"]]
        right_hip_vel = qvel[joint_ids["right_hip_y"]["qvel"]]
        # update angle buffer
        self.left_hip_angle_buffer.append(left_hip_angle)
        self.right_hip_angle_buffer.append(right_hip_angle)
        # velocity activity
        self.left_leg_vel_buffer.append(left_hip_vel)
        self.right_leg_vel_buffer.append(right_hip_vel)

        # 【fix】Humanoid-v4 无 xvel 属性，改用 cvel[3:]（线速度）@ xmat(3,3) 转换到世界坐标
        left_cvel = data.body("left_foot").cvel   # [ang_vel(3), lin_vel(3)]
        right_cvel = data.body("right_foot").cvel
        left_world_vel = left_cvel[3:] @ data.body("left_foot").xmat.reshape(3, 3)
        right_world_vel = right_cvel[3:] @ data.body("right_foot").xmat.reshape(3, 3)
        left_foot_vel = np.sqrt(left_world_vel[0]**2 + left_world_vel[1]**2)
        right_foot_vel = np.sqrt(right_world_vel[0]**2 + right_world_vel[1]**2)

        self.total_steps+=1
        if left_contact:
            self.left_contact_count+=1
        if left_contact and not self.left_contact_prev:
            self.total_land_count += 1
        if right_contact and not self.right_contact_prev:
            self.total_land_count += 1
        if right_contact:
            self.right_contact_count+=1


        # 累计 swing 指标
        if not left_contact and right_contact:
            self.left_swing_time += 1
            self.left_swing_hip_total += abs(left_hip_angle - self.prev_left_hip_angle)
            self.left_swing_speed_total += left_hip_vel

        if right_contact and not left_contact:
            self.right_swing_time += 1
            self.right_swing_hip_total += abs(right_hip_angle - self.prev_right_hip_angle)
            self.right_swing_speed_total += right_hip_vel

        # 累计 stance 指标
        if left_contact and not right_contact:
            self.left_stance_time += 1
            self.left_stance_speed_total += left_hip_vel

        if right_contact and not left_contact:
            self.right_stance_time += 1
            self.right_stance_speed_total += right_hip_vel       

        self.prev_left_hip_angle = left_hip_angle
        self.prev_right_hip_angle = right_hip_angle


        # 速度奖励
        self.speed = advance / 0.015
        target_speed = 1.3
        sigma = 0.3
        forward_reward = self.forward_weight * np.exp(-((self.speed - target_speed) ** 2) / (2 * sigma ** 2))
        # 【对称门控】首次落地前（None）不门控
        if self.last_left_stride is not None and self.last_right_stride is not None:
            self.forward_gate = np.clip(min(self.last_left_stride, self.last_right_stride) / self.forward_gate_ref, 0.0, 1.0)
            forward_reward = forward_reward * self.forward_gate
        else:
            self.forward_gate = 1.0
        # 高速进行惩罚
        speed_penalty = 2.0 * max(0, self.speed - 1.4)
        forward_reward -= speed_penalty

        # 【改】高度奖励：平顶健康带 + 移动门控。旧版帐篷中心 1.2 在自然站高(1.393)之下，
        # 是蹲踞引力（实测躯干被压到 1.21 吃 538 分/集）且帐篷形状打压起伏。
        # 新版：1.25-1.45 平顶（覆盖自然站高，不 fight 周期起伏）；1.05→1.25 线性升（防摔/防躺梯度）；
        # 1.45→1.55 线性降（防跳起）；速度门与 survival_bonus 同款——站立不动拿不到，防止重开站立陷阱
        band = min((torso_height - 1.05) / 0.20, (1.55 - torso_height) / 0.10, 1.0)
        height_reward = self.healthy_weight * max(band, 0.0) \
            * (1.0 if (advance / 0.015) > 0.3 else 0.0)

        # 生存奖励（固定低值）
        survival_bonus = 0.5 if (advance / 0.015) > 0.3 else 0.0  # 【填谷】0.2→0.5、门 0.5→0.3：过渡期速度(0.48-0.55)
        # 骑在旧门槛0.5上反复开关导致地板失效、Q塌、螺旋；0.3 让地板在过渡速度稳定打开。站立(0速)仍拿不到。

        # 控制成本
        ctrl_cost = self.ctrl_cost_weight * np.square(action).sum()

        # 摔倒惩罚
        fall_penalty = self.fall_penalty_weight if terminated else 0.0

        # 步幅奖励：奖励落地时两脚间距
        stride_bonus = 0
        short_stride_penalty = 0
        negative_stride_penalty = 0
        stride_balance_penalty = 0.0
        STRIDE_WEIGHT = 20.0
        STRIDE_MAX = 0.12
        MIN_SPACING = 0.2
        # 【渐变斜坡】min 结算从 0.10 起分、0.32 满分（替代 0.2 硬悬崖）。
        # 地板取当前工作点（min 中位 ~0.087-0.10）：小步幅不白拿（无满足盆地），
        # 梯度从现状正前方开始且更陡（91 分/米），0.10→0.32 全程连续。
        # 验证见 validate_graded_stride.py。
        STRIDE_FLOOR = 0.10
        STRIDE_TOP = 0.32
        BALANCE_MIN_STRIDE = 0.15   # 较弱一侧的步幅下界：两侧都小步 / 居中不交替都会踩到
        BALANCE_CAP = 0.30   # 单次结算封顶：560K 基线 0.233 低于此值所以梯度保留；flail 期旧式 0.425~0.50 被压到 0.30（flail/走路比 从 2.47 降到 1.29）
        SHORT_STRIDE_WEIGHT =7.0
        SHORT_STRIDE_SCALE = 0.06
        NEGATIVE_STRIDE_WEIGHT = 10.0
        NEGATIVE_STRIDE_SCALE = 0.08
        SCALE_ANGLE = 0.3  # hip_y 有效活动范围
        swing_forward_reward = 0.0
        contact_phase_reward = 0.0
        # 【软门控】交替判定提前到步幅结算之前（_check_cycle 有 prev_diff 副作用，每次落地只能调用一次；
        # 原位置的调用改为复用这个结果）。alternate_result: >0非交替, <0交替, 0无信息(首落地/脚间距过小)
        alternate_result = self.compute_alternate_penalty(left_x, right_x, left_contact, right_contact)
        # 【改】位置状态/交换判定提前到步幅结算之前——门控需要知道本次落地是否完成真交换。
        # 函数只读 left_contact_prev、只写自己的状态，与 alternate 的调用先后无副作用
        position_swap_reward, stall_penalty = self.compute_position_swap_reward(
            left_x, right_x, body_x, left_contact, right_contact)
        # 左脚触地，右脚支撑
        if not self.left_contact_prev and left_contact and right_contact:
            left_stride = left_x - right_x
            self.stride_event_count += 1
            self.last_stride_length = left_stride
            if left_stride > MIN_SPACING:
                pass  # 【min结算】stride_bonus 统一在下方按 min(左右步幅) 结算（弱腿门控）
            elif left_stride > 0:
                # 【改】落在前面但步幅不足：不再罚（原墙挡住左脚进双支撑落地），计数保留供监控
                self.short_stride_event_count += 1
                self.last_short_stride_penalty = 0.0
            else:
                negative_stride_penalty = np.clip(-left_stride / NEGATIVE_STRIDE_SCALE, 0, 1) * NEGATIVE_STRIDE_WEIGHT
                self.negative_stride_event_count += 1
                self.last_negative_stride_penalty = negative_stride_penalty

            # 这里直接算 swing_forward_reward
            if self.left_swing_time > 0:
                swing_forward_reward = np.clip(self.left_swing_hip_total / self.left_swing_time / SCALE_ANGLE, 0, 1) * 1.0
            else:
                swing_forward_reward = 0.0

            # 这里直接算 contact_phase_reward
            if self.left_stance_time > 0 and self.right_swing_time > 0:
                stance_q = 1.0 - np.clip(self.left_stance_speed_total / self.left_stance_time / 1.0, 0, 1)
                swing_q = np.clip(self.right_swing_speed_total / max(self.right_swing_time, 1) / 2.0, 0, 1)
                contact_phase_reward = (stance_q + swing_q) / 2 * self.contact_phase_weight
            else:
                contact_phase_reward = 0.0
            self.last_contact_phase_reward = contact_phase_reward
            self.last_swing_forward_reward = swing_forward_reward

            # 结算后清零
            self.left_swing_hip_total = 0.0
            self.left_swing_time = 0
            self.left_swing_speed_total = 0.0
            self.left_stance_time = 0
            self.left_stance_speed_total = 0.0
            self.right_swing_hip_total = 0.0
            self.right_swing_time = 0
            self.right_swing_speed_total = 0.0
        # 右脚触地，左脚支撑
        if not self.right_contact_prev and right_contact and left_contact:
            right_stride = right_x - left_x
            self.stride_event_count += 1
            self.last_stride_length = right_stride
            if right_stride > MIN_SPACING:
                pass  # 【min结算】stride_bonus 统一在下方按 min(左右步幅) 结算（弱腿门控）
            elif right_stride > 0:
                # 【改】落在前面但步幅不足：不再罚，计数保留供监控
                self.short_stride_event_count += 1
                self.last_short_stride_penalty = 0.0
            else:
                negative_stride_penalty = np.clip(-right_stride / NEGATIVE_STRIDE_SCALE, 0, 1) * NEGATIVE_STRIDE_WEIGHT
                self.negative_stride_event_count += 1
                self.last_negative_stride_penalty = negative_stride_penalty

            # 这里直接算 swing_forward_reward
            if self.right_swing_time > 0:
                swing_forward_reward = np.clip(self.right_swing_hip_total / self.right_swing_time / SCALE_ANGLE, 0, 1) * 1.0
            else:
                swing_forward_reward = 0.0
                
            # 这里直接算 contact_phase_reward
            if self.right_stance_time > 0 and self.left_swing_time > 0:
                stance_q = 1.0 - np.clip(self.right_stance_speed_total / self.right_stance_time / 1.0, 0, 1)
                swing_q = np.clip(self.left_swing_speed_total / max(self.left_swing_time, 1) / 2.0, 0, 1)
                contact_phase_reward = (stance_q + swing_q) / 2 * self.contact_phase_weight
            else:
                contact_phase_reward = 0.0
            self.last_contact_phase_reward = contact_phase_reward
            self.last_swing_forward_reward = swing_forward_reward

            self.right_swing_hip_total = 0.0
            self.right_swing_time = 0
            self.right_swing_speed_total = 0.0
            self.left_stance_time = 0
            self.left_stance_speed_total = 0.0
            self.left_swing_hip_total = 0.0
            self.left_swing_time = 0
            self.left_swing_speed_total = 0.0

        # ========== 左右步幅对称惩罚 ==========
        # 在任意落地帧结算，不要求双支撑——实测左脚落地几乎从不落在双支撑期
        # （4 个 seed 里 3 个是 0 次），放上面的 stride 分支里会丢掉绝大部分信号
        any_landing = (not self.left_contact_prev and left_contact) or (not self.right_contact_prev and right_contact)
        if not self.left_contact_prev and left_contact:
            self.last_left_stride = left_x - right_x
        if not self.right_contact_prev and right_contact:
            self.last_right_stride = right_x - left_x
        if any_landing and self.last_left_stride is not None and self.last_right_stride is not None:
            lo = min(self.last_left_stride, self.last_right_stride)
            stride_balance_penalty = min(
                self.stride_balance_weight * (abs(self.last_left_stride - self.last_right_stride) + max(0.0, BALANCE_MIN_STRIDE - lo)),
                BALANCE_CAP)
            # 【min步幅结算·渐变斜坡】bonus 绑到较弱一侧（用户方案 + 斜坡化）：
            # 弱腿部分进步立刻有回报（0.05 起分），不再逼策略跳 0.2 悬崖（跳崖即负步幅源）。
            # 无“缩右腿减亏”漏洞：min 变小两边都归零。保留真交换硬门控。
            if self.last_landing_swapped:
                stride_bonus = np.clip((lo - STRIDE_FLOOR) / (STRIDE_TOP - STRIDE_FLOOR), 0, 1) * STRIDE_WEIGHT
                if stride_bonus > 0:
                    self.stride_bonus_event_count += 1
            self.last_stride_bonus = stride_bonus

        # ========== 新增：触地相位一致性奖励 ==========
        # contact_phase_reward = self.compute_contact_phase_consistency(
        #     left_contact, right_contact, left_foot_vel, right_foot_vel) * self.contact_phase_weight
        
        
        # 身体稳定性：基础奖励 + 姿态惩罚
        # 保留基础奖励 0.3，在此基础上对过度倾斜进行惩罚
        PITCH_THRESH      = 0.2   # 向后仰阈值 0.2 rad ≈ 11°
        PITCH_BACK_W      = 3.0
        PITCH_FWD_W       = 2.0
        ROLL_THRESH       = 0.15   # 左右侧倾阈值 0.15 rad ≈ 8-9°
        ROLL_W            = 5.0

        quat = data.qpos[3:7]
        r = Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]])
        roll,pitch,yaw = r.as_euler("xyz")

        # 基础奖励
        balance_reward = 0.1
        # pitch 惩罚
        pitch_penalty = 0.0
        if pitch < -PITCH_THRESH:
            pitch_penalty = (abs(pitch) - PITCH_THRESH) * PITCH_BACK_W
        elif pitch > PITCH_THRESH:
            pitch_penalty = (pitch - PITCH_THRESH) * PITCH_FWD_W

        # roll 惩罚
        roll_penalty = 0.0
        if abs(roll) > ROLL_THRESH:
            roll_penalty = ((abs(roll) - ROLL_THRESH)) * ROLL_W

        balance_penalty = pitch_penalty + roll_penalty
        net_balance = balance_reward - balance_penalty

        # 左右腿角度相位差
        phase_reward = self.compute_hip_phase_reward()*self.phase_weight


        # 腿部关节不对称活动惩罚
        left_leg_activity = self.compute_leg_activity(
            self.left_hip_angle_buffer,self.left_leg_vel_buffer)
        right_leg_activity = self.compute_leg_activity(
            self.right_hip_angle_buffer,self.right_leg_vel_buffer)
        activity_difference = abs(left_leg_activity-right_leg_activity)
        activity_diff_penalty = (self.activity_diff_penalty_weight*(activity_difference)**2) if (not left_contact) or (not right_contact) else 0


        # 腿部摆动奖励
        if not left_contact:
            self.left_swing_steps+=1
            self.left_swing_total_steps+=1
        else:
            if self.left_swing_steps>0:
                self.left_swing_history.append(self.left_swing_steps)
                self.left_swing_steps=0
        if not right_contact:
            self.right_swing_steps+=1
            self.right_swing_total_steps+=1
        else:
            if self.right_swing_steps>0:
                self.right_swing_history.append(self.right_swing_steps)
                self.right_swing_steps=0


        # 左右支撑时间不对称惩罚
        self.left_contact_hist.append(float(left_contact))
        self.right_contact_hist.append(float(right_contact))
        if len(self.left_contact_hist) >= 20:
            left_support_ratio = np.mean(self.left_contact_hist)
            right_support_ratio = np.mean(self.right_contact_hist)
            support_sym_penalty = abs(left_support_ratio - right_support_ratio)
        else:
            support_sym_penalty = 0.0
        support_sym_penalty *= self.support_sym_weight

        # 【新增】连续支撑不对称惩罚：惩罚一条腿长时间支撑
        # [新] 无交替惩罚（alternate_result 已在步幅结算前算好，勿重复调用——_check_cycle有副作用）
        alternate_penalty = alternate_result * self.alternate_penalty_weight

        # 脚底拖行惩罚
        # foot_drag_penalty = self.compute_foot_drag_penalty(left_contact, right_contact, left_x, right_x) * self.drag_weight

        # 横向躯干偏移惩罚
        lateral_penalty = 0
        lateral_pos = abs(data.qpos[1]-self.initial_y)      # 躯干横向位置
        lateral_vel = abs(data.qvel[1])      # 躯干横向速度
        lateral_penalty = self.lateral_weight*(0.5*lateral_pos + 0.2*lateral_vel)

        # ========== 新增：髋关节侧摆角度惩罚 ==========
        # 惩罚hip_x和hip_z角度的绝对值（腿朝左/右前方摆动）
        hip_xz_angle_penalty = 0.0
        left_hip_x_angle = data.qpos[joint_ids["left_hip_x"]["qpos"]]
        right_hip_x_angle = data.qpos[joint_ids["right_hip_x"]["qpos"]]     
        left_hip_z_angle = data.qpos[joint_ids["left_hip_z"]["qpos"]]
        right_hip_z_angle = data.qpos[joint_ids["right_hip_z"]["qpos"]] 

        HIP_X_ANG_THRESH = 0.15      # 允许±0.15rad（约8.6度）的偏移
        HIP_X_PENALTY_WEIGHT = 1.5
        if abs(left_hip_x_angle) > HIP_X_ANG_THRESH:
            hip_xz_angle_penalty += (abs(left_hip_x_angle) - HIP_X_ANG_THRESH) * HIP_X_PENALTY_WEIGHT
        if abs(right_hip_x_angle) > HIP_X_ANG_THRESH:
            hip_xz_angle_penalty += (abs(right_hip_x_angle) - HIP_X_ANG_THRESH) * HIP_X_PENALTY_WEIGHT

        HIP_Z_ANG_THRESH = 0.10
        HIP_Z_PENALTY_WEIGHT = 0.8
        if abs(left_hip_z_angle) > HIP_Z_ANG_THRESH:
            hip_xz_angle_penalty += (abs(left_hip_z_angle) - HIP_Z_ANG_THRESH) * HIP_Z_PENALTY_WEIGHT
        if abs(right_hip_z_angle) > HIP_Z_ANG_THRESH:
            hip_xz_angle_penalty += (abs(right_hip_z_angle) - HIP_Z_ANG_THRESH) * HIP_Z_PENALTY_WEIGHT


        flight_phase_penalty = self.flight_time_penalty_weight if (not left_contact and not right_contact) else 0.0

        self.left_contact_prev = left_contact
        self.right_contact_prev = right_contact

        reward = (forward_reward + 
                swing_forward_reward +
                stride_bonus +
                net_balance +
                height_reward + 
                fall_penalty +
                phase_reward +  
                survival_bonus +
                position_swap_reward +
                contact_phase_reward +
                - activity_diff_penalty
                - support_sym_penalty
                - alternate_penalty  # 【新增】无交替惩罚
                - lateral_penalty
                - hip_xz_angle_penalty
                - flight_phase_penalty
                # - foot_drag_penalty
                - ctrl_cost
                # - short_stride_penalty   # 【待启用】公式注释保留在左右落地分支
                - negative_stride_penalty
                - stride_balance_penalty)


        info["phase_reward"] = phase_reward
        info["ctrl_cost"] = ctrl_cost
        info["activity_difference"] = activity_difference
        info["support_sym_penalty"] = support_sym_penalty
        info["lateral_penalty"] = lateral_penalty
        info["position_swap_reward"] = position_swap_reward
        info["contact_phase_reward"] = contact_phase_reward
        info["last_contact_phase_reward"] = self.last_contact_phase_reward

        info["stride_bonus"] = stride_bonus
        info["stride_event_count"] = self.stride_event_count
        info["stride_bonus_event_count"] = self.stride_bonus_event_count
        info["stride_length"] = self.last_stride_length
        info["last_stride_bonus"] = self.last_stride_bonus
        info["stride_bonus_rate"] = self.stride_bonus_event_count / max(self.stride_event_count, 1)

        info["short_stride_penalty"] = short_stride_penalty
        info["last_short_stride_penalty"] = self.last_short_stride_penalty
        info["short_stride_event_rate"] = self.short_stride_event_count / max(self.stride_event_count, 1)

        info["negative_stride_penalty"] = negative_stride_penalty
        info["last_negative_stride_penalty"] = self.last_negative_stride_penalty
        info["negative_stride_event_count"] = self.negative_stride_event_count
        info["negative_stride_rate"] = self.negative_stride_event_count / max(self.stride_event_count, 1)

        info["stride_balance_penalty"] = stride_balance_penalty
        info["last_left_stride"] = self.last_left_stride if self.last_left_stride is not None else 0.0
        info["last_right_stride"] = self.last_right_stride if self.last_right_stride is not None else 0.0

        info["hip_xz_angle_penalty"] = hip_xz_angle_penalty
        info["alternate_penalty"] = alternate_penalty  # 【新增】无交替惩罚
        # info["foot_drag_penalty"] = foot_drag_penalty
        info["forward_reward"] = forward_reward
        info["forward_gate"] = self.forward_gate
        info["height_reward"] = height_reward
        info["net_balance"] = net_balance
        info["pitch_angle"] = pitch
        info["roll_angle"] = roll
        info["swing_forward_reward"] = swing_forward_reward
        info["last_swing_forward_reward"] = self.last_swing_forward_reward
        info["body_advance"] = self.body_advance
        info["speed"] = self.speed
        info["position_swap_rate"] = self.position_swap_land_count / self.total_land_count if self.total_land_count > 0 else 0
        info["alternate_reward_rate"] = self.alternate_reward_count / self.total_land_count if self.total_land_count > 0 else 0
        info["alternate_penalty_rate"] = self.alternate_penalty_count / self.total_land_count if self.total_land_count > 0 else 0
        info["stall_penalty"] = stall_penalty
        info["stall_max"] = self.landings_since_swap
        info["flight_phase_penalty"] = flight_phase_penalty
        

        if terminated or truncated:
            mean_left_swing = np.mean(self.left_swing_history) if self.left_swing_history else 0
            mean_right_swing = np.mean(self.right_swing_history) if self.right_swing_history else 0

            info["mean_left_swing"] = mean_left_swing
            info["mean_right_swing"] = mean_right_swing
            

        return obs, reward, terminated, truncated, info


# ======================================================================
# 镜像增广（Mirror Augmentation）
# 原理：环境经 patch_mirror_symmetry 修补后严格左右对称（动力学等变性 4e-12，
# 见 mirror_map.py 与 debug_probes/build_mirror_map.py）。任何一集
# (初始态, 动作序列) 在"镜像初始态 + 镜像动作序列"下整集重放，
# 奖励逐项相等（步态时钟平移半周期使稠密对齐目标同步翻转）。
# 镜像集注入 replay buffer，等价于把右腿主导经验免费翻转为左腿主导经验。
# 数据流：worker 进程内 EpisodeRecorder 录制整集 → done 时 MirrorReplayer
# 在 worker 专用环境中重放 → 经 info["mirror_episode"] 传回主进程 →
# MirrorAugCallback 用 MirrorInjector 按 n_envs 槽位组批量写入 buffer。
# ======================================================================

class MirrorReplayer:
    """在 worker 专用环境中整集镜像重放。产出 (L,·) 数组批，动作已转 buffer 域 [-1,1]。"""

    def __init__(self, env, mirror_map):
        self.env = env                 # WalkRewardWrapper（已修补，无 recorder）
        self.mm = mirror_map
        self.scale = float(env.action_space.high[0])   # env域(±0.4)→buffer域(±1)

    def replay(self, q0, v0, clock, actions, return_infos=False):
        w, mj = self.env, self.env.unwrapped
        w.reset()                      # wrapper 记忆清零（total_steps=0、各 buffer 清空）
        q_m, v_m = self.mm.mirror_qpos_qvel(q0, v0)
        mj.set_state(q_m, v_m)         # 覆盖物理态（内部含 mj_forward）
        # wrapper 记忆与镜像初始态对齐：initial_y 取镜像值 ⇒ lateral 项逐项等价
        w.initial_y = float(mj.data.qpos[1])
        w.prev_body_x = w.prev_body_x_per_frame = float(mj.data.qpos[0])
        # 步态时钟平移半周期：镜像集左右角色互换，稠密对齐目标同步翻转
        w.total_steps = int(clock) + w.gait_period // 2
        obs = mj._get_obs().astype(np.float32)
        obs_l, nxt_l, act_l, rew_l, done_l, to_l, info_l = [], [], [], [], [], [], []
        for a_env in actions:
            a_m = self.mm.mirror_action(a_env)
            obs2, r, term, trunc, info = self.env.step(a_m)
            obs_l.append(obs)
            nxt_l.append(obs2.astype(np.float32))
            act_l.append((a_m / self.scale).astype(np.float32))
            rew_l.append(np.float32(r))
            done_l.append(np.float32(term or trunc))
            to_l.append(bool(info.get("TimeLimit.truncated", False)))
            if return_infos:
                info_l.append(info)
            obs = obs2
            if term or trunc:
                break
        out = dict(obs=np.asarray(obs_l), next_obs=np.asarray(nxt_l),
                   action=np.asarray(act_l), reward=np.asarray(rew_l),
                   done=np.asarray(done_l), timeout=np.asarray(to_l))
        if return_infos:
            out["infos"] = info_l
        return out


class EpisodeRecorder(gym.Wrapper):
    """录制每集初始物理态+时钟+动作序列；done 时整集镜像重放并挂到 info（worker 进程内）。"""

    def __init__(self, env, replayer):
        super().__init__(env)
        self.replayer = replayer
        self._ep = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        mj = self.env.unwrapped
        w = self.env                                   # WalkRewardWrapper
        self._ep = (mj.data.qpos.copy(), mj.data.qvel.copy(), w.total_steps, [])
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if self._ep is not None:
            self._ep[3].append(np.asarray(action, dtype=np.float64))
        if self._ep is not None and (term or trunc):
            q0, v0, clock, acts = self._ep
            self._ep = None
            try:
                info["mirror_episode"] = self.replayer.replay(q0, v0, clock, acts)
            except Exception as _e:                    # 镜像重放失败不得打断训练
                info["mirror_episode"] = None
                print(f"[mirror] 整集重放失败(忽略): {_e}")
        return obs, r, term, trunc, info


class MirrorInjector:
    """镜像整集 → replay buffer 搬运（主进程）。buffer.add 固定按 n_envs 槽位写，
    因此把 pending 队列按 n_envs 条一组批量 add（组内混合不同集的相邻转移，互不影响）。"""

    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.pending = []
        self.episodes = 0
        self.transitions = 0

    def drain_infos(self, infos, buffer):
        for info in infos:
            ep = info.pop("mirror_episode", None) if isinstance(info, dict) else None
            if ep is None:
                continue
            self.episodes += 1
            for i in range(ep["obs"].shape[0]):
                self.pending.append((ep["obs"][i:i + 1], ep["next_obs"][i:i + 1],
                                     ep["action"][i:i + 1], ep["reward"][i:i + 1],
                                     ep["done"][i:i + 1],
                                     {"TimeLimit.truncated": bool(ep["timeout"][i])}))
        self.flush(buffer)

    def flush(self, buffer):
        n = self.n_envs
        while len(self.pending) >= n:
            chunk = self.pending[:n]
            del self.pending[:n]
            obs, nxt, act, rew, done, infos = zip(*chunk)
            buffer.add(np.concatenate(obs), np.concatenate(nxt), np.concatenate(act),
                       np.concatenate(rew), np.concatenate(done), list(infos))
            self.transitions += n


class MirrorAugCallback(BaseCallback):
    """从 collect_rollouts 的 infos 里取 worker 产出的镜像整集并注入 buffer。"""

    def __init__(self, injector, verbose=0):
        super().__init__(verbose)
        self.injector = injector

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            self.injector.drain_infos(infos, self.model.replay_buffer)
        return True


class MirrorEqSAC(SAC):
    """方案A v2：数据锚定等变正则（归一化混合）。
    等变监督 ||π(mirror_obs(s)) − mirror_action(a_real)||² 与 RL 项按归一化比例混合：
        actor_loss = (1−λ)·RL + λ·eq_norm,  eq_norm = eq_MSE / detach(mean(||mirror(a)||²))
    依据：v1（W=1.0 加性）实测行为缺口不动——actor loss 量级 ~35 而正则项仅 ~0.2-0.5，
    梯度比约 1:70 被 Q 梯度淹没（eq_reg_loss 下降是共适应假象）；v2 归一化后与
    动作幅度无关，λ 直接控制约束力，干跑验证缺口收敛后再定默认值。
    SB3 包本体零修改；保存时排除镜像/正则相关属性，SAC.load 可直接读。"""

    def __init__(self, *args, mirror_map=None, eq_mix_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mirror_map = mirror_map
        self.eq_mix_lambda = float(eq_mix_lambda)
        if mirror_map is not None and self.eq_mix_lambda > 0:
            dev = self.device
            self._obs_src = th.as_tensor(mirror_map.obs_src, dtype=th.long, device=dev)
            self._obs_sign = th.as_tensor(mirror_map.obs_sign, dtype=th.float32, device=dev)
            self._act_src = th.as_tensor(mirror_map.act_src, dtype=th.long, device=dev)
            self._act_sign = th.as_tensor(mirror_map.act_sign, dtype=th.float32, device=dev)
        else:
            self._obs_src = None

    def _excluded_save_params(self) -> list[str]:
        """SB3 2.9 中该方法是实例方法（base_class.py:303），勿加 classmethod"""
        return super()._excluded_save_params() + ["mirror_map", "eq_mix_lambda",
                                                  "_obs_src", "_obs_sign", "_act_src", "_act_sign"]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, eq_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # ===== 等变正则（v1/v2/v3 已证伪，本轮 λ=0 整体关闭）=====
            if self._obs_src is not None:
                mean_s, _, _ = self.actor.get_action_dist_params(replay_data.observations)
                eq_target = (th.tanh(mean_s)[:, self._act_src] * self._act_sign).detach()
                eq_obs = replay_data.observations[:, self._obs_src] * self._obs_sign
                mean_ms, _, _ = self.actor.get_action_dist_params(eq_obs)
                eq_pi = th.tanh(mean_ms)
                eq_mse = F.mse_loss(eq_pi, eq_target)
                eq_losses.append(eq_mse.item())
                eq_norm = eq_mse / (th.mean(th.sum(eq_target ** 2, dim=1)).detach() + 1e-6)
                actor_loss = (1.0 - self.eq_mix_lambda) * actor_loss + self.eq_mix_lambda * eq_norm

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(eq_losses) > 0:
            self.logger.record("train/eq_reg_loss", np.mean(eq_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


def make_env(env_id, rank, seed=0, patch=False, with_mirror=False):
    def _init():
        env = gym.make(env_id, max_episode_steps=1500)
        if patch:
            patch_mirror_symmetry(env.unwrapped.model)   # 左右对称修补（左膝弹簧/左髋转子）
        env = WalkRewardWrapper(env)
        if with_mirror:
            # worker 专用镜像重放环境（同修补，无 recorder）
            r_env = gym.make(env_id, max_episode_steps=1500)
            patch_mirror_symmetry(r_env.unwrapped.model)
            r_env = WalkRewardWrapper(r_env)
            env = EpisodeRecorder(env, MirrorReplayer(r_env, MirrorMap(r_env.unwrapped.model)))
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == "__main__":
    env_id = "Humanoid-v4"
    n_train_envs = 8    # 训练环境数量
    n_eval_envs = 1     # 评估环境数量（通常 1 个即可）
    total_timesteps = 2_000_000
    seed = 42
    # 镜像增广总开关：环境左右对称修补 + 整集镜像重放注入（训练/评估环境均修补，仅训练环境录制）
    ENABLE_MIRROR_AUG = True
    # 方案A 等变正则（v1/v2/v3 已被干跑+2M 实测证伪：软损失装不进 MLP 函数等变）。
    # 本轮唯一变量 = min(左右步幅) stride_bonus，故置 0 关闭；代码保留备查。
    EQ_MIX_LAMBDA = 0.0

    # ---- 训练环境 ----
    train_vec_env = SubprocVecEnv(
        [make_env(env_id, i, seed, patch=ENABLE_MIRROR_AUG, with_mirror=ENABLE_MIRROR_AUG)
         for i in range(n_train_envs)]
    )
    # 如果需要，可以包装 VecMonitor，同时用于训练日志
    train_vec_env = VecMonitor(train_vec_env)

    # ---- 评估环境（独立；同步修补保持与训练同一任务定义，但不做镜像录制） ----
    eval_vec_env = SubprocVecEnv(
        [make_env(env_id, n_train_envs + i, seed, patch=ENABLE_MIRROR_AUG)
         for i in range(n_eval_envs)]
    )
    # 评估环境也建议用 VecMonitor 包装，方便 EvalCallback 记录
    eval_vec_env = VecMonitor(eval_vec_env)

    # 策略网络结构（与PPO相同，SAC也支持）
    policy_kwargs = {
        'net_arch': [512, 256],
        'activation_fn': nn.Tanh
    }

    # ============ SAC 模型（方案A：等变性正则子类） ============
    # 主进程本地建一份 MirrorMap（仅取关节轴/索引表，与 worker 内的表同源）
    _mm_env = gym.make(env_id, max_episode_steps=1500)
    if ENABLE_MIRROR_AUG:
        patch_mirror_symmetry(_mm_env.unwrapped.model)
    mirror_map_main = MirrorMap(_mm_env.unwrapped.model)
    _mm_env.close()

    model = MirrorEqSAC(                                    
        'MlpPolicy',
        train_vec_env,
        learning_rate=lambda prog: 2e-4 * prog + 1e-5 * (1 - prog),  # 线性退火：2e-4 -> 1e-5  
        # learning_rate=1e-5,
        buffer_size=1_500_000,
        batch_size=1024,
        tau=0.01,
        gamma=0.99,
        train_freq=512,
        gradient_steps=512,
        ent_coef=0.05,            # 熵系数。回退到 SAC_17_tmp 的 proven 值：0.1 配 lr=1e-5 时 ent/lr=10000（健康组 333），熵项压过奖励信号把策略推成均匀分布
        target_entropy=-8.5,      # 动作维度17
        policy_kwargs=policy_kwargs,
        seed=seed,                  # 固定种子：同一 seed 结果可复现
        verbose=1,
        tensorboard_log="./sac_humanoid_walk/",
        mirror_map=mirror_map_main if ENABLE_MIRROR_AUG else None,
        eq_mix_lambda=EQ_MIX_LAMBDA if ENABLE_MIRROR_AUG else 0.0,
    )

    # ---------- 加载之前保存的模型（warm-start续训）----------
    # warm-start历史上四轮崩溃的根因：learn()开始时buffer里要么没数据(learning_starts默认100，
    # 首批更新1024×重复采样)，要么是均匀随机动作的数据(learning_starts调大后SB3预热用
    # action_space.sample()采样，分布外垃圾)。解法见下方预填充段。
    old_model = SAC.load("./best_model/best_model.zip", env=train_vec_env)
    model.policy.load_state_dict(old_model.policy.state_dict())
    # old_model 只用来取 policy 权重，但 SAC.load 会连同它自己的 9.14GB replay_buffer 一起构造；
    # 不删的话这个 buffer 活到训练结束，和 model 的叠加成 18GB（机器只有 15GB，可用 2GB）
    del old_model
    gc.collect()
    # # model.critic_target.load_state_dict(old_model.policy.state_dict(),strict=False)
    model.replay_buffer.reset()
    # model = SAC.load("./best_model/best_model.zip", env=train_vec_env)

    # ---------- buffer预填充：用加载的策略先收集5万步真实数据，再开始正式训练 ----------
    # 与SB3的learning_starts预热本质不同：这里动作由策略发出(predict)，不是随机采样，
    # critic的首批梯度更新就有足量、分布正确的数据。约3分钟纯收集，无梯度更新，不占磁盘。
    # 调用形式严格复刻 off_policy_algorithm.collect_rollouts 的 buffer.add 路径：
    #   每vec-step批量add一次、动作经scale_action缩放到[-1,1]、done时next_obs用terminal_observation替换。
    n_prefill = 200_000   # 方案A：策略已会走（1792K warm-start），预填充缩至 200K 刷新 buffer
    mirror_injector = MirrorInjector(n_envs=n_train_envs)
    _t0 = _time.time()
    obs = train_vec_env.reset()
    for _i in range(n_prefill // n_train_envs):
        action, _ = model.predict(obs, deterministic=False)      # env空间动作(±0.4)
        buffer_action = model.policy.scale_action(action)         # buffer存[-1,1]缩放版
        new_obs, rewards, dones, infos = train_vec_env.step(action)
        next_obs = np.array(new_obs).copy()
        for _j in range(n_train_envs):                            # VecEnv自动reset，真实终止观测在info里
            if dones[_j] and infos[_j].get("terminal_observation") is not None:
                next_obs[_j] = infos[_j]["terminal_observation"]
        model.replay_buffer.add(obs, next_obs, buffer_action, rewards, dones, infos)
        if ENABLE_MIRROR_AUG:
            mirror_injector.drain_infos(infos, model.replay_buffer)   # 预填充期的镜像整集同样入库
        obs = new_obs
        if (_i + 1) % 1000 == 0:
            print(f"预填充进度: {(_i + 1) * n_train_envs}/{n_prefill} 步, 已用{_time.time()-_t0:.0f}秒")
    print(f"✅ buffer预填充完成: {model.replay_buffer.size() * n_train_envs} 条策略数据, 耗时{_time.time()-_t0:.0f}秒, 开始正式训练")
    if ENABLE_MIRROR_AUG:
        print(f"✅ 镜像增广: 预填充期已注入 {mirror_injector.episodes} 个镜像整集 / "
              f"{mirror_injector.transitions} 条转移（约为真实数据的 1 倍）")

    leg_callback = LegActivityCallback()
    # 回调：达到阈值时停止训练
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=100000,
        verbose=1
    )
    extra_log_callback = SaveExtraParamsOnBest(
        save_path="./best_model/",  # 建议与 best_model_save_path 一致
        leg_callback=leg_callback,   # 传入
        verbose=1
    )
    save_best_callback = SaveBestModelOnNewBest(
        save_dir="./best_model/best_model_snapshots/",
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_vec_env,
        callback_on_new_best=CallbackList([stop_callback,extra_log_callback,save_best_callback]),
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=2000,
        deterministic=True,
        n_eval_episodes=5,    # 旧奖励下单集 std 257 才用 10；新奖励下稳定步态单集 std ~8，5 集够用（eval 开销减半）
        render=False
    )

    # buffer已预填充50万条策略数据，learning_starts设0：learn()从第一步起就用策略动作收集，
    # 首批梯度更新直接在预填充数据上进行。勿再调大——SB3预热期用的是随机动作，会污染buffer。
    model.learning_starts = 0

    model.learn(
        total_timesteps=total_timesteps,
        # reset_num_timesteps=False,          # 关键：保持总步数计数，不从头算
        callback=[leg_callback, eval_callback, MirrorAugCallback(mirror_injector)]
    )
    model.save("sac_humanoid_walk_final")

    print("训练完成！模型已保存。")

