import os,json
import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import Wrapper
from collections import deque
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC,PPO  # 改为 SAC
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold,BaseCallback,CallbackList


class LegActivityCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_gait_data = {}

    def _on_step(self):
        infos = self.locals["infos"]
        for info in infos:
            self.logger.record("gait/activity_difference", info.get("activity_difference", 0.0))
            self.logger.record("gait/phase_reward", info.get("phase_reward", 0.0))
            self.logger.record("gait/stance_swing_reward", info.get("stance_swing_reward", 0.0))
            self.logger.record("gait/alternation_reward", info.get("alternation_reward", 0.0))
            self.logger.record("gait/alternation_raw", info.get("alternation_raw", 0.0))
            self.logger.record("gait/alternation_success", info.get("alternation_success", 0))
            self.logger.record("gait/alternate_flip_penalty", info.get("alternate_flip_penalty", 0.0))
            self.logger.record("gait/alternate_flip_product", info.get("alternate_flip_product", 0.0))
            self.logger.record("gait/alternate_flip_count", info.get("alternate_flip_count", 0))
            self.logger.record("gait/alternate_no_flip_count", info.get("alternate_no_flip_count", 0))
            self.logger.record("gait/step_reward", info.get("step_reward", 0.0))
            self.logger.record("gait/step_length", info.get("step_length", 0.0))
            self.logger.record("gait/flight_penalty", info.get("flight_penalty", 0.0))
            self.logger.record("gait/double_support_penalty", info.get("double_support_penalty", 0.0))
            self.logger.record("gait/hip_x_angle_penalty", info.get("hip_x_angle_penalty", 0.0))
            self.logger.record("gait/lateral_penalty", info.get("lateral_penalty", 0.0))
            self.logger.record("gait/ctrl_cost", info.get("ctrl_cost", 0.0))
            self.logger.record("gait/forward_reward", info.get("forward_reward", 0.0))
            self.logger.record("gait/height_reward", info.get("height_reward", 0.0))
            self.logger.record("gait/net_balance", info.get("net_balance", 0.0))
            self.logger.record("gait/pitch_angle", info.get("pitch_angle", 0.0))
            self.logger.record("gait/roll_angle", info.get("roll_angle", 0.0))
            self.logger.record("gait/speed", info.get("speed", 0.0))
            self.logger.record("gait/foot_diff", info.get("foot_diff", 0.0))
            self.logger.record("gait/body_advance", info.get("body_advance", 0.0))
            self.logger.record("gait/alternation_success_rate", info.get("alternation_success_rate", 0.0))
            self.logger.record("gait/flight_ratio", info.get("flight_ratio", 0.0))
            self.logger.record("gait/double_support_ratio", info.get("double_support_ratio", 0.0))
            self.logger.record("gait/single_support_ratio", info.get("single_support_ratio", 0.0))
            self.logger.record("gait/valid_left_touchdowns", info.get("valid_left_touchdowns", 0))
            self.logger.record("gait/valid_right_touchdowns", info.get("valid_right_touchdowns", 0))

            if "mean_left_swing" in info:
                self.logger.record("gait/mean_left_swing", info["mean_left_swing"])
                self.logger.record("gait/mean_right_swing", info["mean_right_swing"])
                self.logger.record("gait/mean_step_length", info.get("mean_step_length", 0.0))
                self.logger.record("gait/mean_left_step_length", info.get("mean_left_step_length", 0.0))
                self.logger.record("gait/mean_right_step_length", info.get("mean_right_step_length", 0.0))

            gait_data = {
                "activity_difference": info.get("activity_difference", 0.0),
                "phase_reward": info.get("phase_reward", 0.0),
                "stance_swing_reward": info.get("stance_swing_reward", 0.0),
                "alternation_reward": info.get("alternation_reward", 0.0),
                "alternation_raw": info.get("alternation_raw", 0.0),
                "alternation_success": info.get("alternation_success", 0),
                "alternate_flip_penalty": info.get("alternate_flip_penalty", 0.0),
                "alternate_flip_product": info.get("alternate_flip_product", 0.0),
                "alternate_flip_count": info.get("alternate_flip_count", 0),
                "alternate_no_flip_count": info.get("alternate_no_flip_count", 0),
                "step_reward": info.get("step_reward", 0.0),
                "step_length": info.get("step_length", 0.0),
                "flight_penalty": info.get("flight_penalty", 0.0),
                "double_support_penalty": info.get("double_support_penalty", 0.0),
                "hip_x_angle_penalty": info.get("hip_x_angle_penalty", 0.0),
                "lateral_penalty": info.get("lateral_penalty", 0.0),
                "ctrl_cost": info.get("ctrl_cost", 0.0),
                "forward_reward": info.get("forward_reward", 0.0),
                "height_reward": info.get("height_reward", 0.0),
                "net_balance": info.get("net_balance", 0.0),
                "pitch_angle": info.get("pitch_angle", 0.0),
                "roll_angle": info.get("roll_angle", 0.0),
                "speed": info.get("speed", 0.0),
                "foot_diff": info.get("foot_diff", 0.0),
                "body_advance": info.get("body_advance", 0.0),
                "alternation_success_rate": info.get("alternation_success_rate", 0.0),
                "flight_ratio": info.get("flight_ratio", 0.0),
                "double_support_ratio": info.get("double_support_ratio", 0.0),
                "single_support_ratio": info.get("single_support_ratio", 0.0),
                "valid_left_touchdowns": info.get("valid_left_touchdowns", 0),
                "valid_right_touchdowns": info.get("valid_right_touchdowns", 0),
            }

            if "mean_left_swing" in info:
                gait_data.update({
                    "mean_left_swing": info["mean_left_swing"],
                    "mean_right_swing": info["mean_right_swing"],
                    "mean_step_length": info.get("mean_step_length", 0.0),
                    "mean_left_step_length": info.get("mean_left_step_length", 0.0),
                    "mean_right_step_length": info.get("mean_right_step_length", 0.0),
                })

            self.last_gait_data.update(gait_data)
        return True


class SaveExtraParamsOnBest(BaseCallback):
    def __init__(self, save_path, leg_callback, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.leg_callback = leg_callback
        os.makedirs(self.save_path, exist_ok=True)
        self.log_file = os.path.join(self.save_path, "best_extra_logs.jsonl")

    def _on_step(self) -> bool:
        if hasattr(self.parent, 'best_mean_reward'):
            eval_mean_reward = getattr(self.parent, 'last_mean_reward', None)
            if hasattr(self.parent, 'evaluations_length') and self.parent.evaluations_length:
                eval_mean_ep_length = float(np.mean(self.parent.evaluations_length[-1]))
            else:
                eval_mean_ep_length = None

            gait_data = getattr(self.leg_callback, 'last_gait_data', {})

            def value(key):
                return gait_data.get(key)

            extra_data = {
                "timestep": int(self.num_timesteps),
                "eval/mean_ep_length": eval_mean_ep_length,
                "eval/mean_reward": eval_mean_reward,
                "gait/activity_difference": value("activity_difference"),
                "gait/phase_reward": value("phase_reward"),
                "gait/stance_swing_reward": value("stance_swing_reward"),
                "gait/alternation_reward": value("alternation_reward"),
                "gait/alternation_raw": value("alternation_raw"),
                "gait/alternation_success": value("alternation_success"),
                "gait/alternate_flip_penalty": value("alternate_flip_penalty"),
                "gait/alternate_flip_product": value("alternate_flip_product"),
                "gait/alternate_flip_count": value("alternate_flip_count"),
                "gait/alternate_no_flip_count": value("alternate_no_flip_count"),
                "gait/step_reward": value("step_reward"),
                "gait/step_length": value("step_length"),
                "gait/flight_penalty": value("flight_penalty"),
                "gait/double_support_penalty": value("double_support_penalty"),
                "gait/hip_x_angle_penalty": value("hip_x_angle_penalty"),
                "gait/lateral_penalty": value("lateral_penalty"),
                "gait/ctrl_cost": value("ctrl_cost"),
                "gait/forward_reward": value("forward_reward"),
                "gait/height_reward": value("height_reward"),
                "gait/net_balance": value("net_balance"),
                "gait/pitch_angle": value("pitch_angle"),
                "gait/roll_angle": value("roll_angle"),
                "gait/speed": value("speed"),
                "gait/foot_diff": value("foot_diff"),
                "gait/body_advance": value("body_advance"),
                "gait/alternation_success_rate": value("alternation_success_rate"),
                "gait/flight_ratio": value("flight_ratio"),
                "gait/double_support_ratio": value("double_support_ratio"),
                "gait/single_support_ratio": value("single_support_ratio"),
                "gait/valid_left_touchdowns": value("valid_left_touchdowns"),
                "gait/valid_right_touchdowns": value("valid_right_touchdowns"),
                "gait/mean_left_swing": value("mean_left_swing"),
                "gait/mean_right_swing": value("mean_right_swing"),
                "gait/mean_step_length": value("mean_step_length"),
                "gait/mean_left_step_length": value("mean_left_step_length"),
                "gait/mean_right_step_length": value("mean_right_step_length"),
            }

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
    Humanoid-v4 Walking Reward V2

    重新设计Humanoid的奖励函数,强调前进速度,降低存活奖励。
    解决“原地站立”局部最优问题

    目标：
    1. 稳定向前行走，而不是原地站立
    2. 左右脚严格交替 touchdown：L -> R -> L -> R
    3. 以单脚支撑为主，允许短暂双脚支撑
    4. 双脚同时离地强惩罚，抑制 hopping / running
    5. 步长接近目标值，而不是越大越好
    6. 使用 STANCE -> SWING -> TOUCHDOWN 状态机识别有效迈步
    7. hip phase 仅作为很弱的辅助信号
    """

    def __init__(self, env):
        super().__init__(env)

        # 基础奖励
        self.forward_weight = 2.0
        self.target_speed = 1.2
        self.speed_sigma = 0.30
        self.high_speed_threshold = 1.5
        self.high_speed_penalty_weight = 0.8
        self.healthy_weight = 0.6
        self.ctrl_cost_weight = 0.005
        self.fall_penalty_weight = -50.0

        # gait
        self.alternation_weight = 1.5
        self.stance_weight = 0.5
        self.target_step_length = 0.30
        self.step_length_sigma = 0.10
        self.step_length_weight = 1.0
        self.alternation_margin = 0.08
        self.alternation_advance_threshold = 0.10
        self.cycle_scale = 0.25
        self.cycle_weight = 3.0

        # touchdown detector
        self.min_swing_frames = 8
        self.touchdown_vz_threshold = 0.20

        # double support
        self.double_support_max_frames = 6
        self.double_support_penalty_weight = 0.20

        # flight / hopping
        self.flight_penalty_base = 2.0
        self.flight_penalty_increment = 0.3
        self.flight_penalty_max = 3.5

        # weak phase auxiliary reward
        self.phase_weight = 0.05

        # lateral / hip_x
        self.lateral_weight = 0.10
        self.hip_x_angle_threshold = 0.30
        self.hip_x_penalty_weight = 1.5

        # 基础状态
        self.initial_y = None
        self.prev_body_x = None
        self.prev_body_x_per_frame = None
        self.prev_com_x = None
        self.speed = 0.0
        self.body_advance = 0.0

        # 每条腿的状态：stance / swing
        self.left_leg_state = "stance"
        self.right_leg_state = "stance"
        self.left_swing_frames = 0
        self.right_swing_frames = 0

        # 脚垂直速度历史
        self.prev_left_foot_vz = 0.0
        self.prev_right_foot_vz = 0.0

        # gait phase
        self.expected_touchdown = None
        self.last_touchdown_foot = None

        # 上一次 touchdown 的信息
        self.last_touchdown_foot_x = None
        self.last_completed_alternation_x = None
        self.last_alternation_passed = False

        # 相邻 touchdown 的前后位置翻转信号
        self.prev_diff_for_cycle = None
        self.last_alternate_flip_penalty = 0.0
        self.alternate_flip_product = 0.0

        # 统计
        self.total_frames = 0
        self.total_land_count = 0
        self.alternation_success_count = 0
        self.alternation_attempt_count = 0
        self.alternate_flip_count = 0
        self.alternate_no_flip_count = 0
        self.valid_left_touchdowns = 0
        self.valid_right_touchdowns = 0
        self.invalid_touchdowns = 0

        self.flight_frames = 0
        self.flight_frames_total = 0
        self.double_support_frames = 0
        self.double_support_frames_total = 0
        self.single_support_frames_total = 0

        # step
        self.step_lengths = deque(maxlen=50)
        self.left_step_lengths = deque(maxlen=25)
        self.right_step_lengths = deque(maxlen=25)

        # swing history
        self.left_swing_history = []
        self.right_swing_history = []
        self.last_left_swing_duration = 0
        self.last_right_swing_duration = 0

        # phase/activity buffer
        self.leg_activity_buffer_size = 50
        self.left_hip_angle_buffer = deque(maxlen=50)
        self.left_knee_angle_buffer = deque(maxlen=50)
        self.right_hip_angle_buffer = deque(maxlen=50)
        self.right_knee_angle_buffer = deque(maxlen=50)
        self.left_leg_vel_buffer = deque(maxlen=50)
        self.right_leg_vel_buffer = deque(maxlen=50)

        # MuJoCo ID cache
        self.leg_joint_ids = None
        self.left_foot_geom_id = None
        self.right_foot_geom_id = None
        self.floor_geom_id = None

        # debug
        self.foot_diff = 0.0
        self.last_step_length = 0.0
        self.last_touchdown_valid = False

    def _get_leg_joint_ids(self):
        if self.leg_joint_ids is not None:
            return self.leg_joint_ids

        model = self.env.unwrapped.model
        joint_names = ["left_hip_x", "left_hip_y", "left_knee","right_hip_x", "right_hip_y", "right_knee"]

        self.leg_joint_ids = {}
        for name in joint_names:
            jid = model.joint(name).id
            self.leg_joint_ids[name] = {
                "qpos": model.jnt_qposadr[jid],
                "qvel": model.jnt_dofadr[jid]}

        return self.leg_joint_ids

    def _get_contact_geom_ids(self):
        if self.left_foot_geom_id is None:
            model = self.env.unwrapped.model
            self.left_foot_geom_id = model.geom("left_foot").id
            self.right_foot_geom_id = model.geom("right_foot").id
            self.floor_geom_id = model.geom("floor").id

        return self.left_foot_geom_id, self.right_foot_geom_id, self.floor_geom_id

    def _foot_floor_contact(self, foot_geom_id, floor_geom_id):
        data = self.env.unwrapped.data

        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            if ((g1 == foot_geom_id and g2 == floor_geom_id) or(g2 == foot_geom_id and g1 == floor_geom_id)):
                return contact.dist <= 1e-3

        return False

    def _get_foot_contacts(self):
        left_geom, right_geom, floor_geom = self._get_contact_geom_ids()
        return (
            self._foot_floor_contact(left_geom, floor_geom),
            self._foot_floor_contact(right_geom, floor_geom))

    def _get_com_x(self):
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model
        total_mass = np.sum(model.body_mass)

        if total_mass <= 0:
            return float(data.qpos[0])

        return float(np.sum(model.body_mass * data.xipos[:, 0]) / total_mass)

    def _get_foot_velocity(self, body):
        world_vel = body.cvel[3:] @ body.xmat.reshape(3, 3)
        return world_vel, float(np.linalg.norm(world_vel[:2]))

    def compute_hip_phase_reward(self):
        if len(self.left_hip_angle_buffer) < self.leg_activity_buffer_size:
            return 0.0

        left = np.asarray(self.left_hip_angle_buffer, dtype=np.float64)
        right = np.asarray(self.right_hip_angle_buffer, dtype=np.float64)

        left -= left.mean()
        right -= right.mean()

        left_std = np.std(left)
        right_std = np.std(right)

        if left_std < 1e-4 or right_std < 1e-4:
            return 0.0

        corr = np.corrcoef(left, right)[0, 1]
        if not np.isfinite(corr):
            return 0.0

        anti_phase = (1.0 - corr) / 2.0
        activity = (left_std + right_std) / 2.0
        activity = np.clip(activity / 0.5, 0.0, 1.0)

        return float(anti_phase * activity)

    def compute_leg_activity(self, hip_buffer, knee_buffer, vel_buffer):
        if len(hip_buffer) < 40:
            return 0.0

        hip_rom = max(hip_buffer) - min(hip_buffer)
        knee_rom = max(knee_buffer) - min(knee_buffer)
        vel_activity = np.mean(np.abs(np.asarray(vel_buffer, dtype=np.float64)))

        rom_activity = np.clip((hip_rom + knee_rom) / 3.0, 0.0, 1.0)
        vel_activity = np.clip(vel_activity / 10.0, 0.0, 1.0)

        return float(0.6 * rom_activity + 0.4 * vel_activity)

    def compute_stance_swing_reward(self, left_contact, right_contact,left_world_vel, right_world_vel,com_velocity):
        if left_contact and not right_contact:
            stance_speed = np.linalg.norm(left_world_vel[:2])
            stance_quality = np.exp(-(stance_speed / 0.25) ** 2)

            relative_forward_vel = right_world_vel[0] - com_velocity
            swing_quality = np.clip(max(relative_forward_vel, 0.0) / 1.0, 0.0, 1.0)

            return float(0.5 * (stance_quality + swing_quality))

        if right_contact and not left_contact:
            stance_speed = np.linalg.norm(right_world_vel[:2])
            stance_quality = np.exp(-(stance_speed / 0.25) ** 2)

            relative_forward_vel = left_world_vel[0] - com_velocity
            swing_quality = np.clip(max(relative_forward_vel, 0.0) / 1.0, 0.0, 1.0)

            return float(0.5 * (stance_quality + swing_quality))

        return 0.0

    def update_leg_state(self, side, contact, foot_vz):
        """
        单腿状态机：

            STANCE -> SWING
            SWING -> VALID TOUCHDOWN -> STANCE

        有效 touchdown 必须：
        1. 之前处于 SWING
        2. SWING 持续至少 min_swing_frames
        3. 当前重新接触地面
        4. 脚没有明显持续向上运动

        返回：
            touchdown, swing_duration
        """

        if side == "left":
            state = self.left_leg_state
            swing_frames = self.left_swing_frames
            prev_vz = self.prev_left_foot_vz
        else:
            state = self.right_leg_state
            swing_frames = self.right_swing_frames
            prev_vz = self.prev_right_foot_vz

        touchdown = False
        swing_duration = 0

        if state == "stance":
            if not contact:
                state = "swing"
                swing_frames = 1

        elif state == "swing":
            swing_frames += 1

            descending_or_stable = (foot_vz <= self.touchdown_vz_threshold or prev_vz <= self.touchdown_vz_threshold
            )

            if contact and swing_frames >= self.min_swing_frames and descending_or_stable:
                touchdown = True
                swing_duration = swing_frames
                state = "stance"
                swing_frames = 0

        if side == "left":
            self.left_leg_state = state
            self.left_swing_frames = swing_frames
            self.prev_left_foot_vz = foot_vz
        else:
            self.right_leg_state = state
            self.right_swing_frames = swing_frames
            self.prev_right_foot_vz = foot_vz

        return touchdown, swing_duration

    def compute_alternation_reward(self, left_touchdown, right_touchdown, left_x, right_x, body_x):
        """
        严格要求：
            L -> R -> L -> R

        成功交替必须同时满足：
            1. touchdown 顺序正确
            2. 当前 touchdown 脚相对上一次 touchdown 脚前移超过 margin
            3. 身体相对上一次成功交替前移超过 threshold

        第一次 touchdown 只建立 gait phase，不给奖励。
        """

        if left_touchdown == right_touchdown:
            self.last_alternation_passed = False
            return 0.0

        landed = "left" if left_touchdown else "right"
        current_x = left_x if landed == "left" else right_x
        self.total_land_count += 1

        if self.last_touchdown_foot is None:
            self.last_touchdown_foot = landed
            self.expected_touchdown = "right" if landed == "left" else "left"
            self.last_touchdown_foot_x = current_x
            self.last_completed_alternation_x = body_x
            self.last_alternation_passed = False
            return 0.0

        self.alternation_attempt_count += 1

        previous_x = self.last_touchdown_foot_x
        self.last_touchdown_foot_x = current_x

        correct_order = (landed == self.expected_touchdown)
        step_length = 0.0 if previous_x is None else current_x - previous_x
        crossed = step_length > self.alternation_margin
        advanced = (
            self.last_completed_alternation_x is None
            or body_x - self.last_completed_alternation_x >= self.alternation_advance_threshold
        )

        if correct_order and crossed and advanced:
            alternation_raw = 1.0
            self.alternation_success_count += 1
            self.last_completed_alternation_x = body_x
            self.last_alternation_passed = True
        else:
            alternation_raw = -1.0
            self.last_alternation_passed = False

        self.last_touchdown_foot = landed
        self.expected_touchdown = "right" if landed == "left" else "left"

        return float(alternation_raw)

    def compute_alternate_flip_penalty(self, left_touchdown, right_touchdown, left_x, right_x):
        current_diff = (right_x - left_x) / self.cycle_scale

        if not (left_touchdown or right_touchdown):
            return 0.0

        if abs(current_diff) < 1e-3:
            return 0.0

        if self.prev_diff_for_cycle is None:
            self.prev_diff_for_cycle = current_diff
            return 0.0

        product = self.prev_diff_for_cycle * current_diff
        self.prev_diff_for_cycle = current_diff
        self.alternate_flip_product = product

        if product <= 0.0:
            self.alternate_flip_count += 1
        else:
            self.alternate_no_flip_count += 1

        return self.cycle_weight * float(np.clip(product, -2.0, 2.0))

    def compute_step_length_reward(self, landed_foot, left_x, right_x, com_x=None):
        """
        计算相邻 touchdown 的世界坐标 step length。

        当前 step length 定义为：
            current_touchdown_x - previous_touchdown_x

        只有当上次 alternation 被判定为有效时，才给予 step length reward。
        """

        current_x = left_x if landed_foot == "left" else right_x

        if self.last_touchdown_foot_x is None:
            self.last_step_length = 0.0
            return 0.0, 0.0

        step_length = current_x - self.last_touchdown_foot_x
        self.last_step_length = step_length

        self.step_lengths.append(step_length)

        if landed_foot == "left":
            self.left_step_lengths.append(step_length)
        else:
            self.right_step_lengths.append(step_length)

        if not self.last_alternation_passed:
            return 0.0, float(step_length)

        if step_length <= 0.0:
            return 0.0, float(step_length)

        reward = np.exp(-((step_length - self.target_step_length) ** 2) / (2.0 * self.step_length_sigma ** 2))

        return float(reward), float(step_length)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        data = self.env.unwrapped.data

        for buffer in [self.left_hip_angle_buffer,self.left_knee_angle_buffer,self.right_hip_angle_buffer,
                       self.right_knee_angle_buffer,self.left_leg_vel_buffer,self.right_leg_vel_buffer,self.step_lengths,
                       self.left_step_lengths,self.right_step_lengths]:
            buffer.clear()

        self.left_swing_history.clear()
        self.right_swing_history.clear()

        self.initial_y = float(data.qpos[1])
        self.prev_body_x = float(data.qpos[0])
        self.prev_body_x_per_frame = float(data.qpos[0])
        self.prev_com_x = self._get_com_x()

        self.speed = 0.0
        self.body_advance = 0.0

        try:
            left_contact, right_contact = self._get_foot_contacts()
        except Exception:
            left_contact, right_contact = False, False

        self.left_leg_state = "stance" if left_contact else "swing"
        self.right_leg_state = "stance" if right_contact else "swing"

        self.left_swing_frames = 0
        self.right_swing_frames = 0

        self.prev_left_foot_vz = 0.0
        self.prev_right_foot_vz = 0.0

        self.expected_touchdown = None
        self.last_touchdown_foot = None
        self.last_touchdown_foot_x = None
        self.last_completed_alternation_x = None
        self.last_alternation_passed = False
        self.prev_diff_for_cycle = None
        self.last_alternate_flip_penalty = 0.0
        self.alternate_flip_product = 0.0

        self.total_frames = 0
        self.total_land_count = 0
        self.alternation_success_count = 0
        self.alternation_attempt_count = 0
        self.alternate_flip_count = 0
        self.alternate_no_flip_count = 0
        self.valid_left_touchdowns = 0
        self.valid_right_touchdowns = 0
        self.invalid_touchdowns = 0

        self.flight_frames = 0
        self.flight_frames_total = 0
        self.double_support_frames = 0
        self.double_support_frames_total = 0
        self.single_support_frames_total = 0

        self.last_left_swing_duration = 0
        self.last_right_swing_duration = 0

        self.last_step_length = 0.0
        self.last_touchdown_valid = False

        self.left_foot_geom_id = None
        self.right_foot_geom_id = None
        self.floor_geom_id = None

        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data
        dt = float(getattr(self.env.unwrapped, "dt", 0.015))
        self.total_frames += 1

        # COM / root
        com_x = self._get_com_x()
        com_advance = 0.0 if self.prev_com_x is None else com_x - self.prev_com_x
        self.prev_com_x = com_x

        body_x = float(data.qpos[0])
        self.body_advance = 0.0 if self.prev_body_x_per_frame is None else body_x - self.prev_body_x_per_frame
        self.prev_body_x_per_frame = body_x

        self.speed = com_advance / max(dt, 1e-8)

        # feet
        left_foot = data.body("left_foot")
        right_foot = data.body("right_foot")

        left_x = float(left_foot.xpos[0])
        right_x = float(right_foot.xpos[0])
        self.foot_diff = left_x - right_x

        left_world_vel, left_foot_vel = self._get_foot_velocity(left_foot)
        right_world_vel, right_foot_vel = self._get_foot_velocity(right_foot)

        left_foot_vz = float(left_world_vel[2])
        right_foot_vz = float(right_world_vel[2])

        # 真实接触
        left_contact, right_contact = self._get_foot_contacts()

        # gait state machine
        left_touchdown, left_swing_duration = self.update_leg_state("left", left_contact, left_foot_vz)
        right_touchdown, right_swing_duration = self.update_leg_state("right", right_contact, right_foot_vz)

        self.last_touchdown_valid = (left_touchdown or right_touchdown)

        if left_touchdown:
            self.valid_left_touchdowns += 1
            self.left_swing_history.append(left_swing_duration)
            self.last_left_swing_duration = left_swing_duration

        if right_touchdown:
            self.valid_right_touchdowns += 1
            self.right_swing_history.append(right_swing_duration)
            self.last_right_swing_duration = right_swing_duration

        # flight / double support
        if not left_contact and not right_contact:
            self.flight_frames += 1
            self.flight_frames_total += 1
        else:
            self.flight_frames = 0

        if left_contact and right_contact:
            self.double_support_frames += 1
            self.double_support_frames_total += 1
        else:
            self.double_support_frames = 0

        if left_contact != right_contact:
            self.single_support_frames_total += 1

        # joints
        ids = self._get_leg_joint_ids()
        qpos, qvel = data.qpos, data.qvel

        left_hip_angle = float(qpos[ids["left_hip_y"]["qpos"]])
        right_hip_angle = float(qpos[ids["right_hip_y"]["qpos"]])
        left_knee_angle = float(qpos[ids["left_knee"]["qpos"]])
        right_knee_angle = float(qpos[ids["right_knee"]["qpos"]])

        left_hip_vel = float(qvel[ids["left_hip_y"]["qvel"]])
        right_hip_vel = float(qvel[ids["right_hip_y"]["qvel"]])
        left_knee_vel = float(qvel[ids["left_knee"]["qvel"]])
        right_knee_vel = float(qvel[ids["right_knee"]["qvel"]])

        self.left_hip_angle_buffer.append(left_hip_angle)
        self.left_knee_angle_buffer.append(left_knee_angle)
        self.right_hip_angle_buffer.append(right_hip_angle)
        self.right_knee_angle_buffer.append(right_knee_angle)

        self.left_leg_vel_buffer.append(
            abs(left_hip_vel) + abs(left_knee_vel)
        )
        self.right_leg_vel_buffer.append(
            abs(right_hip_vel) + abs(right_knee_vel)
        )

        # 1. forward
        forward_reward = self.forward_weight * np.exp(-((self.speed - self.target_speed) ** 2) / (2.0 * self.speed_sigma ** 2))
        forward_reward -= self.high_speed_penalty_weight * max(0.0, self.speed - self.high_speed_threshold)

        # 2. height
        torso_height = float(qpos[2])
        height_reward = self.healthy_weight * np.clip(torso_height - 1.0, 0.0, 0.5)

        # 3. balance
        quat = qpos[3:7]
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler("xyz")
        balance_reward = 0.25
        pitch_penalty = 0.0
        if pitch < -0.20:
            pitch_penalty = (abs(pitch) - 0.20) * 3.0
        elif pitch > 0.20:
            pitch_penalty = (pitch - 0.20) * 2.0
        roll_penalty = 0.0
        if abs(roll) > 0.15:
            roll_penalty = (abs(roll) - 0.15) * 5.0

        net_balance = balance_reward - pitch_penalty - roll_penalty

        # 4. touchdown alternation
        alternation_raw = self.compute_alternation_reward(
            left_touchdown,
            right_touchdown,
            left_x,
            right_x,
            body_x,
        )
        alternation_reward = self.alternation_weight * alternation_raw

        # 4b. 相邻 touchdown 前后位置翻转信号
        alternate_flip_penalty = self.compute_alternate_flip_penalty(left_touchdown,right_touchdown,left_x,right_x,)
        self.last_alternate_flip_penalty = alternate_flip_penalty

        # 5. target step length
        step_reward = 0.0
        step_length = 0.0

        if left_touchdown:
            step_reward, step_length = self.compute_step_length_reward("left", left_x, right_x, com_x)
        elif right_touchdown:
            step_reward, step_length = self.compute_step_length_reward("right", left_x, right_x, com_x)
        else:
            step_reward, step_length = 0.0, 0.0

        step_reward *= self.step_length_weight

        # 6. single support / swing
        stance_swing_reward = self.stance_weight * \
            self.compute_stance_swing_reward(left_contact,right_contact,left_world_vel, right_world_vel,self.speed)

        # 7. flight penalty
        if not left_contact and not right_contact:
            flight_penalty = min(self.flight_penalty_max, self.flight_penalty_base + 
                                self.flight_penalty_increment * max(self.flight_frames - 1, 0))
        else:
            flight_penalty = 0.0

        # 8. double support penalty
        if (left_contact and right_contact and self.double_support_frames > self.double_support_max_frames):
            double_support_penalty = min(
                1.0, self.double_support_penalty_weight *(self.double_support_frames -self.double_support_max_frames))
        else:
            double_support_penalty = 0.0

        # 9. weak hip phase
        phase_reward = (self.phase_weight *self.compute_hip_phase_reward())

        # 10. activity 只做监控，不直接惩罚
        left_leg_activity = self.compute_leg_activity(
            self.left_hip_angle_buffer,self.left_knee_angle_buffer,self.left_leg_vel_buffer)

        right_leg_activity = self.compute_leg_activity(
            self.right_hip_angle_buffer,self.right_knee_angle_buffer,self.right_leg_vel_buffer)

        activity_difference = abs(left_leg_activity - right_leg_activity)

        # 11. hip_x
        left_hip_x = float(qpos[ids["left_hip_x"]["qpos"]])
        right_hip_x = float(qpos[ids["right_hip_x"]["qpos"]])

        hip_x_angle_penalty = 0.0

        if abs(left_hip_x) > self.hip_x_angle_threshold:
            hip_x_angle_penalty += (abs(left_hip_x) -self.hip_x_angle_threshold) * self.hip_x_penalty_weight

        if abs(right_hip_x) > self.hip_x_angle_threshold:
            hip_x_angle_penalty += (abs(right_hip_x) -self.hip_x_angle_threshold) * self.hip_x_penalty_weight

        # 12. lateral
        lateral_pos = abs(float(qpos[1]) -self.initial_y)
        lateral_vel = abs(float(data.qvel[1]))
        lateral_penalty = self.lateral_weight * (0.5 * lateral_pos +0.2 * lateral_vel)

        # 13. survival / control / fall
        survival_bonus = (0.10 if torso_height > 1.0 else 0.0)

        ctrl_cost = (self.ctrl_cost_weight * np.square(action).sum())

        fall_penalty = (self.fall_penalty_weight if terminated else 0.0)

        # final reward
        reward = (
            forward_reward
            + stance_swing_reward
            + alternation_reward
            + step_reward
            + net_balance
            + height_reward
            + phase_reward
            + survival_bonus
            + fall_penalty
            - alternate_flip_penalty
            - flight_penalty
            - double_support_penalty
            - lateral_penalty
            - hip_x_angle_penalty
            - ctrl_cost
        )

        # info
        info.update({
            "speed": float(self.speed),
            "forward_reward": float(forward_reward),
            "height_reward": float(height_reward),
            "net_balance": float(net_balance),
            "pitch_angle": float(pitch),
            "roll_angle": float(roll),

            "stance_swing_reward": float(stance_swing_reward),
            "alternation_reward": float(alternation_reward),
            "alternation_raw": float(alternation_raw),
            "alternation_success": int(self.last_alternation_passed),

            "alternate_flip_penalty": float(alternate_flip_penalty),
            "alternate_flip_product": float(self.alternate_flip_product),
            "alternate_flip_count": self.alternate_flip_count,
            "alternate_no_flip_count": self.alternate_no_flip_count,
            "prev_diff_for_cycle": float(self.prev_diff_for_cycle) if self.prev_diff_for_cycle is not None else 0.0,

            "step_reward": float(step_reward),
            "step_length": float(step_length),

            "phase_reward": float(phase_reward),

            "flight_penalty": float(flight_penalty),
            "double_support_penalty": float(double_support_penalty),

            "lateral_penalty": float(lateral_penalty),
            "hip_x_angle_penalty": float(hip_x_angle_penalty),

            "activity_difference": float(activity_difference),
            "left_leg_activity": float(left_leg_activity),
            "right_leg_activity": float(right_leg_activity),

            "ctrl_cost": float(ctrl_cost),

            "left_contact": int(left_contact),
            "right_contact": int(right_contact),

            "left_touchdown": int(left_touchdown),
            "right_touchdown": int(right_touchdown),

            "left_leg_state": self.left_leg_state,
            "right_leg_state": self.right_leg_state,

            "left_swing_frames": int(self.left_swing_frames),
            "right_swing_frames": int(self.right_swing_frames),

            "last_left_swing_duration": int(self.last_left_swing_duration),
            "last_right_swing_duration": int(self.last_right_swing_duration),

            "left_foot_vz": float(left_foot_vz),
            "right_foot_vz": float(right_foot_vz),

            "expected_touchdown": self.expected_touchdown,
            "last_touchdown_foot": self.last_touchdown_foot,

            "foot_diff": float(self.foot_diff),
            "body_advance": float(self.body_advance),
            "last_touchdown_foot_x": float(self.last_touchdown_foot_x) if self.last_touchdown_foot_x is not None else 0.0,
            "last_completed_alternation_x": float(self.last_completed_alternation_x) if self.last_completed_alternation_x is not None else 0.0,

            "alternation_success_rate": (self.alternation_success_count / max(self.alternation_attempt_count, 1)),
            "flight_ratio": (self.flight_frames_total / max(self.total_frames, 1)),
            "double_support_ratio": (self.double_support_frames_total / max(self.total_frames, 1)),
            "single_support_ratio": (self.single_support_frames_total / max(self.total_frames, 1)),
            "valid_left_touchdowns": self.valid_left_touchdowns,
            "valid_right_touchdowns": self.valid_right_touchdowns
        })

        if terminated or truncated:
            info["mean_left_swing"] = float(np.mean(self.left_swing_history) if self.left_swing_history else 0.0)
            info["mean_right_swing"] = float(np.mean(self.right_swing_history) if self.right_swing_history else 0.0)
            info["mean_step_length"] = float(np.mean(self.step_lengths) if self.step_lengths else 0.0)
            info["mean_left_step_length"] = float(np.mean(self.left_step_lengths) if self.left_step_lengths else 0.0)
            info["mean_right_step_length"] = float(np.mean(self.right_step_lengths) if self.right_step_lengths else 0.0)

        return obs, float(reward), terminated, truncated, info



def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id, max_episode_steps=1500)
        env = WalkRewardWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == "__main__":
    env_id = "Humanoid-v4"
    n_train_envs = 8    # 训练环境数量
    n_eval_envs = 1     # 评估环境数量（通常 1 个即可）
    total_timesteps = 5_000_000
    seed = 42

    # ---- 训练环境 ----
    train_vec_env = SubprocVecEnv(
        [make_env(env_id, i, seed) for i in range(n_train_envs)]
    )
    # 如果需要，可以包装 VecMonitor，同时用于训练日志
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
        learning_rate=lambda prog: 3e-4 * prog + 1e-5 * (1 - prog),  # 线性增长：3e-4 -> 1e-5  
        buffer_size=1_500_000,
        batch_size=1024,
        tau=0.01,
        gamma=0.99,
        train_freq=512,
        gradient_steps=512,
        ent_coef=0.15,              # 固定高熵探索
        target_entropy=-8.5,      # 动作维度17
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./sac_humanoid_walk/"
    )                                                   

    # ---------- 加载之前保存的模型 ----------
    # model = SAC.load("./best_model/best_model", env=train_vec_env)
    # old_model = SAC.load("./best_model/best_model", env=train_vec_env)
    # model.policy.load_state_dict(old_model.policy.state_dict())
    # model.critic.load_state_dict(old_model.critic.state_dict())
    # model.critic_target.load_state_dict(old_model.critic_target.state_dict())
    # model.replay_buffer.reset()

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
        render=False
    )

    model.learn(
        total_timesteps=total_timesteps,
        # reset_num_timesteps=False,          # 关键：保持总步数计数，不从头算
        callback=[leg_callback,eval_callback]
    )
    model.save("sac_humanoid_walk_final")

    print("训练完成！模型已保存。")
