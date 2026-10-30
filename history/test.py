import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from gymnasium import Wrapper


class WalkRewardWrapper(Wrapper):
    """
    Humanoid-v4 Walking Reward V2.1

    目标：
    1. 稳定向前行走，而不是原地站立
    2. 左右脚严格交替 touchdown：L -> R -> L -> R
    3. 主要采用单脚支撑，允许短暂双脚支撑
    4. 双脚同时离地强惩罚，抑制 hopping / running
    5. 步长接近目标值，而不是越大越好
    6. 使用 swing -> touchdown -> stance 状态机识别真实迈步
    7. hip phase 仅作为弱辅助奖励
    """

    def __init__(self, env):
        super().__init__(env)

        # 基础奖励
        self.forward_weight = 1.5
        self.target_speed = 1.2
        self.speed_sigma = 0.30
        self.high_speed_threshold = 1.5
        self.high_speed_penalty_weight = 0.8
        self.healthy_weight = 0.8
        self.ctrl_cost_weight = 0.005
        self.fall_penalty_weight = -50.0

        # gait
        self.alternation_weight = 1.5
        self.stance_weight = 1.0
        self.target_step_length = 0.30
        self.step_length_sigma = 0.10
        self.step_length_weight = 1.0

        # touchdown detector
        self.min_swing_frames = 8
        self.touchdown_vz_threshold = 0.20

        # double support
        self.double_support_max_frames = 6
        self.double_support_penalty_weight = 0.20

        # flight / hopping
        self.flight_penalty_base = 4.0
        self.flight_penalty_increment = 0.5
        self.flight_penalty_max = 6.0

        # phase
        self.phase_weight = 0.05

        # lateral / hip_x
        self.lateral_weight = 0.10
        self.hip_x_angle_threshold = 0.30
        self.hip_x_penalty_weight = 1.5

        # 状态
        self.initial_y = None
        self.prev_body_x = None
        self.prev_body_x_per_frame = None
        self.prev_com_x = None
        self.speed = 0.0
        self.body_advance = 0.0

        # 每条腿的 gait state
        self.left_leg_state = "stance"
        self.right_leg_state = "stance"
        self.left_swing_frames = 0
        self.right_swing_frames = 0

        # 上一帧脚部垂直速度
        self.prev_left_foot_vz = 0.0
        self.prev_right_foot_vz = 0.0

        # contact
        self.left_contact_prev = False
        self.right_contact_prev = False

        # gait phase
        self.expected_touchdown = None
        self.last_touchdown_foot = None
        self.last_touchdown_x = None

        # 统计
        self.total_frames = 0
        self.total_land_count = 0
        self.alternation_success_count = 0
        self.alternation_attempt_count = 0

        self.flight_frames = 0
        self.flight_frames_total = 0
        self.double_support_frames = 0
        self.double_support_frames_total = 0
        self.single_support_frames_total = 0

        self.valid_left_touchdowns = 0
        self.valid_right_touchdowns = 0
        self.invalid_touchdowns = 0

        # step
        self.step_lengths = deque(maxlen=50)
        self.left_step_lengths = deque(maxlen=25)
        self.right_step_lengths = deque(maxlen=25)

        # swing history
        self.left_swing_history = []
        self.right_swing_history = []

        # phase/activity buffer
        self.leg_activity_buffer_size = 50
        self.left_hip_angle_buffer = deque(maxlen=50)
        self.left_knee_angle_buffer = deque(maxlen=50)
        self.right_hip_angle_buffer = deque(maxlen=50)
        self.right_knee_angle_buffer = deque(maxlen=50)
        self.left_leg_vel_buffer = deque(maxlen=50)
        self.right_leg_vel_buffer = deque(maxlen=50)

        # cache
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
        joint_names = [
            "left_hip_x", "left_hip_y", "left_knee",
            "right_hip_x", "right_hip_y", "right_knee"
        ]

        self.leg_joint_ids = {}
        for name in joint_names:
            jid = model.joint(name).id
            self.leg_joint_ids[name] = {
                "qpos": model.jnt_qposadr[jid],
                "qvel": model.jnt_dofadr[jid]
            }

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

            if ((g1 == foot_geom_id and g2 == floor_geom_id) or
                    (g2 == foot_geom_id and g1 == floor_geom_id)):
                return contact.dist <= 1e-3

        return False

    def _get_foot_contacts(self):
        left_geom, right_geom, floor_geom = self._get_contact_geom_ids()

        return (
            self._foot_floor_contact(left_geom, floor_geom),
            self._foot_floor_contact(right_geom, floor_geom)
        )

    def _get_com_x(self):
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model
        total_mass = np.sum(model.body_mass)

        if total_mass <= 0:
            return float(data.qpos[0])

        return float(np.sum(model.body_mass * data.xipos[:, 0]) / total_mass)

    def _get_foot_velocity(self, body):
        world_vel = body.cvel[3:] @ body.xmat.reshape(3, 3)
        horizontal_speed = np.linalg.norm(world_vel[:2])
        return world_vel, float(horizontal_speed)

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

    def compute_stance_swing_reward(self, left_contact, right_contact,
                                    left_world_vel, right_world_vel,
                                    com_velocity):
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
            SWING -> TOUCHDOWN -> STANCE

        touchdown 必须：
        1. 之前确实处于 swing
        2. swing 至少持续 min_swing_frames
        3. 当前重新接触地面
        4. 脚垂直速度不能明显向上

        返回：
            valid_touchdown
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

        if state == "stance":
            if not contact:
                state = "swing"
                swing_frames = 1

        elif state == "swing":
            swing_frames += 1

            # 使用当前/上一帧的垂直速度判断脚是否处于下降或接近落地状态
            descending = (
                foot_vz <= self.touchdown_vz_threshold
                or prev_vz <= self.touchdown_vz_threshold
            )

            if contact and swing_frames >= self.min_swing_frames and descending:
                touchdown = True
                state = "stance"
                swing_frames = 0

            elif contact and swing_frames >= self.min_swing_frames:
                # 已经满足最小 swing 时间，即使速度略有正值，
                # 只要重新接触地面，也允许触地事件成立
                touchdown = True
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

        return touchdown

    def compute_alternation_reward(self, left_touchdown, right_touchdown):
        """
        gait 层面严格要求：
            L -> R -> L -> R

        第一次有效 touchdown：
            仅建立 gait phase

        后续：
            正确交替 +1
            错误顺序 -1
        """

        if left_touchdown == right_touchdown:
            return 0.0

        landed = "left" if left_touchdown else "right"
        self.total_land_count += 1

        if self.last_touchdown_foot is None:
            self.last_touchdown_foot = landed
            self.expected_touchdown = "right" if landed == "left" else "left"
            return 0.2

        self.alternation_attempt_count += 1

        if landed == self.expected_touchdown:
            reward = 1.0
            self.alternation_success_count += 1
        else:
            reward = -1.0

        self.last_touchdown_foot = landed
        self.expected_touchdown = "right" if landed == "left" else "left"

        return float(reward)

    def compute_step_length_reward(self, landed_foot, left_x, right_x):
        """
        相邻两个有效 touchdown 之间的脚部前移距离。

        注意：
        这里采用世界坐标的连续 touchdown 位置作为第一版 step length。
        后续如果发现整体 COM 前移导致步长估计偏差，再改成相对 COM 的定义。
        """

        current_x = left_x if landed_foot == "left" else right_x

        if self.last_touchdown_x is None:
            self.last_touchdown_x = current_x
            self.last_step_length = 0.0
            return 0.0, 0.0

        step_length = current_x - self.last_touchdown_x
        self.last_touchdown_x = current_x
        self.last_step_length = step_length

        self.step_lengths.append(step_length)

        if landed_foot == "left":
            self.left_step_lengths.append(step_length)
        else:
            self.right_step_lengths.append(step_length)

        if step_length <= 0.0:
            return 0.0, float(step_length)

        reward = np.exp(
            -((step_length - self.target_step_length) ** 2) /
            (2.0 * self.step_length_sigma ** 2)
        )

        return float(reward), float(step_length)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        data = self.env.unwrapped.data

        for buffer in [
            self.left_hip_angle_buffer,
            self.left_knee_angle_buffer,
            self.right_hip_angle_buffer,
            self.right_knee_angle_buffer,
            self.left_leg_vel_buffer,
            self.right_leg_vel_buffer,
            self.step_lengths,
            self.left_step_lengths,
            self.right_step_lengths
        ]:
            buffer.clear()

        self.left_swing_history.clear()
        self.right_swing_history.clear()

        self.initial_y = float(data.qpos[1])
        self.prev_body_x = float(data.qpos[0])
        self.prev_body_x_per_frame = float(data.qpos[0])
        self.prev_com_x = self._get_com_x()

        self.speed = 0.0
        self.body_advance = 0.0

        # 初始化真实 contact
        try:
            self.left_contact_prev, self.right_contact_prev = self._get_foot_contacts()
        except Exception:
            self.left_contact_prev = False
            self.right_contact_prev = False

        # 根据 reset 后的实际接触状态初始化 leg state
        self.left_leg_state = "stance" if self.left_contact_prev else "swing"
        self.right_leg_state = "stance" if self.right_contact_prev else "swing"

        self.left_swing_frames = 0
        self.right_swing_frames = 0

        self.prev_left_foot_vz = 0.0
        self.prev_right_foot_vz = 0.0

        self.expected_touchdown = None
        self.last_touchdown_foot = None
        self.last_touchdown_x = None

        self.total_frames = 0
        self.total_land_count = 0
        self.alternation_success_count = 0
        self.alternation_attempt_count = 0

        self.flight_frames = 0
        self.flight_frames_total = 0
        self.double_support_frames = 0
        self.double_support_frames_total = 0
        self.single_support_frames_total = 0

        self.valid_left_touchdowns = 0
        self.valid_right_touchdowns = 0
        self.invalid_touchdowns = 0

        self.left_foot_geom_id = None
        self.right_foot_geom_id = None
        self.floor_geom_id = None

        self.foot_diff = 0.0
        self.last_step_length = 0.0
        self.last_touchdown_valid = False

        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data
        dt = float(getattr(self.env.unwrapped, "dt", 0.015))
        self.total_frames += 1

        # COM
        com_x = self._get_com_x()
        com_advance = 0.0 if self.prev_com_x is None else com_x - self.prev_com_x
        self.prev_com_x = com_x
        self.speed = com_advance / max(dt, 1e-8)

        # root body
        body_x = float(data.qpos[0])
        self.body_advance = (
            0.0 if self.prev_body_x_per_frame is None
            else body_x - self.prev_body_x_per_frame
        )
        self.prev_body_x_per_frame = body_x

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

        # contact
        left_contact, right_contact = self._get_foot_contacts()

        # ------------------------------------------------------------
        # gait state machine
        # ------------------------------------------------------------

        left_touchdown = self.update_leg_state(
            "left",
            left_contact,
            left_foot_vz
        )

        right_touchdown = self.update_leg_state(
            "right",
            right_contact,
            right_foot_vz
        )

        self.last_touchdown_valid = (
            left_touchdown or right_touchdown
        )

        if left_touchdown:
            self.valid_left_touchdowns += 1

        if right_touchdown:
            self.valid_right_touchdowns += 1

        # ------------------------------------------------------------
        # flight / double support statistics
        # ------------------------------------------------------------

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

        # ============================================================
        # 1. Forward
        # ============================================================

        forward_reward = self.forward_weight * np.exp(
            -((self.speed - self.target_speed) ** 2) /
            (2.0 * self.speed_sigma ** 2)
        )

        forward_reward -= self.high_speed_penalty_weight * max(
            0.0,
            self.speed - self.high_speed_threshold
        )

        # ============================================================
        # 2. Height
        # ============================================================

        torso_height = float(qpos[2])

        height_reward = self.healthy_weight * np.clip(
            torso_height - 1.0,
            0.0,
            0.5
        )

        # ============================================================
        # 3. Balance
        # ============================================================

        quat = qpos[3:7]
        r = Rotation.from_quat([
            quat[1], quat[2], quat[3], quat[0]
        ])

        roll, pitch, yaw = r.as_euler("xyz")

        balance_reward = 0.3

        pitch_penalty = 0.0
        if pitch < -0.20:
            pitch_penalty = (abs(pitch) - 0.20) * 3.0
        elif pitch > 0.20:
            pitch_penalty = (pitch - 0.20) * 2.0

        roll_penalty = 0.0
        if abs(roll) > 0.15:
            roll_penalty = (abs(roll) - 0.15) * 5.0

        net_balance = balance_reward - pitch_penalty - roll_penalty

        # ============================================================
        # 4. Touchdown alternation
        # ============================================================

        alternation_raw = self.compute_alternation_reward(
            left_touchdown,
            right_touchdown
        )

        alternation_reward = (
            self.alternation_weight * alternation_raw
        )

        # ============================================================
        # 5. Step length
        # ============================================================

        step_reward = 0.0
        step_length = 0.0

        if left_touchdown:
            step_reward, step_length = self.compute_step_length_reward(
                "left",
                left_x,
                right_x
            )

        elif right_touchdown:
            step_reward, step_length = self.compute_step_length_reward(
                "right",
                left_x,
                right_x
            )

        step_reward *= self.step_length_weight

        # ============================================================
        # 6. Single support / swing
        # ============================================================

        stance_swing_reward = (
            self.stance_weight *
            self.compute_stance_swing_reward(
                left_contact,
                right_contact,
                left_world_vel,
                right_world_vel,
                self.speed
            )
        )

        # ============================================================
        # 7. Flight penalty
        # ============================================================

        if not left_contact and not right_contact:
            flight_penalty = min(
                self.flight_penalty_max,
                self.flight_penalty_base +
                self.flight_penalty_increment *
                max(self.flight_frames - 1, 0)
            )
        else:
            flight_penalty = 0.0

        # ============================================================
        # 8. Double support penalty
        # ============================================================

        if (
            left_contact and
            right_contact and
            self.double_support_frames >
            self.double_support_max_frames
        ):
            double_support_penalty = min(
                1.0,
                self.double_support_penalty_weight *
                (
                    self.double_support_frames
                    -
                    self.double_support_max_frames
                )
            )
        else:
            double_support_penalty = 0.0

        # ============================================================
        # 9. Hip phase - weak auxiliary
        # ============================================================

        phase_reward = (
            self.phase_weight *
            self.compute_hip_phase_reward()
        )

        # ============================================================
        # 10. Leg activity - only monitoring
        # ============================================================

        left_leg_activity = self.compute_leg_activity(
            self.left_hip_angle_buffer,
            self.left_knee_angle_buffer,
            self.left_leg_vel_buffer
        )

        right_leg_activity = self.compute_leg_activity(
            self.right_hip_angle_buffer,
            self.right_knee_angle_buffer,
            self.right_leg_vel_buffer
        )

        activity_difference = abs(
            left_leg_activity -
            right_leg_activity
        )

        # ============================================================
        # 11. Hip X
        # ============================================================

        left_hip_x = float(qpos[ids["left_hip_x"]["qpos"]])
        right_hip_x = float(qpos[ids["right_hip_x"]["qpos"]])

        hip_x_angle_penalty = 0.0

        if abs(left_hip_x) > self.hip_x_angle_threshold:
            hip_x_angle_penalty += (
                abs(left_hip_x) -
                self.hip_x_angle_threshold
            ) * self.hip_x_penalty_weight

        if abs(right_hip_x) > self.hip_x_angle_threshold:
            hip_x_angle_penalty += (
                abs(right_hip_x) -
                self.hip_x_angle_threshold
            ) * self.hip_x_penalty_weight

        # ============================================================
        # 12. Lateral
        # ============================================================

        lateral_pos = abs(
            float(qpos[1]) -
            self.initial_y
        )

        lateral_vel = abs(
            float(data.qvel[1])
        )

        lateral_penalty = self.lateral_weight * (
            0.5 * lateral_pos +
            0.2 * lateral_vel
        )

        # ============================================================
        # 13. Survival / control / fall
        # ============================================================

        survival_bonus = (
            0.10
            if torso_height > 1.0
            else 0.0
        )

        ctrl_cost = (
            self.ctrl_cost_weight *
            np.square(action).sum()
        )

        fall_penalty = (
            self.fall_penalty_weight
            if terminated
            else 0.0
        )

        # ============================================================
        # Final reward
        # ============================================================

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
            - flight_penalty
            - double_support_penalty
            - lateral_penalty
            - hip_x_angle_penalty
            - ctrl_cost
        )

        # ============================================================
        # Info
        # ============================================================

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

            # contact
            "left_contact": int(left_contact),
            "right_contact": int(right_contact),

            # gait events
            "left_touchdown": int(left_touchdown),
            "right_touchdown": int(right_touchdown),

            "left_leg_state": self.left_leg_state,
            "right_leg_state": self.right_leg_state,

            "left_swing_frames": int(self.left_swing_frames),
            "right_swing_frames": int(self.right_swing_frames),

            "expected_touchdown": self.expected_touchdown,
            "last_touchdown_foot": self.last_touchdown_foot,

            # foot velocity
            "left_foot_vz": float(left_foot_vz),
            "right_foot_vz": float(right_foot_vz),

            # debug
            "foot_diff": float(self.foot_diff),
            "body_advance": float(self.body_advance),

            # statistics
            "alternation_success_rate": (
                self.alternation_success_count /
                max(self.alternation_attempt_count, 1)
            ),
            "flight_ratio": (
                self.flight_frames_total /
                max(self.total_frames, 1)
            ),
            "double_support_ratio": (
                self.double_support_frames_total /
                max(self.total_frames, 1)
            ),
            "single_support_ratio": (
                self.single_support_frames_total /
                max(self.total_frames, 1)
            ),
            "valid_left_touchdowns": self.valid_left_touchdowns,
            "valid_right_touchdowns": self.valid_right_touchdowns
        })

        if terminated or truncated:
            info["mean_left_swing"] = float(
                np.mean(self.left_swing_history)
                if self.left_swing_history else 0.0
            )
            info["mean_right_swing"] = float(
                np.mean(self.right_swing_history)
                if self.right_swing_history else 0.0
            )
            info["mean_step_length"] = float(
                np.mean(self.step_lengths)
                if self.step_lengths else 0.0
            )
            info["mean_left_step_length"] = float(
                np.mean(self.left_step_lengths)
                if self.left_step_lengths else 0.0
            )
            info["mean_right_step_length"] = float(
                np.mean(self.right_step_lengths)
                if self.right_step_lengths else 0.0
            )

        return obs, float(reward), terminated, truncated, info