"""Humanoid-v4 的 SAC 训练主脚本。

- WalkRewardWrapper：把行走拆成推进 / 放置 / 步幅 / 摆动 / 姿态 / 生存 / 税七类奖励，逐帧结算并
  记录步态指标；osc_enable=True 时启用摆腿时钟模板，在摆动窗口接管两条 hip_y。
- 镜像增广：MirrorReplayer / EpisodeRecorder / MirrorInjector / MirrorAugCallback 在 worker 内
  录制整集、镜像后写回 replay buffer。
- 回调：LegActivityCallback（探索强度等监控）、SaveExtraParamsOnBest / SaveBestModelOnNewBest
  （保存最佳权重与快照）、OscCurriculum（模板系数 k 的同步 / 课程）。
- MirrorEqSAC：带等变正则的 SAC 子类（当前 λ=0，关闭）。
入口在文件末尾的 `if __name__ == "__main__"` 块。
"""

import os,json
import gymnasium as gym
import mujoco
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import time as _time
import gc
from gymnasium import Wrapper
from collections import deque
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold,BaseCallback,CallbackList
from stable_baselines3.common.utils import polyak_update
from mirror_map import MirrorMap, patch_mirror_symmetry


class LegActivityCallback(BaseCallback):
    """记录探索强度、动作标准差与最近一次 gait 指标，供 SaveExtraParamsOnBest 取用。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_gait_data = {}

    def _on_step(self):
        infos = self.locals["infos"]
        actions = self.locals.get("actions")
        if actions is not None:
            self.logger.record("train/action_std", float(np.std(np.asarray(actions), axis=0).mean()))
        if self.n_calls % 100 == 0:
            obs = self.model._last_obs
            if obs is not None:
                space = self.model.action_space
                scale = float((space.high[0] - space.low[0]) / 2.0)  # 0.4: actor 空间 → env 空间
                obs_t, _ = self.model.policy.obs_to_tensor(obs)
                with th.no_grad():
                    _, log_std, _ = self.model.policy.actor.get_action_dist_params(obs_t)
                self.logger.record("train/exploration_std", float(th.exp(log_std).mean()) * scale)
        for info in infos:
            self.logger.record_mean("gait/flight_phase_penalty", info["flight_phase_penalty"])
            self.logger.record_mean("gait/activity_difference", info.get("activity_difference",0))
            self.logger.record_mean("gait/phase_reward", info.get("phase_reward",0))
            self.logger.record_mean("gait/ctrl_cost", info.get("ctrl_cost",0))

            self.logger.record_mean("gait/position_swap_reward", info.get("position_swap_reward",0))
            self.logger.record_mean("gait/contact_phase_reward", info.get("contact_phase_reward",0))
            self.logger.record_mean("gait/stride_bonus", info.get("stride_bonus",0))
            self.logger.record_mean("gait/stride_bonus_rate", info.get("stride_bonus_rate",0.0))
            self.logger.record_mean("gait/hip_xz_angle_penalty", info.get("hip_xz_angle_penalty",0))
            self.logger.record_mean("gait/short_stride_event_rate", info.get("short_stride_event_rate",0))
            self.logger.record_mean("gait/negative_stride_penalty", info.get("negative_stride_penalty",0))
            self.logger.record_mean("gait/negative_stride_rate", info.get("negative_stride_rate",0.0))
            self.logger.record_mean("gait/stride_balance_penalty", info.get("stride_balance_penalty",0.0))
            self.logger.record_mean("gait/last_left_stride", info.get("last_left_stride",0.0))
            self.logger.record_mean("gait/last_right_stride", info.get("last_right_stride",0.0))

            self.logger.record_mean("gait/support_sym_penalty", info.get("support_sym_penalty",0))
            self.logger.record_mean("gait/double_support_frac", info.get("double_support_frac", 0.0))
            self.logger.record_mean("gait/flight_frac", info.get("flight_frac", 0.0))
            self.logger.record_mean("gait/stance_frac_L", info.get("stance_frac_L", 0.0))
            self.logger.record_mean("gait/stance_frac_R", info.get("stance_frac_R", 0.0))
            self.logger.record_mean("gait/slip_penalty", info.get("slip_penalty", 0.0))
            self.logger.record_mean("gait/slip_speed_L", info.get("slip_speed_L", 0.0))
            self.logger.record_mean("gait/slip_speed_R", info.get("slip_speed_R", 0.0))
            self.logger.record_mean("gait/lateral_penalty", info.get("lateral_penalty",0))
            self.logger.record_mean("gait/alternate_penalty", info.get("alternate_penalty",0))
            self.logger.record_mean("gait/stall_penalty", info.get("stall_penalty",0))
            self.logger.record_mean("gait/speed", info.get("speed", 0))
            self.logger.record_mean("gait/position_swap_rate", info.get("position_swap_rate", 0))
            self.logger.record_mean("gait/alternate_reward_rate", info.get("alternate_reward_rate", 0))
            self.logger.record_mean("gait/alternate_penalty_rate", info.get("alternate_penalty_rate", 0))
            self.logger.record_mean("gait/forward_reward", info.get("forward_reward", 0))
            self.logger.record_mean("gait/forward_gauss", info.get("forward_gauss", 0))
            self.logger.record_mean("gait/speed_penalty", info.get("speed_penalty", 0))
            self.logger.record_mean("gait/walk_gate", info.get("walk_gate", 0))
            self.logger.record_mean("gait/single_frac", info.get("single_frac", 0))
            self.logger.record_mean("gait/survival_bonus", info.get("survival_bonus", 0))
            self.logger.record_mean("gait/ds_landing_rate", info.get("ds_landing_rate", 0))
            self.logger.record_mean("gait/flat_gate", info.get("flat_gate", 1.0))
            self.logger.record_mean("gait/speed_ramp", info.get("speed_ramp", 1.0))
            self.logger.record_mean("gait/forward_gate", info.get("forward_gate", 1.0))
            if "asym_injected" in info:
                self.logger.record_mean("gait/asym_injected", info["asym_injected"])
                self.logger.record_mean("gait/asym_inject_rate", info["asym_inject_rate"])
            self.logger.record_mean("gait/dense_ds_factor", info.get("dense_ds_factor", 1.0))
            self.logger.record_mean("gait/tiny_step_frame_frac", info.get("tiny_step_frame_frac", 0.0))
            self.logger.record_mean("gait/swap_gate", info.get("swap_gate", 1.0))
            self.logger.record_mean("gait/phase_reward_unpaid", info.get("phase_reward_unpaid", 0.0))
            self.logger.record_mean("gait/place_reward", info.get("place_reward", 0.0))
            self.logger.record_mean("gait/place_operand", info.get("place_operand", 0.0))
            self.logger.record_mean("gait/short_stride_penalty", info.get("short_stride_penalty", 0.0))
            self.logger.record_mean("gait/alternate_paid", info.get("alternate_paid", 0.0))
            self.logger.record_mean("gait/osc_active", info.get("osc_active", 0.0))
            self.logger.record_mean("gait/osc_active_L", info.get("osc_active_L", 0.0))
            self.logger.record_mean("gait/osc_active_R", info.get("osc_active_R", 0.0))
            self.logger.record_mean("gait/osc_phase", info.get("osc_phase", 0.0))
            if "asym_convert_rate" in info:
                self.logger.record_mean("gait/asym_convert_rate", info["asym_convert_rate"])
            self.logger.record_mean("gait/height_reward", info.get("height_reward", 0))
            self.logger.record_mean("gait/net_balance", info.get("net_balance", 0))
            self.logger.record_mean("gait/pitch_angle", info.get("pitch_angle", 0))
            self.logger.record_mean("gait/roll_angle", info.get("roll_angle", 0))
            self.logger.record_mean("gait/swing_forward_reward", info.get("swing_forward_reward", 0))
            self.logger.record_mean("gait/reach_reward", info.get("reach_reward", 0))
            self.logger.record_mean("gait/swing_min_reward", info.get("swing_min_reward", 0))
            self.logger.record_mean("gait/last_rom_L_deg", info.get("last_rom_L_deg", 0.0))
            self.logger.record_mean("gait/last_rom_R_deg", info.get("last_rom_R_deg", 0.0))
            self.logger.record_mean("gait/last_reach_L", info.get("last_reach_L", 0.0))
            self.logger.record_mean("gait/last_reach_R", info.get("last_reach_R", 0.0))
            for k, v in info.items():
                if not k.startswith("ev_"):
                    continue
                if k.startswith(("ev_spacing_air_", "ev_spacing_ds_", "ev_hip_at_land_")):
                    continue
                if not k.startswith(("ev_step_adv_", "ev_spacing_", "ev_reach_",
                                     "ev_rom_", "ev_lift_", "ev_knee_ext_", "ev_slip_")):
                    continue
                if k.rsplit("_", 1)[-1] not in ("mean", "n", "p50", "negfrac"):
                    continue
                self.logger.record_mean(f"gait/{k}", float(v))


            self.last_gait_data.update({
                "activity_difference": info.get("activity_difference",0),
                "phase_reward": info.get("phase_reward",0),
                "position_swap_reward": info.get("position_swap_reward",0),
                "contact_phase_reward": info.get("contact_phase_reward",0),
                "stride_bonus": info.get("stride_bonus",0),
                "stride_event_count": info.get("stride_event_count",0),
                "stride_bonus_event_count": info.get("stride_bonus_event_count",0),
                "stride_length": info.get("stride_length",0.0),
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
                "speed": info.get("speed", 0.0),
                "position_swap_rate": info.get("position_swap_rate", 0.0),
                "alternate_reward_rate": info.get("alternate_reward_rate", 0.0),
                "alternate_penalty_rate": info.get("alternate_penalty_rate", 0.0),
                "short_stride_penalty": info.get("short_stride_penalty", 0.0),
                "short_stride_event_rate": info.get("short_stride_event_rate", 0.0),
                "negative_stride_penalty": info.get("negative_stride_penalty", 0.0),
                "negative_stride_event_count": info.get("negative_stride_event_count", 0.0),
                "negative_stride_rate": info.get("negative_stride_rate", 0.0),
                "stride_balance_penalty": info.get("stride_balance_penalty", 0.0),
                "last_left_stride": info.get("last_left_stride", 0.0),
                "last_right_stride": info.get("last_right_stride", 0.0),
                "flight_phase_penalty": info.get("flight_phase_penalty", 0.0),
                "forward_gate": info.get("forward_gate", 1.0),
                "tiny_step_frame_frac": info.get("tiny_step_frame_frac", 0.0)
            })
        return True


class SaveExtraParamsOnBest(BaseCallback):
    """每次刷新最佳评估时，把最近一次 gait 指标追加到 best_extra_logs.jsonl。"""

    def __init__(self, save_path, leg_callback, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.leg_callback = leg_callback  # 引用 LegActivityCallback 实例
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
                "gait/position_swap_reward": gait_data.get("position_swap_reward"),
                "gait/contact_phase_reward": gait_data.get("contact_phase_reward"),
                "gait/stride_bonus": gait_data.get("stride_bonus"),
                "gait/stride_event_count": gait_data.get("stride_event_count"),
                "gait/stride_bonus_event_count": gait_data.get("stride_bonus_event_count"),
                "gait/stride_length": gait_data.get("stride_length"),
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
                "gait/speed": gait_data.get("speed"),
                "gait/position_swap_rate": gait_data.get("position_swap_rate"),
                "gait/alternate_reward_rate": gait_data.get("alternate_reward_rate"),
                "gait/alternate_penalty_rate": gait_data.get("alternate_penalty_rate"),
                "gait/ctrl_cost":gait_data.get("ctrl_cost"),
                "gait/short_stride_penalty": gait_data.get("short_stride_penalty"),
                "gait/short_stride_event_rate": gait_data.get("short_stride_event_rate"),
                "gait/negative_stride_penalty": gait_data.get("negative_stride_penalty"),
                "gait/negative_stride_event_count": gait_data.get("negative_stride_event_count"),
                "gait/negative_stride_rate": gait_data.get("negative_stride_rate"),
                "gait/last_left_stride": gait_data.get("last_left_stride"),
                "gait/last_right_stride": gait_data.get("last_right_stride"),
                "gait/flight_phase_penalty": gait_data.get("flight_phase_penalty"),
                "gait/forward_gate": gait_data.get("forward_gate"),
                "gait/tiny_step_frame_frac": gait_data.get("tiny_step_frame_frac")
            }

            extra_data = {key: _json_safe(value) for key, value in extra_data.items()}

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
        if self.model is not None:
            timestep = self.num_timesteps
            path = os.path.join(self.save_dir, f"best_model_{timestep}.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"✅ 新最佳模型已保存至 {path}")
        return True


class WalkRewardWrapper(Wrapper):
    """把 Humanoid-v4 包装成行走任务：逐帧计算奖励并记录步态指标。

    奖励分三层：推进收入（速度）、步态形状（放置 / 步幅 / 摆动 / 姿态）、门控与税
    （决定收入能不能拿到）。osc_enable=True 时启用摆腿时钟模板：摆动窗口由时钟接管
    两条 hip_y，支撑期把控制权交回策略。
    """
    def __init__(self, env, swing_drive_weight=0.0, episode_log_path=None,
                 asym_inject_p=0.0, asym_inject_d=(-0.45, -0.36), osc_enable=False):
        super().__init__(env)
        self.asym_inject_p = asym_inject_p
        self.asym_inject_d = asym_inject_d
        self._reset_episode_state()
        self._inj_adr = None
        self._inj_dof = None
        self.asym_inject_count = 0     # 注入过的集数
        self.asym_convert_count = 0    # 其中"摔→截断"被转换的次数
        self.asym_inject_len = 0.0     # 注入集的累计长度（求均值用）
        self.asym_inject_len_n = 0
        self.asym_ep_count = 0
        self.initial_y = None
        self.forward_weight = 0.4
        self.forward_gate_ref = 0.15
        self.healthy_weight = 1.0
        self.ctrl_cost_weight = 0.005
        self.fall_penalty_weight = -50.0
        self.phase_weight = 1.0
        self.activity_rom_weight = 0.6
        self.activity_vel_weight = 0.4
        self.activity_diff_penalty_weight = 0.5
        self.support_sym_weight = 2.0
        self.diff_scale = 0.18
        self.alternate_penalty_weight = 2.3
        self.stall_threshold = 4
        self.stall_cap = 4
        self.stall_weight = 0.5
        self.lateral_weight = 0.1
        self.stride_balance_weight = 0.5
        self.swing_drive_weight = swing_drive_weight
        self._foot_geom_ids = None
        self.double_support_frac = 0.0
        self.flight_frac = 0.0
        self.single_frac = 0.0
        self.stance_frac_L = 0.0
        self.stance_frac_R = 0.0
        self.walk_gate = 1.0
        self.walk_gate_lo = -0.20
        self.walk_gate_ref = 0.10
        self.walk_gate_floor = 0.20
        self.speed_gate_lo = 0.10
        self.speed_gate_ref = 0.30
        self.place_weight = 3.0
        self.place_off = 0.06
        self.place_span = 0.26
        self.flat_gate = 1.0
        self.flat_gate_floor = 0.6
        self.slip_weight = 0.3
        self.slip_free_zone = 0.10
        self.slip_cap = 0.7
        self.reach_floor = 0.0
        self.reach_top = 0.20
        self.dense_ds_ref = 0.25
        self.dense_ds_floor = 0.55
        self._dense_ds_factor = 1.0
        self.swap_gate_floor = 0.75
        self.swap_gate_ref = 0.5
        self.swap_gate = 1.0
        self.osc_enable = osc_enable
        # 摆腿时钟模板：摆动窗口由时钟接管两条 hip_y，支撑期交回策略。
        # 驱动定律：T1 = 固定摆动窗口内正弦驱动；T2 = 触地即止。
        self.osc_A = 0.15
        self.osc_swing_frac = 22.0 / 60.0
        self.osc_wmax = 28               # < 半周期 30 ⇒ 加上单驱动仲裁，同一帧不可能两腿都在摆
        self.osc_being_driven = (False, False)        # 本帧左右髋是否正被 T2 驱动（触地覆盖率的分子）
        self.osc_k_start = 1.0           # T1 无课程：k 恒定 1.0
        self.osc_frames = 0
        self.osc_aidx = None             # {left,right}_hip_y 的动作索引，惰性解析
        self.osc_phase = 0.0
        self.osc_active = (0.0, 0.0)     # 本帧左/右髋是否被模板接管（"模板真在生效"的判别量）
        self.osc_k_now = 0.0
        self.osc_k_forced = None
        self.swing_min_weight = 2.0
        self.swing_min_floor_deg = 0.5
        self.swing_min_top_deg = 14.0
        self._ev = {k: {s: deque(maxlen=200) for s in "LR"}
                    for k in ("reach", "lift", "rom", "knee_ext", "step_adv", "slip",
                              "spacing", "spacing_ds", "spacing_air", "hip_at_land")}
        self._swing_st = {s: dict(n=0, hip_min=0.0, hip_max=0.0, knee_min=0.0, lift_max=0.0) for s in "LR"}
        self._stance_st = {s: dict(n=0, slip=0.0) for s in "LR"}
        self._foot_geom_one = None
        self._foot_radius = 0.0
        self._knee_idx = None
        self.episode_log_path = episode_log_path
        self.prev_body_x_per_frame = None


        self.flight_time_penalty_weight = 0.60
        self.tiny_step_weight = 0.30
        self.tiny_step_ref = 0.34
        self.tiny_step_cap = 2.0
        self._ep_start_x = None          # 集均速的锚点（reset 时写入初始 body_x）
        self.leg_activity_buffer_size = 50
        self.left_hip_angle_buffer = deque(maxlen=self.leg_activity_buffer_size)
        self.right_hip_angle_buffer = deque(maxlen=self.leg_activity_buffer_size)
        self.left_leg_vel_buffer = deque(maxlen=self.leg_activity_buffer_size)
        self.right_leg_vel_buffer = deque(maxlen=self.leg_activity_buffer_size)

        self.foot_diff_buffer = deque(maxlen=120)
        self.left_contact_hist = deque(maxlen=200)
        self.right_contact_hist = deque(maxlen=200)

        self.left_stride_history = deque(maxlen=20)
        self.right_stride_history = deque(maxlen=20)

        self.leg_joint_ids = None

        self.prev_body_x = None

        self.position_swap_weight = 1.0
        self.gait_period = 60
        self.contact_phase_weight = 0.3
        self.foot_diff = 0.0
        self.body_advance = 0.0


    def set_osc_k(self, k):
        """课程回调用的 setter；k=None 表示退回本 env 自己的帧数排程。

        必须定义在 __init__ 之外：放进 __init__ 会把初始化后半段吞成方法体，
        导致计数器属性缺失、reset() 第一行就 AttributeError。
        """
        self.osc_k_forced = None if k is None else float(k)

    def compute_position_swap_reward(self, left_x, right_x, body_x, left_contact, right_contact):
        """检测左右脚前后位置交换、奖励左脚领先落地、惩罚连续不换脚"""
        swap_reward = 0.0
        stall_penalty = 0.0
        self.last_landing_swapped = False   # 非落地帧恒 False，供步幅门控读取

        self.foot_diff = abs(left_x - right_x)  # 正数表示左脚在前

        left_just_landed = left_contact and not self.left_contact_prev
        right_just_landed = right_contact and not self.right_contact_prev

        if self.prev_body_x is not None:
            self.body_advance = body_x - self.prev_body_x
        else:
            self.body_advance = 0

        if left_just_landed or right_just_landed:
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


    def _speed_ramp(self, v):
        """height / survival / 碎步税共用的连续速度门：v ≥ speed_gate_ref 时恒 1。

        用连续斜坡代替二值阈值，避免"跌破 0.3 就一次性免掉碎步税"。
        """
        return np.clip((v - self.speed_gate_lo) / (self.speed_gate_ref - self.speed_gate_lo), 0.0, 1.0)

    def compute_hip_phase_reward(self):
        if len(self.left_hip_angle_buffer) < self.leg_activity_buffer_size:
            return 0
        left = np.array(self.left_hip_angle_buffer)
        right = np.array(self.right_hip_angle_buffer)

        left_forward = max(0.0, -np.min(left))
        right_forward = max(0.0, -np.min(right))

        left = left - left.mean()
        right = right - right.mean()
        corr = np.corrcoef(left, right)[0,1]
        anti_phase = (1 - corr) / 2.0

        activity = min(left_forward, right_forward) + 0.1 * min(np.std(left), np.std(right))
        activity = np.clip(activity / 0.5, 0, 1)  # 0.5 rad (~28°) 为典型前摆幅度

        reward = anti_phase * activity
        return reward

    def compute_leg_activity(self,hip_buffer,vel_buffer):
        if len(hip_buffer) < 40:
            return 0.0
        hip_rom = (max(hip_buffer)-min(hip_buffer))
        rom_activity = hip_rom
        vel_activity = np.mean(np.abs(np.array(vel_buffer)))
        rom_activity = np.clip(rom_activity / 3.0,0,1)
        vel_activity = np.clip(vel_activity / 10.0,0,1)
        activity = (self.activity_rom_weight *rom_activity+self.activity_vel_weight *vel_activity)
        return activity

    def compute_alternate_penalty(self, left_x, right_x, left_contact, right_contact):
        diff = (right_x - left_x) / self.diff_scale
        if abs(diff) < 0.01:return 0
        res = 0
        if left_contact and not self.left_contact_prev:
            res = self._check_cycle(diff)
        elif right_contact and not self.right_contact_prev:
            res = self._check_cycle(diff)

        if res < 0:
            self.alternate_reward_count += 1
        elif res > 0:
            self.alternate_penalty_count += 1

        return res

    def _check_cycle(self, current_diff):
        if self.prev_diff is None:
            self.prev_diff = current_diff
            return 0.0
        prev_diff = self.prev_diff
        product = prev_diff * current_diff
        self.prev_diff = current_diff
        return np.sign(product) * min(abs(prev_diff), abs(current_diff))


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

    def _reset_episode_state(self):
        """把每集开始时需要清零的计数器 / 缓存恢复到初始值（__init__ 与 reset 共用）。"""

        self._inj_active = False
        self.asym_inject_steps = 0     # 当前这条注入集已走的步数
        self.prev_diff = None
        self.stride_event_count = 0
        self.stride_bonus_event_count = 0
        self.last_stride_length = 0.0
        self.last_left_stride = None   # 左脚落地时 left_x - right_x，与右脚配对
        self.last_right_stride = None  # 右脚落地时 right_x - left_x
        self.last_contact_phase_reward = 0.0
        self.last_swing_forward_reward = 0.0
        self.last_stride_bonus = 0.0
        self._osc_air = {"left": 0, "right": 0}       # 该脚已连续腾空多少帧
        self.osc_land_total = 0
        self.osc_land_in_drive = 0
        self.osc_cap_cut = 0             # 被 osc_wmax 截断的次数（非零就说明有脚长时间悬空）
        self._osc_trunc = {"left": False, "right": False}
        self.osc_double_air = 0          # 双腾空帧数（这些帧谁都不驱动 ⇒ 无左右偏好）
        self.osc_end_contact = 0         # 被触地正常结束（设计意图）
        self.osc_end_cap = 0             # 被 osc_wmax 截断过（脚悬空超过 28 帧）
        self.left_swing_hip_min = 0.0
        self.left_swing_hip_max = 0.0
        self.right_swing_hip_min = 0.0
        self.right_swing_hip_max = 0.0
        self.last_rom_L_deg = 0.0      # 每条腿"上一次完成的摆动行程°"，落地帧写入
        self.last_rom_R_deg = 0.0
        self.last_reach_L = 0.0
        self.last_reach_R = 0.0
        self._prev_landing_body_x = None
        self.event_stats = {}
        self._stats_dirty = True
        self._ep_ret = 0.0
        self._ep_len = 0
        self.last_drive_L = 0.0
        self.last_drive_R = 0.0
        self.left_swing_flex_total = 0.0
        self.right_swing_flex_total = 0.0
        self.left_swing_drive_time = 0
        self.right_swing_drive_time = 0
        self.short_stride_event_count = 0
        self.negative_stride_event_count = 0
        self.speed = 0.0
        self.left_contact_prev = True
        self._osc_air = {"left": 0, "right": 0}
        self.osc_land_total = self.osc_land_in_drive = self.osc_cap_cut = 0
        self.osc_end_contact = self.osc_end_cap = self.osc_double_air = 0
        self.place_operand_acc = 0.0
        self.place_frames = 0
        self.abd_y_sum = 0.0
        self.abd_y_near = 0
        self._osc_trunc = {"left": False, "right": False}
        self.right_contact_prev = True
        self.total_land_count = 0
        self.position_swap_land_count = 0
        self.alternate_reward_count = 0
        self.alternate_penalty_count = 0
        self.landings_since_swap = 0
        self.tiny_step_frame_count = 0
        self.speed_ramp_frame_sum = 0.0
        self.prev_left_hip_angle = 0.0
        self.prev_right_hip_angle = 0.0
        self.left_contact_count = 0
        self.right_contact_count = 0
        self.total_steps = 0
        self.tiny_step_frame_count = 0
        self.speed_ramp_frame_sum = 0.0
        self.left_swing_steps = 0
        self.right_swing_steps = 0
        self.left_swing_total_steps = 0
        self.right_swing_total_steps = 0
        self.left_swing_history = []
        self.right_swing_history = []
        self.prev_position_state = None
        self.speed = 0.0
        self.curr_position_state = 'centered'
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

    def reset(self, seed=None, options=None):
        """重置环境并清零本集计数器（滑窗、事件统计、模板相位等）。"""
        obs, info = self.env.reset(seed=seed, options=options)

        self.left_hip_angle_buffer.clear()
        self.right_hip_angle_buffer.clear()
        self.left_leg_vel_buffer.clear()
        self.right_leg_vel_buffer.clear()
        self.foot_diff_buffer.clear()
        self.left_contact_hist.clear()
        self.right_contact_hist.clear()
        for sides in self._ev.values():
            for dq in sides.values():
                dq.clear()
        for s in "LR":
            self._swing_st[s] = dict(n=0, hip_min=0.0, hip_max=0.0, knee_min=0.0, lift_max=0.0)
            self._stance_st[s] = dict(n=0, slip=0.0)
        self._reset_episode_state()
        self.left_stride_history.clear()
        self.right_stride_history.clear()

        self.initial_y = self.env.unwrapped.data.qpos[1]
        self.prev_body_x = self.env.unwrapped.data.qpos[0]
        self._ep_start_x = float(self.env.unwrapped.data.qpos[0])
        self.prev_body_x_per_frame = self.env.unwrapped.data.qpos[0]


        if self.asym_inject_p > 0.0:
            rng = getattr(self.env.unwrapped, "np_random", None) or np.random
            if rng.uniform() < self.asym_inject_p:
                u = self.env.unwrapped
                if self._inj_adr is None:
                    jid = u.model.joint("left_hip_y").id
                    self._inj_adr = int(u.model.jnt_qposadr[jid])
                    self._inj_dof = int(u.model.jnt_dofadr[jid])
                lo, hi = self.asym_inject_d
                u.data.qpos[self._inj_adr] += rng.uniform(lo, hi)
                u.data.qvel[self._inj_dof] = 0.0      # 只搬位形、不给速度，避免变成踹一脚
                mujoco.mj_forward(u.model, u.data)
                obs = u._get_obs()                    # 重算观测，策略才真看见这个新状态
                self._inj_active = True
                self.asym_inject_count += 1

        return obs, info

    def _flush_event(self, stat, side, value):
        self._ev[stat][side].append(float(value))
        self._stats_dirty = True

    def _rebuild_event_stats(self):
        """把分腿事件窗口压成标量 gauge（只在有事件刷新时重算）。"""
        out = {}
        for stat, sides in self._ev.items():
            for side, dq in sides.items():
                if not dq:
                    continue
                arr = np.asarray(dq)
                out[f"ev_{stat}_{side}_mean"] = float(arr.mean())
                out[f"ev_{stat}_{side}_n"] = float(arr.size)
                if stat == "reach":
                    out[f"ev_reach_{side}_p50"] = float(np.percentile(arr, 50))
                    out[f"ev_reach_{side}_p90"] = float(np.percentile(arr, 90))
                elif stat == "hip_at_land":
                    out[f"ev_hip_at_land_{side}_p10"] = float(np.percentile(arr, 10))
                    out[f"ev_hip_at_land_{side}_p50"] = float(np.percentile(arr, 50))
                    out[f"ev_hip_at_land_{side}_p90"] = float(np.percentile(arr, 90))
                elif stat.startswith("spacing"):
                    out[f"ev_{stat}_{side}_p10"] = float(np.percentile(arr, 10))
                    out[f"ev_{stat}_{side}_p50"] = float(np.percentile(arr, 50))
                    out[f"ev_{stat}_{side}_p90"] = float(np.percentile(arr, 90))
                    out[f"ev_{stat}_{side}_negfrac"] = float(np.mean(arr < 0))
        self.event_stats = out
        self._stats_dirty = False

    def _update_gait_event_stats(self, body_x, sides):
        """分腿摆动/落地事件统计（纯诊断，不参与奖励）。

        sides 每项: (side, contact, prev_contact, hip角, knee角, 脚x, 同侧髋x, 球底离地, 滑移速度,
                    另一脚x, 另一脚是否支撑)
        reach   = 落地帧 脚x − 同侧髋x      （正=迈到身体前方）
        rom     = 摆动期髋角行程°           lift = 摆动期球底最大离地
        knee_ext= 摆动期最折点到落地的伸膝°  step_adv = 到该次落地的躯干位移（真单步前进）
        slip    = 支撑段接触点切向位移积分   （踝位移不等于滑行：脚是球，能滚）
        spacing = 落地帧 脚x − 另一脚x      （stride 族用的观测量；按另一脚支撑/腾空分组）
        """
        dt = 0.015
        for (side, contact, prev, hip, knee, foot_x, hip_x, clear, slip,
             other_x, other_contact) in sides:
            sw = self._swing_st[side]
            st = self._stance_st[side]
            if contact:
                st["n"] += 1
                st["slip"] += slip * dt
            else:
                if sw["n"] == 0:
                    sw["hip_min"] = sw["hip_max"] = hip
                    sw["knee_min"] = knee
                    sw["lift_max"] = clear
                else:
                    sw["hip_min"] = min(sw["hip_min"], hip)
                    sw["hip_max"] = max(sw["hip_max"], hip)
                    sw["knee_min"] = min(sw["knee_min"], knee)
                    sw["lift_max"] = max(sw["lift_max"], clear)
                sw["n"] += 1
            if contact and not prev and sw["n"] >= 2:            # 真落地帧
                self._flush_event("rom", side, float(np.degrees(sw["hip_max"] - sw["hip_min"])))
                self._flush_event("hip_at_land", side, float(-np.degrees(hip)))
                self._flush_event("knee_ext", side, float(np.degrees(knee - sw["knee_min"])))
                self._flush_event("lift", side, sw["lift_max"])
                self._flush_event("reach", side, foot_x - hip_x)
                spacing = foot_x - other_x
                self._flush_event("spacing", side, spacing)
                self._flush_event("spacing_ds" if other_contact else "spacing_air",
                                  side, spacing)
                if self._prev_landing_body_x is not None:
                    self._flush_event("step_adv", side, body_x - self._prev_landing_body_x)
                self._prev_landing_body_x = body_x
                sw["n"] = 0
            if prev and not contact and st["n"] >= 1:            # 真离地帧
                self._flush_event("slip", side, st["slip"])
                st["n"] = 0
                st["slip"] = 0.0
        if self._stats_dirty:
            self._rebuild_event_stats()

    def _log_episode(self, ep_len, ep_ret):
        """eval 侧每集一行统计（训练 env 不传路径故不写）。单写者、小行、追加，异常时静默降级。"""
        if self._stats_dirty:
            self._rebuild_event_stats()
        rec = {"ep_len": int(ep_len), "ret": float(ep_ret), "speed": float(self.speed),
               "flight_frac": float(self.flight_frac),
               "double_support_frac": float(self.double_support_frac),
               "single_frac": float(self.single_frac),
               "walk_gate": float(self.walk_gate),
               "flat_gate": float(self.flat_gate),
               "dense_ds_factor": float(self._dense_ds_factor),
               "swap_rate": float(self.position_swap_land_count / max(self.total_land_count, 1)),
               "swap_gate": float(self.swap_gate),
               "b17": 1,
               "place_operand_mean": float(self.place_operand_acc / max(self.place_frames, 1)),
               "place_reward": float(self.place_weight * self.place_operand_acc),
               "abd_y_p50_proxy": float(self.abd_y_sum / max(self.place_frames, 1)),
               "abd_y_near_limit_frac": float(self.abd_y_near / max(self.place_frames, 1)),
               "b16": 1,
               "osc_land_cover": float(self.osc_land_in_drive / max(self.osc_land_total, 1)),
               "osc_landings": int(self.osc_land_total),
               "osc_cap_cut": int(self.osc_cap_cut),
               "osc_double_air_frac": float(self.osc_double_air / max(self.total_steps, 1)),
               # osc_end_by = 被触地正常结束的那一摆占比（与 osc_end_cap 互斥，两者相加 == osc_landings）
               "osc_end_by": float(self.osc_end_contact / max(self.osc_land_total, 1)),
               "osc_end_contact": int(self.osc_end_contact),
               "osc_end_cap": int(self.osc_end_cap),
               "b15": 1,
               "speed_ramp_mean": float(self.speed_ramp_frame_sum / max(self.total_steps, 1)),
               "tiny_step_frame_frac": float(self.tiny_step_frame_count / max(self.total_steps, 1)),
               "land_per_s": float(self.total_land_count / max(self.total_steps * 0.015, 1e-6)),
               "ep_landings": int(self.total_land_count),
               "stance_frac_L": float(self.stance_frac_L),
               "stance_frac_R": float(self.stance_frac_R)}
        rec.update(self.event_stats)
        try:
            with open(self.episode_log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"[episode_log] 关闭写入 {self.episode_log_path}: {e}")
            self.episode_log_path = None

    def _foot_contact_state(self, model, data):
        """脚 geom 与地面的真接触标志 + 接触点切向滑移速度（替代旧的 body_z<0.08 判据）。

        旧判据为何错：humanoid.xml 第61-62行，脚是半径 0.075 的球 geom、球心装在 foot body
        原点上方 0.10 → 真正着地时 body_z = 0.075-0.10 = -0.025。所以 `body_z<0.08` 的真实
        含义是"球底离地 < 0.105 m"，实测（944K 模型 det rollout）判为"支撑"的时间里左腿 62%、
        右腿 38% 其实脚在空中，落地结算/摆动累加/支撑对称/腾空全部系统性早报 2-3 步。

        滑移为何不用踝位移算：脚是**球**，球可以纯滚动（球心照样前移），用 foot body x 的位移
        会把滚动记成滑行。这里直接算脚上"接触点那个物质点"的世界速度切向分量：纯滚动时为 0。
        实测旧策略滑移速度中位 左1.5/右1.8 m/s、每支撑段积分 0.14-0.23 m —— 是砸地+打滑，
        不是滚。
        返回 (left_hit, right_hit, slip_left, slip_right)，slip 仅在真接触时非零。
        """
        if self._foot_geom_ids is None:
            ids = []
            for bname in ("left_foot", "right_foot"):
                bid = model.body(bname).id
                ids.append(frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid))
            self._foot_geom_ids = (ids[0], ids[1])
            self._foot_geom_one = (min(ids[0]), min(ids[1]))
            self._foot_radius = float(model.geom(self._foot_geom_one[0]).size[0])
            self._knee_idx = (int(model.jnt_qposadr[model.joint("left_knee").id]),
                              int(model.jnt_qposadr[model.joint("right_knee").id]))
        gl, gr = self._foot_geom_ids
        left_hit = right_hit = False
        slip_l = slip_r = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
            if b1 != 0 and b2 != 0:
                continue                      # 只认与地面（world body 0）的接触
            if g1 in gl or g2 in gl:
                bid, side = model.body("left_foot").id, "L"
                left_hit = True
            elif g1 in gr or g2 in gr:
                bid, side = model.body("right_foot").id, "R"
                right_hit = True
            else:
                continue
            omega = data.cvel[bid][:3]        # MuJoCo cvel: 3 角速度 + 3 线速度，世界系
            v_com = data.cvel[bid][3:6]
            v_point = v_com + np.cross(omega, c.pos - data.xipos[bid])
            tangential = float(np.hypot(v_point[0], v_point[1]))
            if side == "L":
                slip_l = max(slip_l, tangential)
            else:
                slip_r = max(slip_r, tangential)
        return left_hit, right_hit, slip_l, slip_r

    def step(self, action):
        """推进一帧：可选摆腿模板改写动作 → 环境 step → 计算奖励与步态指标。"""
        if self.osc_enable:
            if self.osc_aidx is None:
                m = self.env.unwrapped.model
                self.osc_aidx = {s: [i for i in range(m.nu)
                                     if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                                     == f"{s}_hip_y"][0] for s in ("left", "right")}
            self.osc_frames += 1
            # k 的来源：外部课程（OscCurriculumCallback 按 model.num_timesteps 推过来）优先，
            # 否则用本 env 自己的常量。**训练与 eval 必须吃同一个 k**，否则评的不是部署的控制器。
            k = self.osc_k_start if self.osc_k_forced is None else self.osc_k_forced
            self.osc_k_now = k
            # 时钟仍然算（phase_reward 用它），但 T2 的驱动**不再由它决定窗口**。
            phase = (self.total_steps % self.gait_period) / float(self.gait_period)
            action = np.asarray(action, dtype=np.float64).copy()
            act_flag = []
            for side in ("left", "right"):
                off = 0.0 if side == "left" else 0.5     # 右腿严格晚半周期 ⇒ 交替是结构保证的
                p = (phase - off) % 1.0
                if p < self.osc_swing_frac:
                    u = p / self.osc_swing_frac
                    pulse = -self.osc_A * np.sin(np.pi * u)   # 负=髋前摆（hip_y 负动作把脚送到前面）
                    i = self.osc_aidx[side]
                    action[i] = float(np.clip((1.0 - k) * action[i] + k * pulse, -1.0, 1.0))
                    act_flag.append(1.0)
                else:
                    act_flag.append(0.0)
            self.osc_phase = phase
            self.osc_active = (act_flag[0], act_flag[1])
            _hitL, _hitR, _, _ = self._foot_contact_state(self.env.unwrapped.model,
                                                          self.env.unwrapped.data)
            self.osc_being_driven = (act_flag[0] > 0.0, act_flag[1] > 0.0)
            for _s, _h in (("left", _hitL), ("right", _hitR)):
                self._osc_air[_s] = self._osc_air[_s] + 1 if not _h else 0
                if self._osc_air[_s] == 1:
                    self._osc_trunc[_s] = False
        obs, _, terminated, truncated, info = self.env.step(action)
        mujoco_env = self.env.unwrapped
        data = mujoco_env.data
        x_velocity = data.qvel[0]
        body_x = data.qpos[0]
        advance = body_x - self.prev_body_x_per_frame
        self.prev_body_x_per_frame = body_x
        torso_height = data.qpos[2]   # 使用 qpos
        left_foot = data.body("left_foot")
        right_foot = data.body("right_foot")
        left_x = left_foot.xpos[0]
        right_x = right_foot.xpos[0]
        left_contact, right_contact, slip_l, slip_r = self._foot_contact_state(
            mujoco_env.model, data)
        if self.osc_enable:
            for _s, _hit, _prev, _drv in (("left", left_contact, self.left_contact_prev,
                                           self.osc_being_driven[0]),
                                          ("right", right_contact, self.right_contact_prev,
                                           self.osc_being_driven[1])):
                if _hit and not _prev:
                    self.osc_land_total += 1
                    if _drv:
                        self.osc_land_in_drive += 1
                    # 每一摆按结束方式二选一（互斥 ⇒ 两列相加 == osc_landings）
                    if self._osc_trunc[_s]:
                        self.osc_end_cap += 1
                    else:
                        self.osc_end_contact += 1
                    self._osc_trunc[_s] = False

        def _slip_cost(speed_mps, in_contact):
            if not in_contact:
                return 0.0
            return min(max(speed_mps - self.slip_free_zone, 0.0), self.slip_cap)
        slip_penalty = self.slip_weight * (_slip_cost(slip_l, left_contact)
                                           + _slip_cost(slip_r, right_contact))
        joint_ids=self._get_leg_joint_ids()
        qvel=data.qvel
        left_hip_angle = data.qpos[joint_ids["left_hip_y"]["qpos"]]
        right_hip_angle = data.qpos[joint_ids["right_hip_y"]["qpos"]]
        left_hip_vel = qvel[joint_ids["left_hip_y"]["qvel"]]
        right_hip_vel = qvel[joint_ids["right_hip_y"]["qvel"]]
        self.left_hip_angle_buffer.append(left_hip_angle)
        self.right_hip_angle_buffer.append(right_hip_angle)
        self.left_leg_vel_buffer.append(left_hip_vel)
        self.right_leg_vel_buffer.append(right_hip_vel)

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


        if not left_contact and right_contact:
            self.left_swing_time += 1
            self.left_swing_hip_total += abs(left_hip_angle - self.prev_left_hip_angle)
            self.left_swing_speed_total += left_hip_vel

        if not right_contact and left_contact:
            self.right_swing_time += 1
            self.right_swing_hip_total += abs(right_hip_angle - self.prev_right_hip_angle)
            self.right_swing_speed_total += right_hip_vel

        if not left_contact:
            if self.left_swing_drive_time == 0:      # 摆动起始帧：用当前角初始化行程窗
                self.left_swing_hip_min = self.left_swing_hip_max = left_hip_angle
            self.left_swing_flex_total += max(0.0, self.prev_left_hip_angle - left_hip_angle)
            self.left_swing_drive_time += 1
            self.left_swing_hip_min = min(self.left_swing_hip_min, left_hip_angle)
            self.left_swing_hip_max = max(self.left_swing_hip_max, left_hip_angle)
        if not right_contact:
            if self.right_swing_drive_time == 0:
                self.right_swing_hip_min = self.right_swing_hip_max = right_hip_angle
            self.right_swing_flex_total += max(0.0, self.prev_right_hip_angle - right_hip_angle)
            self.right_swing_drive_time += 1
            self.right_swing_hip_min = min(self.right_swing_hip_min, right_hip_angle)
            self.right_swing_hip_max = max(self.right_swing_hip_max, right_hip_angle)

        if left_contact and not right_contact:
            self.left_stance_time += 1
            self.left_stance_speed_total += left_hip_vel

        if right_contact and not left_contact:
            self.right_stance_time += 1
            self.right_stance_speed_total += right_hip_vel

        self.prev_left_hip_angle = left_hip_angle
        self.prev_right_hip_angle = right_hip_angle

        # 【摆动驱动】本步顶部的摆动窗口驱动（结算前读取，避开落地分支的复位时序）
        dL_now = self.left_swing_flex_total / max(self.left_swing_drive_time, 1)
        dR_now = self.right_swing_flex_total / max(self.right_swing_drive_time, 1)


        self.speed = advance / 0.015
        target_speed = 0.95
        sigma = 0.28
        forward_gauss = self.forward_weight * np.exp(-((self.speed - target_speed) ** 2) / (2 * sigma ** 2))
        if self.last_left_stride is not None and self.last_right_stride is not None:
            self.forward_gate = np.clip(min(self.last_left_stride, self.last_right_stride) / self.forward_gate_ref, 0.0, 1.0)
            forward_gauss = forward_gauss * self.forward_gate
        else:
            self.forward_gate = 1.0
        speed_penalty = 1.0 * max(0, self.speed - 1.6)

        band = min((torso_height - 1.05) / 0.20, (1.55 - torso_height) / 0.10, 1.0)
        height_reward = self.healthy_weight * max(band, 0.0) \
            * self._speed_ramp(self.speed)

        survival_bonus = 0.5 * self._speed_ramp(self.speed)
        # 骑在旧门槛0.5上反复开关导致地板失效、Q塌、螺旋；0.3 让地板在过渡速度稳定打开。站立(0速)仍拿不到。

        ctrl_cost = self.ctrl_cost_weight * np.square(action).sum()

        fall_penalty = self.fall_penalty_weight if terminated else 0.0

        stride_bonus = 0
        reach_reward = 0.0
        swing_min_reward = 0.0
        short_stride_penalty = 0
        negative_stride_penalty = 0
        stride_balance_penalty = 0.0
        STRIDE_WEIGHT = 35.0
        STRIDE_MAX = 0.12
        MIN_SPACING = 0.2
        STRIDE_FLOOR = 0.10
        STRIDE_TOP = 0.20
        STRIDE_CAD_REF = 2.5
        BALANCE_MIN_STRIDE = 0.15   # 较弱一侧的步幅下界：两侧都小步 / 居中不交替都会踩到
        BALANCE_CAP = 0.30
        SHORT_STRIDE_WEIGHT =7.0
        SHORT_STRIDE_SCALE = 0.06
        NEGATIVE_STRIDE_WEIGHT = 3.0
        NEGATIVE_STRIDE_SCALE = 0.08
        SCALE_ANGLE = 0.3  # hip_y 有效活动范围
        swing_forward_reward = 0.0
        contact_phase_reward = 0.0
        _land_L = (not self.left_contact_prev) and left_contact
        _land_R = (not self.right_contact_prev) and right_contact
        self._dense_ds_factor = self.dense_ds_floor + (1.0 - self.dense_ds_floor) * np.clip(
            self.double_support_frac / self.dense_ds_ref, 0.0, 1.0)
        alternate_result = self.compute_alternate_penalty(left_x, right_x, left_contact, right_contact)
        position_swap_reward, stall_penalty = self.compute_position_swap_reward(
            left_x, right_x, body_x, left_contact, right_contact)
        if not self.left_contact_prev and left_contact and right_contact:
            left_stride = left_x - right_x
            self.stride_event_count += 1
            self.last_stride_length = left_stride
            if left_stride > MIN_SPACING:
                pass
            elif left_stride > 0:
                self.short_stride_event_count += 1
            else:
                pass

            if self.left_swing_time > 0:
                swing_forward_reward = np.clip(self.left_swing_hip_total / self.left_swing_time / SCALE_ANGLE, 0, 1) * 1.0
            else:
                swing_forward_reward = 0.0

            if self.left_stance_time > 0 and self.right_swing_time > 0:
                stance_q = 1.0 - np.clip(self.left_stance_speed_total / self.left_stance_time / 1.0, 0, 1)
                swing_q = np.clip(self.right_swing_speed_total / max(self.right_swing_time, 1) / 2.0, 0, 1)
                contact_phase_reward = (stance_q + swing_q) / 2 * self.contact_phase_weight
            else:
                contact_phase_reward = 0.0
            self.last_contact_phase_reward = contact_phase_reward
            self.last_swing_forward_reward = swing_forward_reward

            self.left_swing_hip_total = 0.0
            self.left_swing_time = 0
            self.left_swing_speed_total = 0.0
            self.left_stance_time = 0
            self.left_stance_speed_total = 0.0
            self.right_swing_hip_total = 0.0
            self.right_swing_time = 0
            self.right_swing_speed_total = 0.0
        if not self.right_contact_prev and right_contact and left_contact:
            right_stride = right_x - left_x
            self.stride_event_count += 1
            self.last_stride_length = right_stride
            if right_stride > MIN_SPACING:
                pass
            elif right_stride > 0:
                self.short_stride_event_count += 1
            else:
                pass

            if self.right_swing_time > 0:
                swing_forward_reward = np.clip(self.right_swing_hip_total / self.right_swing_time / SCALE_ANGLE, 0, 1) * 1.0
            else:
                swing_forward_reward = 0.0

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
        any_landing = (not self.left_contact_prev and left_contact) or (not self.right_contact_prev and right_contact)
        if not self.left_contact_prev and left_contact:
            self.last_left_stride = left_x - right_x
            self.last_rom_L_deg = float(np.degrees(self.left_swing_hip_max - self.left_swing_hip_min))
            reach_l = left_x - data.body("left_thigh").xpos[0]
            if self.swing_drive_weight > 0:
                reach_reward += self.swing_drive_weight * np.clip(
                    (reach_l - self.reach_floor) / (self.reach_top - self.reach_floor), 0, 1)
                self.last_reach_L = reach_l
                self.last_drive_L = dL_now
                self.left_swing_flex_total = 0.0
                self.left_swing_drive_time = 0
        if not self.right_contact_prev and right_contact:
            self.last_right_stride = right_x - left_x
            self.last_rom_R_deg = float(np.degrees(self.right_swing_hip_max - self.right_swing_hip_min))
            reach_r = right_x - data.body("right_thigh").xpos[0]
            if self.swing_drive_weight > 0:
                reach_reward += self.swing_drive_weight * np.clip(
                    (reach_r - self.reach_floor) / (self.reach_top - self.reach_floor), 0, 1)
                self.last_reach_R = reach_r
                self.last_drive_R = dR_now
                self.right_swing_flex_total = 0.0
                self.right_swing_drive_time = 0
        _neg_this = None
        if _land_L and self.last_left_stride is not None:
            _neg_this = self.last_left_stride
        if _land_R and self.last_right_stride is not None:
            _neg_this = self.last_right_stride if _neg_this is None else min(_neg_this, self.last_right_stride)
        if _neg_this is not None and _neg_this < 0:
            negative_stride_penalty = np.clip(-_neg_this / NEGATIVE_STRIDE_SCALE, 0, 1) * NEGATIVE_STRIDE_WEIGHT
            self.negative_stride_event_count += 1
        if any_landing:
            swing_min_reward = self.swing_min_weight * np.clip(
                (min(self.last_rom_L_deg, self.last_rom_R_deg) - self.swing_min_floor_deg)
                / (self.swing_min_top_deg - self.swing_min_floor_deg), 0, 1)
        if any_landing and self.last_left_stride is not None and self.last_right_stride is not None:
            lo = min(self.last_left_stride, self.last_right_stride)
            stride_balance_penalty = min(
                self.stride_balance_weight * (abs(self.last_left_stride - self.last_right_stride) + max(0.0, BALANCE_MIN_STRIDE - lo)),
                BALANCE_CAP)
            if self.last_landing_swapped:
                stride_bonus = np.clip((lo - STRIDE_FLOOR) / (STRIDE_TOP - STRIDE_FLOOR), 0, 1) * STRIDE_WEIGHT
                _land_per_s = self.total_land_count / max(self.total_steps * 0.015, 1e-6)
                stride_bonus *= min(1.0, STRIDE_CAD_REF / max(_land_per_s, 1e-6))
                if stride_bonus > 0:
                    self.stride_bonus_event_count += 1
            self.last_stride_bonus = stride_bonus


        PITCH_THRESH      = 0.2   # 向后仰阈值 0.2 rad ≈ 11°
        PITCH_BACK_W      = 3.0
        PITCH_FWD_W       = 2.0
        ROLL_THRESH       = 0.15   # 左右侧倾阈值 0.15 rad ≈ 8-9°
        ROLL_W            = 5.0

        quat = data.qpos[3:7]
        r = Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]])
        roll,pitch,yaw = r.as_euler("xyz")

        balance_reward = 0.1
        pitch_penalty = 0.0
        if pitch < -PITCH_THRESH:
            pitch_penalty = (abs(pitch) - PITCH_THRESH) * PITCH_BACK_W
        elif pitch > PITCH_THRESH:
            pitch_penalty = (pitch - PITCH_THRESH) * PITCH_FWD_W

        roll_penalty = 0.0
        if abs(roll) > ROLL_THRESH:
            roll_penalty = ((abs(roll) - ROLL_THRESH)) * ROLL_W

        balance_penalty = pitch_penalty + roll_penalty
        net_balance = balance_reward - balance_penalty

        # 左右腿角度相位差
        phase_reward = self.compute_hip_phase_reward()*self.phase_weight


        left_leg_activity = self.compute_leg_activity(
            self.left_hip_angle_buffer,self.left_leg_vel_buffer)
        right_leg_activity = self.compute_leg_activity(
            self.right_hip_angle_buffer,self.right_leg_vel_buffer)
        activity_difference = abs(left_leg_activity-right_leg_activity)
        activity_diff_penalty = (self.activity_diff_penalty_weight*(activity_difference)**2) if (not left_contact) or (not right_contact) else 0


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


        self.left_contact_hist.append(float(left_contact))
        self.right_contact_hist.append(float(right_contact))
        if len(self.left_contact_hist) >= 20:
            left_support_ratio = np.mean(self.left_contact_hist)
            right_support_ratio = np.mean(self.right_contact_hist)
            support_sym_penalty = abs(left_support_ratio - right_support_ratio)
            l_h = np.asarray(self.left_contact_hist)
            r_h = np.asarray(self.right_contact_hist)
            self.double_support_frac = float(np.mean((l_h > 0.5) & (r_h > 0.5)))
            self.flight_frac = float(np.mean((l_h < 0.5) & (r_h < 0.5)))
            self.single_frac = float(1.0 - self.flight_frac - self.double_support_frac)
            self.stance_frac_L = float(left_support_ratio)
            self.stance_frac_R = float(right_support_ratio)
            self.walk_gate = self.walk_gate_floor + (1.0 - self.walk_gate_floor) * np.clip(
                ((self.double_support_frac - self.flight_frac) - self.walk_gate_lo)
                / (self.walk_gate_ref - self.walk_gate_lo), 0.0, 1.0)
            self.flat_gate = self.flat_gate_floor + (1.0 - self.flat_gate_floor) * self.walk_gate
        else:
            support_sym_penalty = 0.0
        support_sym_penalty *= self.support_sym_weight
        _swap_rate = self.position_swap_land_count / max(self.total_land_count, 1)
        self.swap_gate = self.swap_gate_floor + (1.0 - self.swap_gate_floor) * np.clip(
            _swap_rate / self.swap_gate_ref, 0.0, 1.0)

        alternate_penalty = alternate_result * self.alternate_penalty_weight


        lateral_penalty = 0
        lateral_pos = abs(data.qpos[1]-self.initial_y)      # 躯干横向位置
        lateral_vel = abs(data.qvel[1])      # 躯干横向速度
        lateral_penalty = self.lateral_weight*(0.5*lateral_pos + 0.2*lateral_vel)

        # ========== 髋关节侧摆角度惩罚 ==========
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

        HIP_Z_ANG_THRESH = 0.16
        HIP_Z_PENALTY_WEIGHT = 0.8
        if abs(left_hip_z_angle) > HIP_Z_ANG_THRESH:
            hip_xz_angle_penalty += (abs(left_hip_z_angle) - HIP_Z_ANG_THRESH) * HIP_Z_PENALTY_WEIGHT
        if abs(right_hip_z_angle) > HIP_Z_ANG_THRESH:
            hip_xz_angle_penalty += (abs(right_hip_z_angle) - HIP_Z_ANG_THRESH) * HIP_Z_PENALTY_WEIGHT


        flight_phase_penalty = self.flight_time_penalty_weight if (not left_contact and not right_contact) else 0.0

        _ts_lo = None
        if self.last_left_stride is not None and self.last_right_stride is not None:
            _ts_lo = min(self.last_left_stride, self.last_right_stride)
        _v_mean = ((body_x - self._ep_start_x) / max(self.total_steps * 0.015, 1e-6)
                   if self._ep_start_x is not None else 0.0)
        if _ts_lo is not None and self.total_land_count > 5:
            short_stride_penalty = self.tiny_step_weight * np.clip(
                (self.tiny_step_ref - _ts_lo) / self.tiny_step_ref, 0.0, self.tiny_step_cap) \
                * self._speed_ramp(_v_mean)
            if short_stride_penalty > 0:
                self.tiny_step_frame_count += 1
        else:
            short_stride_penalty = 0.0

        gl_id, gr_id = self._foot_geom_one
        self._update_gait_event_stats(body_x, [
            ("L", left_contact, self.left_contact_prev, left_hip_angle,
             data.qpos[self._knee_idx[0]], left_x, data.body("left_thigh").xpos[0],
             float(data.geom_xpos[gl_id][2] - self._foot_radius), slip_l, right_x, right_contact),
            ("R", right_contact, self.right_contact_prev, right_hip_angle,
             data.qpos[self._knee_idx[1]], right_x, data.body("right_thigh").xpos[0],
             float(data.geom_xpos[gr_id][2] - self._foot_radius), slip_r, left_x, left_contact),
        ])

        self.left_contact_prev = left_contact
        self.right_contact_prev = right_contact

        place_operand = (
            np.clip((left_x - body_x + self.place_off) / self.place_span, 0.0, 1.0) * float(left_contact)
            + np.clip((right_x - body_x + self.place_off) / self.place_span, 0.0, 1.0) * float(right_contact))
        place_reward = self.place_weight * place_operand
        self.place_operand_acc = getattr(self, "place_operand_acc", 0.0) + place_operand
        self.place_frames = getattr(self, "place_frames", 0) + 1
        if getattr(self, "_abd_adr", None) is None:
            self._abd_adr = mujoco_env.model.jnt_qposadr[mujoco_env.model.joint("abdomen_y").id]
        self.abd_y_sum = getattr(self, "abd_y_sum", 0.0) + float(np.degrees(data.qpos[self._abd_adr]))
        self.abd_y_near = getattr(self, "abd_y_near", 0) + int(abs(abs(float(np.degrees(data.qpos[self._abd_adr]))) - 30.0) < 2.0)

        forward_reward = forward_gauss * self.walk_gate - speed_penalty
        # 奖励求和：推进 + 步态形状（摆动 / 步幅 / 姿态 / 放置）+ 生存 − 税；
        # walk_gate / flat_gate / _dense_ds_factor 分别乘在对应项上。
        reward = (forward_reward +
                swing_forward_reward +
                reach_reward * self.swap_gate +
                swing_min_reward * self.walk_gate +
                stride_bonus +
                net_balance +
                height_reward * self.flat_gate * self._dense_ds_factor +
                fall_penalty +
                place_reward +
                survival_bonus * self.flat_gate * self._dense_ds_factor +
                position_swap_reward * self.walk_gate * self._dense_ds_factor +
                contact_phase_reward +
                - activity_diff_penalty
                - support_sym_penalty
                - alternate_penalty * (1.0 if alternate_penalty > 0 else self.walk_gate)
                - lateral_penalty
                - hip_xz_angle_penalty
                - flight_phase_penalty
                - ctrl_cost
                - short_stride_penalty
                - negative_stride_penalty
                - stride_balance_penalty
                - slip_penalty)


        info["phase_reward"] = phase_reward
        info["ctrl_cost"] = ctrl_cost
        info["activity_difference"] = activity_difference
        info["activity_diff_paid"] = self.activity_diff_penalty_weight * activity_difference ** 2
        info["fall_penalty"] = fall_penalty
        info["support_sym_penalty"] = support_sym_penalty
        info["slip_penalty"] = slip_penalty
        info["slip_speed_L"] = slip_l if left_contact else 0.0
        info["slip_speed_R"] = slip_r if right_contact else 0.0
        info["double_support_frac"] = self.double_support_frac
        info["flight_frac"] = self.flight_frac
        info["stance_frac_L"] = self.stance_frac_L
        info["stance_frac_R"] = self.stance_frac_R
        info["lateral_penalty"] = lateral_penalty
        info["position_swap_reward"] = position_swap_reward
        info["contact_phase_reward"] = contact_phase_reward

        info["stride_bonus"] = stride_bonus
        info["stride_event_count"] = self.stride_event_count
        info["stride_bonus_event_count"] = self.stride_bonus_event_count
        info["stride_length"] = self.last_stride_length
        _n_land = max(self.total_land_count, 1)
        info["ds_landing_rate"] = self.stride_event_count / _n_land
        info["stride_bonus_rate"] = self.stride_bonus_event_count / _n_land

        info["short_stride_penalty"] = short_stride_penalty
        info["lo_defined"] = float(self.last_left_stride is not None
                                   and self.last_right_stride is not None)
        info["lo_held"] = (float(min(self.last_left_stride, self.last_right_stride))
                           if info["lo_defined"] else float("nan"))
        info["land_count"] = self.total_land_count
        info["ep_frames"] = self.total_steps
        info["short_stride_event_rate"] = self.short_stride_event_count / _n_land
        info["tiny_step_frame_frac"] = self.tiny_step_frame_count / max(self.total_steps, 1)

        info["negative_stride_penalty"] = negative_stride_penalty
        info["negative_stride_event_count"] = self.negative_stride_event_count
        info["negative_stride_rate"] = self.negative_stride_event_count / _n_land

        info["stride_balance_penalty"] = stride_balance_penalty
        info["last_left_stride"] = self.last_left_stride if self.last_left_stride is not None else 0.0
        info["last_right_stride"] = self.last_right_stride if self.last_right_stride is not None else 0.0

        info["hip_xz_angle_penalty"] = hip_xz_angle_penalty
        info["alternate_penalty"] = alternate_penalty
        info["forward_reward"] = forward_reward
        info["forward_gauss"] = forward_gauss
        info["speed_penalty"] = speed_penalty
        info["walk_gate"] = self.walk_gate
        info["dense_ds_factor"] = self._dense_ds_factor
        info["survival_bonus"] = survival_bonus
        info["single_frac"] = self.single_frac
        info["forward_gate"] = self.forward_gate
        info["height_reward"] = height_reward
        info["flat_gate"] = self.flat_gate
        info["speed_ramp"] = self._speed_ramp(self.speed)
        self.speed_ramp_frame_sum += float(info["speed_ramp"])
        info["swap_gate"] = self.swap_gate
        info["phase_reward_unpaid"] = phase_reward * self.swap_gate
        info["place_reward"] = place_reward
        info["place_operand"] = place_operand
        info["alternate_paid"] = alternate_penalty * (1.0 if alternate_penalty > 0 else self.walk_gate)
        info["net_balance"] = net_balance
        info["pitch_angle"] = pitch
        info["roll_angle"] = roll
        info["swing_forward_reward"] = swing_forward_reward
        info["reach_reward"] = reach_reward
        info["swing_min_reward"] = swing_min_reward
        info["swing_min_reward_gated"] = swing_min_reward * self.walk_gate
        info["last_rom_L_deg"] = self.last_rom_L_deg
        info["last_rom_R_deg"] = self.last_rom_R_deg
        info["last_reach_L"] = self.last_reach_L
        info["last_reach_R"] = self.last_reach_R
        info.update(self.event_stats)
        info["speed"] = self.speed
        info["position_swap_rate"] = self.position_swap_land_count / self.total_land_count if self.total_land_count > 0 else 0
        info["alternate_reward_rate"] = self.alternate_reward_count / self.total_land_count if self.total_land_count > 0 else 0
        info["alternate_penalty_rate"] = self.alternate_penalty_count / self.total_land_count if self.total_land_count > 0 else 0
        info["stall_penalty"] = stall_penalty
        info["flight_phase_penalty"] = flight_phase_penalty

        if self._inj_active:
            self.asym_inject_steps += 1
            if terminated:
                terminated = False
                truncated = True
                self.asym_convert_count += 1
            if terminated or truncated:
                self.asym_inject_len += self.asym_inject_steps
                self.asym_inject_len_n += 1
                self.asym_inject_steps = 0
                self._inj_active = False
        if terminated or truncated:
            self.asym_ep_count += 1
        if self.asym_inject_p > 0.0:
            info["asym_injected"] = float(self._inj_active)
            info["asym_inject_rate"] = self.asym_inject_count / max(1, self.asym_ep_count)
            info["asym_convert_rate"] = self.asym_convert_count / max(1, self.asym_inject_count)
        info["osc_phase"] = self.osc_phase if self.osc_enable else 0.0
        info["osc_active_L"] = self.osc_active[0] if self.osc_enable else 0.0
        info["osc_land_cover"] = (self.osc_land_in_drive / max(self.osc_land_total, 1)
                                  if self.osc_enable else 0.0)
        info["osc_cap_cut"] = float(self.osc_cap_cut) if self.osc_enable else 0.0
        info["osc_double_air"] = float(self.osc_double_air) if self.osc_enable else 0.0
        info["osc_air_L"], info["osc_air_R"] = (float(self._osc_air["left"]), float(self._osc_air["right"])) \
            if self.osc_enable else (0.0, 0.0)
        info["osc_active_R"] = self.osc_active[1] if self.osc_enable else 0.0
        info["osc_active"] = info["osc_active_L"]

        self._ep_ret += float(reward)
        self._ep_len += 1

        if terminated or truncated:
            mean_left_swing = np.mean(self.left_swing_history) if self.left_swing_history else 0
            mean_right_swing = np.mean(self.right_swing_history) if self.right_swing_history else 0

            if self.episode_log_path:
                self._log_episode(self._ep_len, self._ep_ret)
                self._ep_ret = 0.0
                self._ep_len = 0


        return obs, reward, terminated, truncated, info


class MirrorReplayer:
    """在 worker 专用环境中整集镜像重放。产出 (L,·) 数组批，动作已转 buffer 域 [-1,1]。"""

    def __init__(self, env, mirror_map):
        self.env = env
        self.mm = mirror_map
        self.scale = float(env.action_space.high[0])   # env域(±0.4)→buffer域(±1)

    def replay(self, q0, v0, clock, actions, return_infos=False):
        w, mj = self.env, self.env.unwrapped
        w.reset()
        q_m, v_m = self.mm.mirror_qpos_qvel(q0, v0)
        mj.set_state(q_m, v_m)         # 覆盖物理态（内部含 mj_forward）
        w.initial_y = float(mj.data.qpos[1])
        w.prev_body_x = w.prev_body_x_per_frame = float(mj.data.qpos[0])
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
    """带等变正则的 SAC 子类：在 actor loss 里混入镜像等变项。

    等变监督 ||π(mirror_obs(s)) − mirror_action(a_real)||²，按动作幅度归一化后与 RL 项混合：
        actor_loss = (1−λ)·RL + λ·eq_norm,  eq_norm = eq_MSE / detach(mean(||mirror(a)||²))
    归一化让 λ 直接控制约束力，且与动作幅度无关。SB3 本体零修改；保存时排除镜像 /
    正则相关属性，SAC.load 可直接读。当前 λ=0（整体关闭）。
    """

    def __init__(self, *args, mirror_map=None, eq_mix_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
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
        return super()._excluded_save_params() + ["eq_mix_lambda",
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

            # ===== 等变正则（本轮 λ=0，整体关闭）=====
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


class OscCurriculum(BaseCallback):
    """把同一个模板系数 k 按全局 env 步数同时推给训练 env 和评测 env。

    评测必须吃部署时的那套控制器，否则两边量的是两码事。k0=k1=1.0 时退化为恒等同步器
    （当前配置），不再爬升。
    """

    def __init__(self, eval_vec_env, hold=200_000, ramp=400_000, k0=1.0, k1=1.0, verbose=0):
        super().__init__(verbose)
        self.eval_vec_env = eval_vec_env
        self.hold, self.ramp, self.k0, self.k1 = hold, ramp, k0, k1

    def _on_step(self):
        if self.n_calls % 512 == 1:          # ≈4096 env 步推一次，IPC 可忽略
            prog = 0.0 if self.num_timesteps <= self.hold else \
                min((self.num_timesteps - self.hold) / max(self.ramp, 1), 1.0)
            k = self.k0 + (self.k1 - self.k0) * prog
            self.model.env.env_method("set_osc_k", k)
            try:
                self.eval_vec_env.env_method("set_osc_k", k)
            except Exception as e:           # 评测 env 推不到 ⇒ k 会退回 env 自身排程，必须吵
                print(f"[osc] 评测 env 的 k 推送失败（退用 env 自身排程）: {e}")
        return True


def make_env(env_id, rank, seed=0, patch=False, with_mirror=False, swing_drive_weight=0.0,
             episode_log_path=None, asym_inject_p=0.0, osc_enable=False):
    def _init():
        env = gym.make(env_id, max_episode_steps=1500)
        if patch:
            patch_mirror_symmetry(env.unwrapped.model)   # 左右对称修补（左膝弹簧/左髋转子）
        env = WalkRewardWrapper(env, swing_drive_weight=swing_drive_weight,
                                episode_log_path=episode_log_path,
                                asym_inject_p=asym_inject_p, osc_enable=osc_enable)
        if with_mirror:
            r_env = gym.make(env_id, max_episode_steps=1500)
            patch_mirror_symmetry(r_env.unwrapped.model)
            r_env = WalkRewardWrapper(r_env, swing_drive_weight=swing_drive_weight)
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
    ENABLE_MIRROR_AUG = True
    EQ_MIX_LAMBDA = 0.0
    SWING_DRIVE_WEIGHT = 2.0

    ASYM_INJECT_P = 0.0
    OSC_ENABLE = True
    FROM_SCRATCH = False
    # ⚠ 镜像整集重放这一轮**关闭**：MirrorReplayer 重放的是录制的 a_net 序列，而振荡器注入在 env 内部，
    #   镜像集会在**同一侧**再注入一次 ⇒ 左右错配的数据。`patch` 仍然保留（那是动力学对称修补，必须有）。
    WITH_MIRROR_REPLAY = False
    train_vec_env = SubprocVecEnv(
        [make_env(env_id, i, seed, patch=ENABLE_MIRROR_AUG, with_mirror=WITH_MIRROR_REPLAY,
                  swing_drive_weight=SWING_DRIVE_WEIGHT, asym_inject_p=ASYM_INJECT_P,
                  osc_enable=OSC_ENABLE)
         for i in range(n_train_envs)]
    )
    train_vec_env = VecMonitor(train_vec_env)

    os.makedirs("./eval_logs", exist_ok=True)
    eval_vec_env = SubprocVecEnv(
        [make_env(env_id, n_train_envs + i, seed, patch=ENABLE_MIRROR_AUG,
                  swing_drive_weight=SWING_DRIVE_WEIGHT,
                  osc_enable=OSC_ENABLE,          # 评测评的是"带振荡器的控制器"，必须与训练一致
                  episode_log_path="./eval_logs/eval_gait.jsonl")
         for i in range(n_eval_envs)]
    )
    eval_vec_env = VecMonitor(eval_vec_env)

    policy_kwargs = {
        'net_arch': [512, 256],
        'activation_fn': nn.Tanh
    }

    # ============ SAC 模型（等变性正则子类） ============
    _mm_env = gym.make(env_id, max_episode_steps=1500)
    if ENABLE_MIRROR_AUG:
        patch_mirror_symmetry(_mm_env.unwrapped.model)
    mirror_map_main = MirrorMap(_mm_env.unwrapped.model)
    _mm_env.close()

    model = MirrorEqSAC(
        'MlpPolicy',
        train_vec_env,
        learning_rate=lambda prog: 2e-4 * prog + 1e-5 * (1 - prog),  # 线性退火：2e-4 -> 1e-5
        buffer_size=1_500_000,
        batch_size=1024,
        tau=0.01,
        gamma=0.99,
        train_freq=512,
        gradient_steps=512,
        ent_coef=0.10,
        target_entropy=-8.5,      # 动作维度17
        policy_kwargs=policy_kwargs,
        seed=seed,                  # 固定种子：同一 seed 结果可复现
        verbose=1,
        tensorboard_log="./sac_humanoid_walk/",
        mirror_map=mirror_map_main if ENABLE_MIRROR_AUG else None,
        eq_mix_lambda=EQ_MIX_LAMBDA if ENABLE_MIRROR_AUG else 0.0,
    )

    if FROM_SCRATCH:
        print("⚠ 跳过 warm-start 权重加载，策略随机初始化")
    else:
        old_model = SAC.load("./best_model/best_model.zip", env=train_vec_env)
        model.policy.load_state_dict(old_model.policy.state_dict())
        # old_model 只用来取 policy 权重，但 SAC.load 会连同它自己的 9.14GB replay_buffer 一起构造；
        # 不删的话这个 buffer 活到训练结束，和 model 的叠加成 18GB（机器只有 15GB，可用 2GB）
        del old_model
        gc.collect()
        model.replay_buffer.reset()

    mirror_injector = MirrorInjector(n_envs=n_train_envs)
    n_prefill = 0 if FROM_SCRATCH else 200_000
    if n_prefill > 0:
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
        n_eval_episodes=5,
        render=False
    )

    model.learning_starts = 100_000 if FROM_SCRATCH else 0
    if FROM_SCRATCH:
        print(f"⚠ 随机预热 {model.learning_starts} 环境步（无权重先验、无策略预填充）")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[leg_callback, eval_callback, MirrorAugCallback(mirror_injector),
                  OscCurriculum(eval_vec_env)]
    )
    model.save("sac_humanoid_walk_final")

    print("训练完成！模型已保存。")
