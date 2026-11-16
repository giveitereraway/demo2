import torch
import numpy as np
from gymnasium import spaces
from typing import Literal
from types import SimpleNamespace
from .task_base import BaseTask
from ..core.simulatior import AircraftSimulator
from ..core.catalog import Catalog as c
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward
from ..utils.utils import get_AO_TA_R, get2d_AO_TA_R, in_range_rad, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor
from algorithms.ppo.ppo_actor import PPOActor


def _get_enemy_radar_track(env, agent_id):
    """查询环境中针对该智能体的退化敌机雷达航迹覆盖；干净环境返回 None。"""
    overrides = getattr(env, "_radar_track_overrides", None)
    if overrides and agent_id in overrides:
        return overrides[agent_id]
    getter = getattr(env, "get_enemy_radar_track", None)
    if callable(getter):
        return getter(agent_id)
    return None



class SingleCombatTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        self.use_artillery = getattr(self.config, 'use_artillery', False)
        if self.use_baseline:
            self.baseline_agent = self.load_agent(self.config.baseline_type)

        self.reward_functions = [
            AltitudeReward(self.config), # 高度奖励
            PostureReward(self.config), # 姿态奖励
            EventDrivenReward(self.config) # 事件奖励
        ]

        self.termination_conditions = [
            LowAltitude(self.config), # 高度过低，终止
            ExtremeState(self.config), # 极端状态，终止
            Overload(self.config), # 加速过快，终止
            SafeReturn(self.config), # 被击毁或把所有敌机击毁，终止
            Timeout(self.config), # 到了max_step，终止
        ]

    @property
    def num_agents(self) -> int:
        return 2 if not self.use_baseline else 1

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °) 经度
            c.position_lat_geod_deg,            # 1. latitude   (unit: °) 纬度
            c.position_h_sl_m,                  # 2. altitude   (unit: m) 高度
            c.attitude_roll_rad,                # 3. roll       (unit: rad) 横滚角度
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad) 俯仰角度
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad) 航向角度
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s) 北向速度
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s) 东向速度
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s) 向下速度
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s) 机体坐标系X轴速度
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s) 机体坐标系Y轴速度
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s) 机体坐标系Z轴速度
            c.velocities_vc_mps,                # 12. vc        (unit: m/s) 校正空速
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G) 飞行员X轴过载
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G) 飞行员Y轴过载
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G) 飞行员Z轴过载
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        """
        norm_obs = np.zeros(15)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # DEBUG: 检查原始状态是否有 NaN
        if np.isnan(ego_obs_list).any() or np.isnan(enm_obs_list).any():
            print(f"[NaN Warning] agent_id={agent_id}, step={env.current_step}")
            print(f"  ego_obs_list NaN count: {np.isnan(ego_obs_list).sum()}")
            print(f"  enm_obs_list NaN count: {np.isnan(enm_obs_list).sum()}")
            print(f"  ego_obs_list: {ego_obs_list}")
            print(f"  enm_obs_list: {enm_obs_list}")
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        radar_track = _get_enemy_radar_track(env, agent_id)
        if radar_track is not None:
            enm_cur_ned = np.array(radar_track[:3])
            enm_obs_list = enm_obs_list.copy()
            enm_obs_list[2] = radar_track[3]
            enm_obs_list[6:9] = radar_track[4:7]
            enm_obs_list[9] = radar_track[7]
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        enm_v_body_x = enm_obs_list[9]
        enm_alt_m = enm_obs_list[2]
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_obs_list[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_obs_list[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_obs_list[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_obs_list[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_obs_list[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_obs_list[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_obs_list[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_obs_list[12] / 340            # 8. ego vc   (unit: mh)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_v_body_x - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_alt_m - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20  - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4
            return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._agent_die_flag = {}
        if self.use_baseline:
            self.baseline_agent.reset()
        return super().reset(env)

    def step(self, env):
        def _orientation_fn(AO):
            if AO >= 0 and AO <= 0.5236:  # [0, pi/6]
                return 1 - AO / 0.5236
            elif AO >= -0.5236 and AO <= 0: # [-pi/6, 0]
                return 1 + AO / 0.5236
            return 0
        def _distance_fn(R):
            if R <=1: # [0, 1km]
                return 1
            elif R > 1 and R <= 3: # [1km, 3km]
                return (3 - R) / 2.
            else:
                return 0
        if self.use_artillery:
            for agent_id in env.agents.keys():
                ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                        env.agents[agent_id].get_velocity()])
                for enm in env.agents[agent_id].enemies:
                    if enm.is_alive:
                        enm_feature = np.hstack([enm.get_position(),
                                                enm.get_velocity()])
                        AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
                        enm.bloods -= _orientation_fn(AO) * _distance_fn(R/1000)
                        # if agent_id == 'A0100' and enm.uid == 'B0100':
                        #     print(f"AO: {AO * 180 / np.pi}, {_orientation_fn(AO)}, dis:{R/1000}, {_distance_fn(R/1000)}")

    def get_reward(self, env, agent_id, info=...):
        if self._agent_die_flag.get(agent_id, False):
            return 0.0, info
        else:
            self._agent_die_flag[agent_id] = not env.agents[agent_id].is_alive
            return super().get_reward(env, agent_id, info=info)

    def load_agent(self, name):
        if name == 'pursue':
            return PursueAgent()
        elif name == 'maneuver':
            return ManeuverAgent(maneuver='n')
        elif name == 'dodge':
            return DodgeMissileAgent()
        elif name == 'straight':
            return StraightFlyAgent()
        else:
            raise NotImplementedError

class HierarchicalSingleCombatTask(SingleCombatTask):

    def __init__(self, config: str):
        super().__init__(config)
        # 使用 heading 任务训练好的 PPOActor 作为低层策略
        heading_args = SimpleNamespace()
        heading_args.gain = 0.01
        heading_args.hidden_size = '128 128'
        heading_args.act_hidden_size = '128 128'
        heading_args.activation_id = 1
        heading_args.use_feature_normalization = False
        heading_args.use_recurrent_policy = True
        heading_args.recurrent_hidden_size = 128
        heading_args.recurrent_hidden_layers = 1
        heading_args.use_prior = False

        obs_space = spaces.Box(low=-10, high=10., shape=(12,))
        act_space = spaces.MultiDiscrete([41, 41, 41, 30])
        # 低层控制器会在每个 rollout 子进程中各加载一份，固定放在 CPU 可避免多进程抢占显存。
        device = torch.device("cpu")

        self.lowlevel_policy = PPOActor(heading_args, obs_space, act_space, device=device)
        self.lowlevel_policy.load_state_dict(
            torch.load(get_root_dir() + '/model/actor_heading.pt', map_location=device)
        )
        self.lowlevel_policy.eval()
        self.lowlevel_policy.requires_grad_(False)
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def _normalize_lowlevel_action(self, action):
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return norm_act

    def _run_lowlevel_policy(self, env, agent_id, delta_control):
        if self.use_baseline and agent_id in env.enm_ids:
            return self.baseline_agent.get_action(env.agents[agent_id])

        # 将高层指令和本机状态拼成低层航向控制器需要的 12 维输入。
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12, dtype=np.float32)
        input_obs[0:3] = np.asarray(delta_control, dtype=np.float32)
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)

        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            _action, _, _rnn_states = self.lowlevel_policy(
                input_obs,
                self._inner_rnn_states[agent_id],
                masks,
                deterministic=True,
            )
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        return self._normalize_lowlevel_action(action)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        action = np.asarray(action, dtype=np.int32).reshape(-1)
        delta_control = np.array([
            self.norm_delta_altitude[action[0]],
            self.norm_delta_heading[action[1]],
            self.norm_delta_velocity[action[2]],
        ], dtype=np.float32)
        return self._run_lowlevel_policy(env, agent_id, delta_control)

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)


class TacticalHierarchicalSingleCombatTask(HierarchicalSingleCombatTask):
    PURE_PURSUIT = 0
    LEAD_PURSUIT = 1
    LAG_PURSUIT = 2
    DISENGAGE = 3
    CLIMB_POSITION = 4
    DIVE_ACCELERATE = 5
    LEVEL_ACCELERATE = 6
    LEVEL_DECELERATE = 7
    DEFENSIVE_TURN_LEFT = 8
    DEFENSIVE_TURN_RIGHT = 9
    HIGH_YOYO = 10
    LOW_YOYO = 11

    def load_action_space(self):
        self.action_space = spaces.Discrete(12)

    def _safe_unit_vector(self, vector, fallback=None):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            return vector / norm
        if fallback is None:
            return np.zeros_like(vector, dtype=np.float32)
        fallback = np.asarray(fallback, dtype=np.float32)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm > 1e-6:
            return fallback / fallback_norm
        return np.zeros_like(vector, dtype=np.float32)

    def _ego_heading_vector(self, env, agent_id):
        ego = env.agents[agent_id]
        ego_xy = np.asarray(ego.get_velocity()[:2], dtype=np.float32)
        if np.linalg.norm(ego_xy) > 1e-6:
            return self._safe_unit_vector(ego_xy)

        # 低速或速度向量异常时，回退到 JSBSim 当前航向，避免战术动作方向抖动。
        heading = ego.get_property_value(c.attitude_heading_true_rad)
        return np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)

    def _rotate_vector(self, vector, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        return np.array([
            vector[0] * cos_angle - vector[1] * sin_angle,
            vector[0] * sin_angle + vector[1] * cos_angle,
        ], dtype=np.float32)

    def _combat_geometry(self, env, agent_id):
        ego = env.agents[agent_id]
        enm = ego.enemies[0]
        ego_pos = np.asarray(ego.get_position(), dtype=np.float32)
        ego_vel = np.asarray(ego.get_velocity(), dtype=np.float32)
        radar_track = _get_enemy_radar_track(env, agent_id)
        if radar_track is not None:
            enm_pos = np.asarray(radar_track[:3], dtype=np.float32)
            enm_vel = np.array([radar_track[4], radar_track[5], -radar_track[6]], dtype=np.float32)
        else:
            enm_pos = np.asarray(enm.get_position(), dtype=np.float32)
            enm_vel = np.asarray(enm.get_velocity(), dtype=np.float32)

        relative = enm_pos - ego_pos
        relative_xy = relative[:2]
        distance_xy = np.linalg.norm(relative_xy)
        if distance_xy < 1e-6:
            relative_xy = self._ego_heading_vector(env, agent_id)
            distance_xy = 0.0

        return {
            "ego": ego,
            "enm": enm,
            "ego_vel_xy": ego_vel[:2],
            "enm_vel_xy": enm_vel[:2],
            "relative_xy": relative_xy.astype(np.float32),
            "distance_xy": float(distance_xy),
            "ego_heading_xy": self._ego_heading_vector(env, agent_id),
            "enm_heading_xy": self._safe_unit_vector(enm_vel[:2], fallback=relative_xy),
        }

    def _heading_error_to_vector(self, env, agent_id, target_xy):
        target_xy = np.asarray(target_xy, dtype=np.float32)

        if np.linalg.norm(target_xy) < 1e-6:
            return 0.0

        ego_xy = self._ego_heading_vector(env, agent_id)
        target_xy = self._safe_unit_vector(target_xy, fallback=ego_xy)
        dot = np.clip(np.dot(ego_xy, target_xy), -1.0, 1.0)
        # 二维向量叉积只需要 z 分量，用来判断目标在当前速度方向的左/右侧。
        cross_z = ego_xy[0] * target_xy[1] - ego_xy[1] * target_xy[0]
        heading_error = np.arctan2(cross_z, dot)
        return np.clip(heading_error, -np.pi / 6, np.pi / 6)

    def _quantize_heading_error(self, heading_error):
        # 航向指令量化回原分层任务的 5 个档位，减少低层控制器的输入分布偏移。
        index = np.argmin(np.abs(self.norm_delta_heading - heading_error))
        return float(self.norm_delta_heading[index])

    def _altitude_step_to_enemy(self, env, agent_id):
        ego = env.agents[agent_id]
        enm = ego.enemies[0]
        radar_track = _get_enemy_radar_track(env, agent_id)
        enm_alt = float(radar_track[3]) if radar_track is not None else enm.get_property_value(c.position_h_sl_m)
        delta_altitude = enm_alt - ego.get_property_value(c.position_h_sl_m)
        if delta_altitude > 250:
            return 0.1
        if delta_altitude < -250:
            return -0.1
        return 0.0

    def _lead_target_vector(self, geometry):
        # 提前量追击：按敌机短时速度外推目标点，鼓励抢占敌机前方。
        ego_speed = np.linalg.norm(geometry["ego_vel_xy"])
        lead_time = np.clip(geometry["distance_xy"] / (ego_speed + 1e-6), 1.0, 3.0)
        return geometry["relative_xy"] + geometry["enm_vel_xy"] * lead_time

    def _lag_target_vector(self, geometry):
        # 滞后追击：瞄准敌机后方一点，减少过冲并更容易进入尾随位置。
        lag_distance = np.clip(geometry["distance_xy"] * 0.35, 300.0, 900.0)
        return geometry["relative_xy"] - geometry["enm_heading_xy"] * lag_distance

    def _blend_with_enemy_vector(self, geometry, enemy_weight=0.7):
        enemy_vector = self._safe_unit_vector(geometry["relative_xy"], fallback=geometry["ego_heading_xy"])
        ego_vector = geometry["ego_heading_xy"]
        return enemy_vector * enemy_weight + ego_vector * (1.0 - enemy_weight)

    def _apply_tactical_safety(self, env, agent_id, action_id, delta_altitude, delta_velocity, distance_xy):
        ego = env.agents[agent_id]
        altitude = ego.get_property_value(c.position_h_sl_m)
        altitude_limit = getattr(self.config, "altitude_limit", 2500)

        # 接近低高度终止线时禁止继续俯冲，极低时主动给爬升指令。
        if delta_altitude < 0.0 and altitude <= altitude_limit + 1000:
            delta_altitude = 0.1 if altitude <= altitude_limit + 500 else 0.0

        # 距离过近时，追击类动作不再强制加速，避免贴近后姿态奖励恶化。
        close_distance = min(getattr(self.config, "PostureReward_target_dist", 3.0) * 1000 * 0.35, 1200.0)
        if action_id in (self.PURE_PURSUIT, self.LEAD_PURSUIT) and distance_xy < close_distance:
            delta_velocity = 0.0

        return delta_altitude, delta_velocity

    def _tactical_action_to_delta_control(self, env, agent_id, action_id):
        geometry = self._combat_geometry(env, agent_id)
        relative_xy = geometry["relative_xy"]
        ego_heading_xy = geometry["ego_heading_xy"]
        distance_xy = geometry["distance_xy"]

        delta_altitude = 0.0
        delta_velocity = 0.0

        if action_id == self.PURE_PURSUIT:
            # 纯追击：直接指向敌机当前位置。
            target_xy = relative_xy
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = 0.05
        elif action_id == self.LEAD_PURSUIT:
            # 提前量追击：瞄准敌机短时预测位置。
            target_xy = self._lead_target_vector(geometry)
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = 0.05 if distance_xy > 2000.0 else 0.0
        elif action_id == self.LAG_PURSUIT:
            # 滞后追击：瞄准敌机后方，适合距离较近时尾随占位。
            target_xy = self._lag_target_vector(geometry)
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = -0.05 if distance_xy < 1500.0 else 0.0
        elif action_id == self.DISENGAGE:
            # 脱离：背离敌机并加速拉开空间。
            target_xy = -relative_xy
            delta_velocity = 0.05
        elif action_id == self.CLIMB_POSITION:
            # 爬升占位：向敌机方向保持压力，同时换取高度优势。
            target_xy = relative_xy
            delta_altitude = 0.1
        elif action_id == self.DIVE_ACCELERATE:
            # 俯冲加速：向敌机方向压低高度并恢复能量。
            target_xy = relative_xy
            delta_altitude = -0.1
            delta_velocity = 0.05
        elif action_id == self.LEVEL_ACCELERATE:
            # 平飞加速：保持当前航向恢复速度。
            target_xy = ego_heading_xy
            delta_velocity = 0.05
        elif action_id == self.LEVEL_DECELERATE:
            # 平飞减速：保持当前航向并降低速度，缓解近距离过冲。
            target_xy = ego_heading_xy
            delta_velocity = -0.05
        elif action_id == self.DEFENSIVE_TURN_LEFT:
            # 左防御转弯：基于本机当前航向左转，而不是围绕敌机相对位置旋转。
            target_xy = self._rotate_vector(ego_heading_xy, np.pi / 2)
        elif action_id == self.DEFENSIVE_TURN_RIGHT:
            # 右防御转弯：基于本机当前航向右转。
            target_xy = self._rotate_vector(ego_heading_xy, -np.pi / 2)
        elif action_id == self.HIGH_YOYO:
            # 高悠悠：朝敌机方向偏转、爬升、减速，换取高度和角度优势。
            target_xy = self._blend_with_enemy_vector(geometry, enemy_weight=0.7)
            delta_altitude = 0.1
            delta_velocity = -0.05
        elif action_id == self.LOW_YOYO:
            # 低悠悠：朝敌机方向偏转、俯冲、加速，先换速度再转入进攻。
            target_xy = self._blend_with_enemy_vector(geometry, enemy_weight=0.7)
            delta_altitude = -0.1
            delta_velocity = 0.05
        else:
            raise ValueError(f"Unknown tactical action id: {action_id}")

        delta_altitude, delta_velocity = self._apply_tactical_safety(
            env, agent_id, action_id, delta_altitude, delta_velocity, distance_xy
        )
        delta_heading = self._quantize_heading_error(self._heading_error_to_vector(env, agent_id, target_xy))
        return np.array([delta_altitude, delta_heading, delta_velocity], dtype=np.float32)

    def normalize_action(self, env, agent_id, action):
        """将战术动作编号转换为低层飞控动作。"""
        action_id = int(np.asarray(action).reshape(-1)[0])
        delta_control = self._tactical_action_to_delta_control(env, agent_id, action_id)
        return self._run_lowlevel_policy(env, agent_id, delta_control)


class StraightFlyAgent:

    def normalize_action(self, action):
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.   # 0~40 => -1~1
        norm_act[1] = action[1] / 20 - 1.   # 0~40 => -1~1
        norm_act[2] = action[2] / 20 - 1.   # 0~40 => -1~1
        norm_act[3] = action[3] / 58 + 0.4  # 0~29 => 0.4~0.9
        return norm_act

    def get_action(self, sim: AircraftSimulator):
        action = np.array([20, 18.6, 20, 0])
        return self.normalize_action(action)

    def reset(self):
        pass


class BaselineAgent:
    def __init__(self) -> None:
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.actor.eval()
        self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.velocities_u_mps,                 #  5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  8. vc        (unit: m/s)
            c.position_h_sl_m                   #  9. altitude  (unit: m)
        ]
        self.reset()

    def normalize_action(self, action):
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.   # 0~40 => -1~1
        norm_act[1] = action[1] / 20 - 1.   # 0~40 => -1~1
        norm_act[2] = action[2] / 20 - 1.   # 0~40 => -1~1
        norm_act[3] = action[3] / 58 + 0.4  # 0~29 => 0.4~0.9
        return norm_act

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    def set_delta_value(self, sim: AircraftSimulator):
        raise NotImplementedError

    def get_observation(self, sim: AircraftSimulator, delta_value):
        obs = sim.get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = obs[9] / 5000                  #  3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(obs[3])                 #  4. ego_roll_sin
        norm_obs[5] = np.cos(obs[3])                 #  5. ego_roll_cos
        norm_obs[6] = np.sin(obs[4])                 #  6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[4])                 #  7. ego_pitch_cos
        norm_obs[8] = obs[5] / 340                   #  8. ego_v_x   (unit: mh)
        norm_obs[9] = obs[6] / 340                   #  9. ego_v_y    (unit: mh)
        norm_obs[10] = obs[7] / 340                  #  10. ego_v_z    (unit: mh)
        norm_obs[11] = obs[8] / 340                  #  11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs

    def get_action(self, sim: AircraftSimulator):
        delta_value = self.set_delta_value(sim)
        observation = self.get_observation(sim, delta_value)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return self.normalize_action(action)


class PursueAgent(BaselineAgent):
    def __init__(self) -> None:
        super().__init__()

    def set_delta_value(self, sim: AircraftSimulator):
        # NOTE: only adapt for 1v1
        ego_x, ego_y, ego_z = sim.get_position()
        ego_vx, ego_vy, ego_vz = sim.get_velocity()
        enm_x, enm_y, enm_z = sim.enemies[0].get_position()
        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = sim.enemies[0].get_property_value(c.velocities_u_mps) - \
                         sim.get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, maneuver: Literal['l', 'r', 'n']) -> None:
        super().__init__()
        self.turn_interval = 30
        self.dodge_missile = True # if set true, start turn when missile is detected
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        # self.target_altitude_list = [8000, 7000, 7500, 5500, 6000, 6000]
        # self.target_velocity_list = [340, 340, 340, 340, 243, 243]
        self.target_altitude_list = [6096] * 4
        self.target_velocity_list = [243] * 4

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None

    def set_delta_value(self, sim: AircraftSimulator):
        step_list = np.arange(1, len(self.target_heading_list)+1) * self.turn_interval / 0.2
        cur_heading = sim.get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
        if not self.dodge_missile or len(sim.under_missiles) != 0:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            delta_altitude = self.target_altitude_list[i] - sim.get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - sim.get_property_value(c.velocities_u_mps)
            self.step += 1
        else:
            delta_heading = self.init_heading  - cur_heading
            delta_altitude = 6096 - sim.get_property_value(c.position_h_sl_m)
            delta_velocity = 243 - sim.get_property_value(c.velocities_u_mps)

        return np.array([delta_altitude, delta_heading, delta_velocity])


class DodgeMissileAgent:
    def __init__(self) -> None:
        self.model_path = get_root_dir() + '/model/dodge_missile_model.pt'
        self.actor = BaselineActor(input_dim=21, use_mlp_actlayer=True)
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
        ]
        self.reset()

    def get_observation(self, sim: AircraftSimulator):
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(sim.get_property_values(self.state_var))
        enm_obs_list = np.array(sim.enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 120.0, 60.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 120.0, 60.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        enm_v_body_x = enm_obs_list[9]
        enm_alt_m = enm_obs_list[2]
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_obs_list[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_obs_list[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_obs_list[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_obs_list[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_obs_list[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_obs_list[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_obs_list[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_obs_list[12] / 340            # 8. ego vc   (unit: mh)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_v_body_x - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_alt_m - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        if len(sim.under_missiles) != 0 and sim.under_missiles[0].is_alive:
            missile_sim = sim.under_missiles[0]
        else:
            missile_sim = None
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        norm_obs = np.expand_dims(norm_obs, axis=0)
        return norm_obs

    def get_action(self, sim: AircraftSimulator):
        obs = self.get_observation(sim)
        _action, self.rnn_states = self.actor(obs, self.rnn_states)
        action = _action.squeeze().detach().cpu().numpy().squeeze()
        return action

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))
