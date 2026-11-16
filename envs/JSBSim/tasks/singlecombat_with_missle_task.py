import numpy as np
from gymnasium import spaces
from collections import deque

from .singlecombat_task import (
    SingleCombatTask,
    HierarchicalSingleCombatTask,
    TacticalHierarchicalSingleCombatTask,
    _get_enemy_radar_track,
)
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R


class SingleCombatDodgeMissileTask(SingleCombatTask):
    """This task aims at training agent to dodge missile attacking
    """
    def __init__(self, config):
        super().__init__(config)

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

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
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        enm_v_body_x = enm_obs_list[9]
        enm_alt_m = enm_obs_list[2]
        radar_track = _get_enemy_radar_track(env, agent_id)
        if radar_track is not None:
            enm_feature = np.array([
                radar_track[0], radar_track[1], radar_track[2],
                radar_track[4], radar_track[5], radar_track[6],
            ])
            enm_v_body_x = radar_track[7]
            enm_alt_m = radar_track[3]
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_v_body_x - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_alt_m - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        return super().reset(env)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [Rule-based missile launch]
            target = agent.enemies[0].get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < self.max_attack_angle)
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= self.max_attack_distance and self.remaining_missiles[agent_id] > 0 and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step


class HierarchicalSingleCombatDodgeMissileTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        return HierarchicalSingleCombatTask.load_action_space(self)

    def get_obs(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        return SingleCombatDodgeMissileTask.step(self, env)


class TacticalMissileAwareMixin:
    def _incoming_missile_geometry(self, env, agent_id):
        ego = env.agents[agent_id]
        missile_sim = ego.check_missile_warning()
        if missile_sim is None:
            return None

        ego_pos = np.asarray(ego.get_position(), dtype=np.float32)
        missile_pos = np.asarray(missile_sim.get_position(), dtype=np.float32)
        ego_vel = np.asarray(ego.get_velocity(), dtype=np.float32)
        missile_vel = np.asarray(missile_sim.get_velocity(), dtype=np.float32)

        relative = missile_pos - ego_pos
        relative_xy = relative[:2]
        distance = float(np.linalg.norm(relative))
        los_xy = self._safe_unit_vector(relative_xy, fallback=self._ego_heading_vector(env, agent_id))
        relative_velocity = missile_vel - ego_vel
        closing_speed = max(0.0, -float(np.dot(relative, relative_velocity) / (distance + 1e-6)))

        danger_distance = getattr(self.config, "tactical_missile_danger_distance", 9000.0)
        watch_distance = getattr(self.config, "max_attack_distance", 14000.0)
        danger_closing_speed = getattr(self.config, "tactical_missile_danger_closing_speed", 150.0)
        dangerous = distance <= danger_distance or (
            distance <= watch_distance and closing_speed >= danger_closing_speed
        )

        return {
            "distance": distance,
            "closing_speed": closing_speed,
            "los_xy": los_xy,
            "away_xy": -los_xy,
            "left_beam_xy": self._rotate_vector(los_xy, np.pi / 2),
            "right_beam_xy": self._rotate_vector(los_xy, -np.pi / 2),
            "dangerous": dangerous,
        }

    def _mix_tactical_vectors(self, primary, secondary, primary_weight):
        primary_vector = self._safe_unit_vector(primary)
        secondary_vector = self._safe_unit_vector(secondary, fallback=primary_vector)
        return primary_vector * primary_weight + secondary_vector * (1.0 - primary_weight)

    def _preferred_missile_beam(self, missile_geometry, ego_heading_xy):
        left_score = np.dot(ego_heading_xy, missile_geometry["left_beam_xy"])
        right_score = np.dot(ego_heading_xy, missile_geometry["right_beam_xy"])
        return missile_geometry["left_beam_xy"] if left_score >= right_score else missile_geometry["right_beam_xy"]

    def _missile_aware_action_to_delta_control(self, env, agent_id, action_id, missile_geometry):
        geometry = self._combat_geometry(env, agent_id)
        relative_xy = geometry["relative_xy"]
        ego_heading_xy = geometry["ego_heading_xy"]
        distance_xy = geometry["distance_xy"]
        preferred_beam_xy = self._preferred_missile_beam(missile_geometry, ego_heading_xy)
        close_threat_distance = getattr(self.config, "tactical_missile_close_distance", 3500.0)
        close_threat = missile_geometry["distance"] <= close_threat_distance

        delta_altitude = 0.0
        delta_velocity = 0.0

        if action_id in (self.PURE_PURSUIT, self.LEAD_PURSUIT, self.LAG_PURSUIT):
            # 来袭导弹较近时，追击类动作改为带横切角的 crank，避免持续正对敌机暴露。
            beam_weight = 0.8 if close_threat else 0.6
            target_xy = self._mix_tactical_vectors(preferred_beam_xy, relative_xy, beam_weight)
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = 0.0 if close_threat else 0.05
        elif action_id == self.DISENGAGE:
            # 脱离动作优先远离来袭导弹，同时保留少量当前航向惯性。
            target_xy = self._mix_tactical_vectors(missile_geometry["away_xy"], ego_heading_xy, 0.85)
            delta_velocity = 0.05
        elif action_id == self.CLIMB_POSITION:
            # 爬升占位在导弹威胁下改为横切爬升，兼顾规避和高度安全。
            target_xy = self._mix_tactical_vectors(preferred_beam_xy, relative_xy, 0.7)
            delta_altitude = 0.1
        elif action_id == self.DIVE_ACCELERATE:
            # 俯冲加速转为 away + beam，避免只朝敌机方向俯冲导致被导弹追尾。
            target_xy = self._mix_tactical_vectors(missile_geometry["away_xy"], preferred_beam_xy, 0.75)
            delta_altitude = -0.1
            delta_velocity = 0.05
        elif action_id == self.LEVEL_ACCELERATE:
            # 平飞加速优先把速度投到 away/beam 方向，保留规避能量。
            target_xy = self._mix_tactical_vectors(missile_geometry["away_xy"], preferred_beam_xy, 0.6)
            delta_velocity = 0.05
        elif action_id == self.LEVEL_DECELERATE:
            # 导弹威胁下不主动减速，保留能量做横切。
            target_xy = preferred_beam_xy
        elif action_id == self.DEFENSIVE_TURN_LEFT:
            # 左/右防御转弯改为相对导弹视线的 beam/notch。
            target_xy = missile_geometry["left_beam_xy"]
            delta_velocity = 0.05
        elif action_id == self.DEFENSIVE_TURN_RIGHT:
            target_xy = missile_geometry["right_beam_xy"]
            delta_velocity = 0.05
        elif action_id == self.HIGH_YOYO:
            # 高悠悠在导弹威胁下解释为横切爬升，减少迎头暴露。
            target_xy = self._mix_tactical_vectors(preferred_beam_xy, relative_xy, 0.65)
            delta_altitude = 0.1
        elif action_id == self.LOW_YOYO:
            # 低悠悠解释为横切加速脱离，用能量换规避窗口。
            target_xy = self._mix_tactical_vectors(preferred_beam_xy, missile_geometry["away_xy"], 0.55)
            delta_altitude = -0.1
            delta_velocity = 0.05
        else:
            raise ValueError(f"Unknown tactical action id: {action_id}")

        delta_altitude, delta_velocity = self._apply_tactical_safety(
            env, agent_id, action_id, delta_altitude, delta_velocity, distance_xy
        )
        delta_heading = self._quantize_heading_error(self._heading_error_to_vector(env, agent_id, target_xy))
        return np.array([delta_altitude, delta_heading, delta_velocity], dtype=np.float32)

    def _tactical_action_to_delta_control(self, env, agent_id, action_id):
        missile_geometry = self._incoming_missile_geometry(env, agent_id)
        if missile_geometry is None or not missile_geometry["dangerous"]:
            return TacticalHierarchicalSingleCombatTask._tactical_action_to_delta_control(
                self, env, agent_id, action_id
            )
        return self._missile_aware_action_to_delta_control(env, agent_id, action_id, missile_geometry)


class TacticalHierarchicalSingleCombatDodgeMissileTask(
    TacticalMissileAwareMixin, TacticalHierarchicalSingleCombatTask, SingleCombatDodgeMissileTask
):

    def __init__(self, config: str):
        TacticalHierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        return TacticalHierarchicalSingleCombatTask.load_action_space(self)

    def get_obs(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        return TacticalHierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        return SingleCombatDodgeMissileTask.step(self, env)


class SingleCombatShootMissileTask(SingleCombatDodgeMissileTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])
    
    def get_obs(self, env, agent_id):
        return super().get_obs(env, agent_id)
    
    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        super().reset(env)

    def _get_launch_geometry(self, agent):
        target = agent.enemies[0].get_position() - agent.get_position()
        heading = agent.get_velocity()
        distance = np.linalg.norm(target)
        attack_angle = np.rad2deg(np.arccos(np.clip(
            np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1
        )))
        return distance, attack_angle

    def _can_launch_missile(self, env, agent_id, agent):
        # 1v1 RL 射击也使用 YAML 中的发射窗口，避免策略只凭 shoot_flag 打光导弹。
        distance, attack_angle = self._get_launch_geometry(agent)
        shoot_interval = env.current_step - self._last_shoot_time[agent_id]
        return agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0 \
            and attack_angle <= self.max_attack_angle and distance <= self.max_attack_distance \
            and shoot_interval >= self.min_attack_interval
    
    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = self._can_launch_missile(env, agent_id, agent)
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step


class HierarchicalSingleCombatShootTask(HierarchicalSingleCombatTask, SingleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatShootMissileTask.load_observation_space(self)

    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        return SingleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        self._shoot_action[agent_id] = action[-1]
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-1].astype(np.int32))

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        SingleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatShootMissileTask.step(self, env)


class TacticalHierarchicalSingleCombatShootTask(
    TacticalMissileAwareMixin, TacticalHierarchicalSingleCombatTask, SingleCombatShootMissileTask
):
    def __init__(self, config: str):
        TacticalHierarchicalSingleCombatTask.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatShootMissileTask.load_observation_space(self)

    def load_action_space(self):
        # 战术机动动作 + 导弹发射动作。
        self.action_space = spaces.Tuple([spaces.Discrete(12), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        return SingleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """将战术动作和发射标志转换为低层飞控动作。"""
        action = np.asarray(action, dtype=np.int32).reshape(-1)
        self._shoot_action[agent_id] = action[-1]
        return TacticalHierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[0])

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatShootMissileTask.step(self, env)
