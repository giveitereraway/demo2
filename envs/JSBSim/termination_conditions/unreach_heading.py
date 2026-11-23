import math
from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition


class UnreachHeading(BaseTerminationCondition):
    """
    UnreachHeading [0, 1]
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        uid = list(config.aircraft_configs.keys())[0]
        aircraft_config = config.aircraft_configs[uid]
        self.max_heading_increment = aircraft_config['max_heading_increment']
        self.max_altitude_increment = aircraft_config['max_altitude_increment']
        self.max_velocities_u_increment = aircraft_config['max_velocities_u_increment']
        self.check_interval = aircraft_config['check_interval']
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10

    def advance_target(self, env, agent_id):
        """按当前课程阶段以上一目标为基准采样，返回限幅前的指令增量。"""
        sim = env.agents[agent_id]
        delta = self.increment_size[env.heading_turn_counts]
        delta_heading = env.np_random.uniform(-delta, delta) * self.max_heading_increment
        delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment
        delta_velocities_u = env.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
        new_heading = (sim.get_property_value(c.target_heading_deg) + delta_heading + 360) % 360
        new_altitude = max(sim.get_property_value(c.target_altitude_ft) + delta_altitude, 15000)
        new_velocities_u = max(sim.get_property_value(c.target_velocities_u_mps) + delta_velocities_u, 120.0)
        sim.set_property_value(c.target_heading_deg, new_heading)
        sim.set_property_value(c.target_altitude_ft, new_altitude)
        sim.set_property_value(c.target_velocities_u_mps, new_velocities_u)
        env.heading_turn_counts += 1
        self.log(f'current_step:{env.current_step} target_heading:{new_heading} '
                 f'target_altitude_ft:{new_altitude} target_velocities_u_mps:{new_velocities_u}')
        return delta_heading, delta_altitude, delta_velocities_u

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target heading in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        done = False
        success = False
        check_time = env.agents[agent_id].get_property_value(c.heading_check_time)
        # check heading when simulation_time exceed check_time
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) > 10:
                done = True
            # if current target heading is reached, random generate a new target heading
            else:
                self.advance_target(env, agent_id)
                env.agents[agent_id].set_property_value(c.heading_check_time, check_time + self.check_interval)
        if done:
            self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}')
            info['heading_turn_counts'] = env.heading_turn_counts
        success = False
        return done, success, info
