import sys
import os
import pytest
import gymnasium as gym
import numpy as np
from itertools import product
from types import SimpleNamespace
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.simulatior import MissileSimulator
from envs.JSBSim.reward_functions.event_driven_reward import EventDrivenReward
from envs.JSBSim.tasks.singlecombat_task import SingleCombatTask, TacticalHierarchicalSingleCombatTask
from envs.JSBSim.utils.utils import parse_config
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv


class TestSingleControlEnv:

    def test_env(self):
        # Env Settings test
        env = SingleControlEnv("1/heading")
        assert env.num_agents == 1
        for agent in env.agents.values():
            assert len(agent.partners) == 0
            assert len(agent.enemies) == 0
        action_space = env.action_space

        # DataType test
        obs_shape = (env.num_agents, *env.observation_space.shape)
        act_shape = (env.num_agents, *env.action_space.shape)
        reward_shape = (env.num_agents, 1)
        done_shape = (env.num_agents, 1)

        env.seed(0)
        action_space.seed(0)
        obs = env.reset()
        assert obs.shape == obs_shape

        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            actions = np.array([action_space.sample() for _ in range(env.num_agents)])
            obs, reward, done, info = env.step(actions)
            assert obs.shape == obs_shape and actions.shape == act_shape \
                and reward.shape == reward_shape and done.shape == done_shape
            act_buf.append(actions)
            obs_buf.append(obs)
            rew_buf.append(reward)
            done_buff.append(done)
            if done:
                assert env.current_step <= env.max_steps
                break

        # Repetition test (same seed => same data)
        env.seed(0)
        obs = env.reset()
        t = 0
        assert np.linalg.norm(obs - obs_buf[t]) < 1e-8
        while t < len(done_buff):
            obs, reward, done, info = env.step(act_buf[t])
            assert np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8 \
                and np.all(reward == rew_buf[t]) and np.all(done == done_buff[t])
            t += 1

    @pytest.mark.parametrize("vecenv", [DummyVecEnv, SubprocVecEnv])
    def test_vec_env(self, vecenv):
        parallel_num = 4
        envs = vecenv([lambda: SingleControlEnv("1/heading") for _ in range(parallel_num)])

        # DataType test
        obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
        act_shape = (parallel_num, envs.num_agents, *envs.action_space.shape)
        reward_shape = (parallel_num, envs.num_agents, 1)
        done_shape = (parallel_num, envs.num_agents, 1)

        obss = envs.reset()
        assert obss.shape == obs_shape

        actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
        while True:
            obss, rewards, dones, infos = envs.step(actions)
            assert obss.shape == obs_shape and actions.shape == act_shape \
                and rewards.shape == reward_shape and dones.shape == done_shape \
                and infos.shape[0] == parallel_num and isinstance(infos[0], dict)
            # terminate if any of the parallel envs has been done
            if np.any(dones):
                break
        envs.close()


class TestSingleCombatEnv:

    def test_no_weapon_selfplay_enables_artillery(self):
        config = parse_config('1v1/NoWeapon/Selfplay')
        task = SingleCombatTask(config)
        assert config.use_artillery is True
        assert task.use_artillery is True

    @pytest.mark.parametrize("config", ["1v1/NoWeapon/vsBaseline", "1v1/NoWeapon/Selfplay",
                                        "1v1/DodgeMissile/vsBaseline", "1v1/DodgeMissile/Selfplay",
                                        "1v1/DodgeMissile/HierarchyVsBaseline", "1v1/DodgeMissile/HierarchySelfplay"])
    def test_env(self, config):
        # Env Settings test
        env = SingleCombatEnv(config)
        for agent in env.agents.values():
            assert len(agent.partners) == 0
            assert len(agent.enemies) == 1
        action_space = env.action_space

        # DataType test
        obs_shape = (env.num_agents, *env.observation_space.shape)
        reward_shape = (env.num_agents, 1)
        done_shape = (env.num_agents, 1)

        env.seed(0)
        action_space.seed(0)
        obs = env.reset()
        assert obs.shape == obs_shape

        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            # Legal action inputs: List / np.ndarray
            if env.current_step % 2 == 0:
                actions = [action_space.sample() for _ in range(env.num_agents)]
            else:
                actions = np.array([action_space.sample() for _ in range(env.num_agents)])
            obs, rewards, dones, info = env.step(actions)
            assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape
            # save previous data
            act_buf.append(actions)
            obs_buf.append(obs)
            rew_buf.append(rewards)
            done_buff.append(dones)
            if np.all(dones):
                assert env.current_step <= env.max_steps
                break

        # Repetition test (same seed => same data)
        env.seed(0)
        obs = env.reset()
        t = 0
        assert np.linalg.norm(obs - obs_buf[t]) < 1e-8
        while t < len(done_buff):
            obs, rewards, dones, info = env.step(act_buf[t])
            assert np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8 \
                and np.all(rewards == rew_buf[t]) and np.all(dones == done_buff[t])
            t += 1

    @pytest.mark.parametrize("config, is_shoot", [
        ("1v1/NoWeapon/TacticalHierarchySelfplay", False),
        ("1v1/DodgeMissile/TacticalHierarchySelfplay", False),
        ("1v1/ShootMissile/TacticalHierarchySelfplay", True),
    ])
    def test_tactical_hierarchy_env(self, config, is_shoot):
        env = SingleCombatEnv(config)
        assert isinstance(env.action_space, gym.spaces.Tuple if is_shoot else gym.spaces.Discrete)
        if is_shoot:
            assert isinstance(env.action_space[0], gym.spaces.Discrete)
            assert env.action_space[0].n == 12 and env.action_space[1].n == 2
        else:
            assert env.action_space.n == 12

        obs_shape = (env.num_agents, *env.observation_space.shape)
        reward_shape = (env.num_agents, 1)
        done_shape = (env.num_agents, 1)

        env.seed(0)
        env.action_space.seed(0)
        obs = env.reset()
        assert obs.shape == obs_shape

        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        done_buf = []
        for _ in range(3):
            actions = [env.action_space.sample() for _ in range(env.num_agents)]
            obs, rewards, dones, info = env.step(actions)
            assert obs.shape == obs_shape
            assert rewards.shape == reward_shape
            assert dones.shape == done_shape
            obs_buf.append(obs)
            act_buf.append(actions)
            rew_buf.append(rewards)
            done_buf.append(dones)
            if np.all(dones):
                break

        env.seed(0)
        obs = env.reset()
        assert np.linalg.norm(obs - obs_buf[0]) < 1e-8
        for t, actions in enumerate(act_buf):
            obs, rewards, dones, info = env.step(actions)
            assert np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8
            assert np.allclose(rewards, rew_buf[t])
            assert np.all(dones == done_buf[t])
        env.close()

    def test_tactical_dodge_uses_missile_aware_action_translation(self, monkeypatch):
        env = SingleCombatEnv("1v1/DodgeMissile/TacticalHierarchySelfplay")
        try:
            env.seed(0)
            env.reset()
            agent_id = (env.ego_ids + env.enm_ids)[0]
            agent = env.agents[agent_id]

            action_id = env.task.LEVEL_ACCELERATE
            base_delta = TacticalHierarchicalSingleCombatTask._tactical_action_to_delta_control(
                env.task, env, agent_id, action_id
            )
            monkeypatch.setattr(agent, "check_missile_warning", lambda: None)
            fallback_delta = env.task._tactical_action_to_delta_control(env, agent_id, action_id)
            assert np.allclose(fallback_delta, base_delta)

            class FakeMissile:
                def __init__(self, position, velocity):
                    self.position = position
                    self.velocity = velocity

                def get_position(self):
                    return self.position

                def get_velocity(self):
                    return self.velocity

            ego_position = agent.get_position()
            ego_heading = env.task._ego_heading_vector(env, agent_id)
            missile_position = ego_position + np.array([ego_heading[0], ego_heading[1], 0.0]) * 1000.0
            missile_velocity = np.array([-ego_heading[0], -ego_heading[1], 0.0]) * 600.0
            fake_missile = FakeMissile(missile_position, missile_velocity)
            monkeypatch.setattr(agent, "check_missile_warning", lambda: fake_missile)

            missile_delta = env.task._tactical_action_to_delta_control(env, agent_id, action_id)
            assert missile_delta.shape == (3,)
            assert np.any(np.isclose(missile_delta[1], env.task.norm_delta_heading))
            assert not np.allclose(missile_delta, base_delta)
        finally:
            env.close()

    def test_shoot_missile_respects_launch_window(self):
        env = SingleCombatEnv("1v1/ShootMissile/Selfplay")
        try:
            env.seed(0)
            env.reset()
            agent_ids = (env.ego_ids + env.enm_ids)[:env.num_agents]

            # 默认射击场景开局距离大于最大射程，shoot_flag=1 也不能直接发射。
            for agent_id in agent_ids:
                agent = env.agents[agent_id]
                target = agent.enemies[0].get_position() - agent.get_position()
                assert np.linalg.norm(target) > env.task.max_attack_distance

            before_remaining = env.task.remaining_missiles.copy()
            before_launched = {
                agent_id: len(env.agents[agent_id].launch_missiles)
                for agent_id in agent_ids
            }
            actions = [
                np.array([20, 20, 20, 15, 1], dtype=np.int32)
                for _ in range(env.num_agents)
            ]

            env.step(actions)

            assert env.task.remaining_missiles == before_remaining
            for agent_id in agent_ids:
                assert len(env.agents[agent_id].launch_missiles) == before_launched[agent_id]
        finally:
            env.close()

    def test_missile_hit_state_persists_for_stats_and_event_reward(self, monkeypatch):
        class FakeAircraft:
            def __init__(self, uid, color, position):
                self.uid = uid
                self.color = color
                self.dt = 1 / 60
                self.lon0, self.lat0, self.alt0 = 120.0, 60.0, 0.0
                self.launch_missiles = []
                self.under_missiles = []
                self._position = np.asarray(position, dtype=np.float64)
                self._velocity = np.array([300.0, 0.0, 0.0], dtype=np.float64)
                self._posture = np.zeros(3)
                self._shotdown = False
                self._crash = False

            @property
            def is_alive(self):
                return not self._shotdown and not self._crash

            @property
            def is_shotdown(self):
                return self._shotdown

            @property
            def is_crash(self):
                return self._crash

            def shotdown(self):
                self._shotdown = True

            def get_geodetic(self):
                return np.array([120.0, 60.0, 6000.0], dtype=np.float64)

            def get_position(self):
                return self._position

            def get_velocity(self):
                return self._velocity

            def get_rpy(self):
                return self._posture

        parent = FakeAircraft("A0100", "Red", [0.0, 0.0, 6000.0])
        target = FakeAircraft("B0100", "Blue", [0.0, 0.0, 6000.0])
        missile = MissileSimulator.create(parent=parent, target=target, uid="A01002")
        monkeypatch.setattr(missile, "_guidance", lambda: (np.zeros(2), 0.0))
        env = SimpleNamespace(agents={"A0100": parent, "B0100": target})
        reward = EventDrivenReward(SimpleNamespace(EventDrivenReward_scale=1, EventDrivenReward_potential=True))
        reward.reset(None, env)

        # 命中后再次 run 也不能把 HIT 终态覆盖为 MISS，否则统计和事件奖励都会漏掉命中。
        missile.run()
        assert missile.is_success
        assert target.is_shotdown
        missile._t = missile._t_max + 1
        missile.run()
        assert missile.is_success

        assert reward.get_reward(None, env, "A0100") == 200
        assert reward.get_reward(None, env, "B0100") == -200

    @pytest.mark.parametrize("config", ["1v1/NoWeapon/vsBaseline", "1v1/NoWeapon/Selfplay"])
    def test_agent_crash(self, config):
        # if no weapon, once enemy die, env terminate!
        env = SingleCombatEnv(config)
        env.seed(0)
        obs = env.reset()
        env.agents[env.ego_ids[0]].crash()
        actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
        obs, rewards, dones, info = env.step(actions)
        assert np.min(rewards) < -100  # crash reward!
        assert np.all(dones)

    @pytest.mark.parametrize("config", ["1v1/DodgeMissile/vsBaseline", "1v1/DodgeMissile/Selfplay"])
    def test_agent_shotdown(self, config):
        # if has weapon, once enemy die, env terminate until no missile warning!
        env = SingleCombatEnv(config)
        env.seed(0)
        obs = env.reset()
        crash_id = env.ego_ids[0]   # ego shotdown
        while True:
            # mannual crash
            if env.current_step == 1:
                from envs.JSBSim.core.simulatior import MissileSimulator
                env.add_temp_simulator(MissileSimulator.create(env.agents[crash_id], env.agents[crash_id].enemies[0], 'C0000'))
                env.agents[crash_id].shotdown()
                crash_obs = obs[0]
            actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])

            obs, rewards, dones, info = env.step(actions)

            if np.all(dones):
                break
            elif env.current_step == 2:
                assert np.min(rewards) < -50  # shot down reward!
            elif env.current_step > 2:
                # ego obs is not changed!
                assert dones[0][0] == True \
                    and np.linalg.norm(obs[0, :9] - crash_obs[:9]) < 1e-8 \
                    and rewards[0][0] == 0.0 \
                    and any([missile.is_alive for missile in env.agents[crash_id].launch_missiles])

    @pytest.mark.parametrize("vecenv, config", list(product(
        [DummyVecEnv, SubprocVecEnv], ["1v1/DodgeMissile/Selfplay", "1v1/DodgeMissile/HierarchyVsBaseline"])))
    def test_vec_env(self, vecenv, config):
        parallel_num = 4
        envs = vecenv([lambda: SingleCombatEnv(config) for _ in range(parallel_num)])

        # DataType test
        obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
        reward_shape = (parallel_num, envs.num_agents, 1)
        done_shape = (parallel_num, envs.num_agents, 1)

        obss = envs.reset()
        assert obss.shape == obs_shape

        # Legal action inputs: List / np.ndarray (first ego, then enm)
        actions = [[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)]
        while True:
            obss, rewards, dones, infos = envs.step(actions)
            # check parallel env's data type
            assert obss.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape \
                and infos.shape[0] == parallel_num and isinstance(infos[0], dict)
            # terminate if any of the parallel envs has been done
            if np.any(np.all(dones, axis=1)):
                break
        envs.close()


class TestJSBSimRunner:

    @pytest.mark.parametrize("args", [
        "--env-name SingleControl --algorithm-name ppo --scenario-name 1/heading",
        "--env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/DodgeMissile/Selfplay --use-selfplay --selfplay-algorithm fsp",
        "--env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/DodgeMissile/vsBaseline",
        "--env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/DodgeMissile/HierarchySelfplay",  # whether to use selfplay is optional
        "--env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/DodgeMissile/HierarchyVsBaseline"])
    def test_training(self, args, tmp_path, monkeypatch):
        from runner import experiment_tracking
        from scripts.train.train_jsbsim import main
        args += ' --experiment-name pytest --seed 1 --n-training-threads 1 --n-rollout-threads 5 --cuda' \
                ' --log-interval 1 --save-interval 1 --use-eval --eval-interval 1 --eval-episodes 10' \
                ' --num-mini-batch 5 --buffer-size 1000 --num-env-steps 1e4' \
                ' --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3' \
                ' --hidden-size 32 --act-hidden-size 32 --recurrent-hidden-size 32 --recurrent-hidden-layers 1 --data-chunk-length 8'
        monkeypatch.setattr(experiment_tracking, "REPO_ROOT", tmp_path)
        main([*args.split(), "--swanlab-mode", "disabled"])
        assert list(tmp_path.rglob("actor_latest.pt"))



class TestMultipleCombatEnv:

    @pytest.mark.parametrize("config", ["2v2/NoWeapon/Selfplay", "2v2/NoWeapon/HierarchySelfplay",
                                        "2v2/ShootMissile/HierarchySelfplay"])
    def test_env(self, config):
        # Env Settings test
        env = MultipleCombatEnv(config)
        assert env.num_agents == 4
        for agent in env.agents.values():
            assert len(agent.partners) == 1
            assert len(agent.enemies) == 2

        obs_shape = (env.num_agents, *env.observation_space.shape)
        share_obs_shape = (env.num_agents, *env.share_observation_space.shape)
        reward_shape = (env.num_agents, 1)
        done_shape = (env.num_agents, 1)

        # DataType test
        env.seed(0)
        env.action_space.seed(0)
        obs, share_obs = env.reset()
        assert obs.shape == obs_shape and share_obs.shape == share_obs_shape

        obs_buf = [obs]
        share_buf = [share_obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            actions = [env.action_space.sample() for _ in range(env.num_agents)]
            obs, share_obs, rewards, dones, info = env.step(actions)
            assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape and share_obs_shape
            # save previous data
            obs_buf.append(obs)
            share_buf.append(share_obs)
            act_buf.append(actions)
            rew_buf.append(rewards)
            done_buff.append(dones)
            if np.any(dones[0]):
                assert env.current_step <= env.max_steps
                break

        # Repetition test (same seed => same data)
        env.seed(0)
        obs, share_obs = env.reset()
        t = 0
        assert np.linalg.norm(obs - obs_buf[t]) < 1e-8 \
            and np.linalg.norm(share_obs - share_buf[t]) < 1e-8
        while t < len(done_buff):
            obs, share_obs, rewards, dones, info = env.step(act_buf[t])
            assert np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8 \
                and np.linalg.norm(share_obs - share_buf[t+1]) < 1e-8 \
                and np.all(rewards == rew_buf[t]) and np.all(dones == done_buff[t])
            t += 1

    def test_agent_die(self):
        env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
        uid = list(env.agents.keys())[0]
        partner_id = env.agents[uid].partners[0].uid
        enemy0_id = env.agents[uid].enemies[0].uid
        enemy1_id = env.agents[uid].enemies[1].uid
        env.seed(0)
        env.reset()
        while True:
            actions = [np.array([20, 18.6, 20, 0]) for _ in range(env.num_agents)]
            if env.current_step == 20:
                env.agents[partner_id].crash()
            if env.current_step == 40:
                env.agents[enemy0_id].crash()
            if env.current_step == 60:
                env.agents[enemy1_id].crash()

            obs, share_obs, rewards, dones, info = env.step(actions)
            rewards = env._unpack(rewards)
            dones = env._unpack(dones)
            if env.current_step > 20:
                assert dones[partner_id] == True and rewards[partner_id] == 0.0
                if env.current_step > 40:
                    assert dones[enemy0_id] == True and rewards[enemy0_id] == 0.0
            if env.current_step == 61:
                assert np.all(list(dones.values()))
                break

        # if has weapon, once all enemies die, env terminate until no missile warning!
        env.seed(0)
        env.reset()
        while True:
            actions = [np.array([20, 18.6, 20, 0]) for _ in range(env.num_agents)]

            if env.current_step == 20:
                env.agents[enemy0_id].crash()
            if env.current_step == 40:
                env.agents[enemy1_id].crash()
                from envs.JSBSim.core.simulatior import MissileSimulator
                env.add_temp_simulator(MissileSimulator.create(env.agents[enemy1_id], env.agents[uid], uid="C0000"))

            obs, share_obs, rewards, dones, info = env.step(actions)
            rewards = env._unpack(rewards)
            dones = env._unpack(dones)
            if env.current_step > 20:
                assert dones[enemy0_id] == True and rewards[enemy0_id] == 0.0
                if env.current_step > 40:
                    assert dones[enemy1_id] == True and rewards[enemy1_id] == 0.0
            if np.all(list(dones.values())):
                assert not env._tempsims["C0000"].is_alive
                break

    @pytest.mark.parametrize("vecenv, config", list(product(
        [ShareDummyVecEnv, ShareSubprocVecEnv], ["2v2/NoWeapon/Selfplay", "2v2/NoWeapon/HierarchySelfplay",
                                        "2v2/ShootMissile/HierarchySelfplay"])))
    def test_vec_env(self, vecenv, config):
        parallel_num = 4
        envs = vecenv([lambda: MultipleCombatEnv(config) for _ in range(parallel_num)])
        assert envs.num_agents == 4
        obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
        share_obs_shape = (parallel_num, envs.num_agents, *envs.share_observation_space.shape)
        reward_shape = (parallel_num, envs.num_agents, 1)
        done_shape = (parallel_num, envs.num_agents, 1)

        # DataType test
        obs, share_obs = envs.reset()
        assert obs.shape == obs_shape and share_obs.shape == share_obs_shape
        while True:
            actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
            obs, share_obs, rewards, dones, info = envs.step(actions)
            assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape and share_obs_shape
            break
        envs.close()
