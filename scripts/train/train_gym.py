#!/usr/bin/env python
import sys
import os
import gymnasium as gym
import torch
import random
import numpy as np
from contextlib import ExitStack
import logging
import setproctitle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from runner.experiment_tracking import create_run_dir, track_swanlab
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv


class GymEnv:
    def __init__(self, env):
        self.env = env
        self.action_shape = self.env.action_space.shape
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reset_seed = None

    def reset(self):
        observation, _ = self.env.reset(seed=self.reset_seed)
        self.reset_seed = None
        return np.array(observation).reshape((1, -1))

    def step(self, action):
        action = np.array(action).reshape(self.action_shape)
        return self._step(action.item() if action.ndim == 0 else action)

    def _step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = np.array(observation).reshape((1, -1))
        done = np.array(terminated or truncated).reshape((1, -1))
        reward = np.array(reward).reshape((1, -1))
        return observation, reward, done, info

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()
    
    def seed(self, seed=None):
        self.reset_seed = seed
    

class GymHybridEnv(GymEnv):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.discrete_dims = self.action_space[0].shape[0]
        self.continuous_dims = self.action_space[1].shape[0]
        self.action_shape = (self.discrete_dims+self.continuous_dims,)

    def step(self, action):
        action = np.array(action).reshape(self.action_shape)
        discrete_a, continuous_a = action[:self.discrete_dims].astype(np.int32), action[self.discrete_dims:]
        return self._step((discrete_a, continuous_a))


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = gym.make(all_args.scenario_name)
            env = GymEnv(env)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1: 
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = gym.make(all_args.scenario_name)
            if 'Moving' in  all_args.scenario_name:
                env = GymHybridEnv(env)
            elif 'Volleyball' in all_args.scenario_name:
                pass
            else:
                env = GymEnv(env)
            env.seed(all_args.seed + rank * 1001)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1: 
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    group = parser.add_argument_group("Gym Env parameters")
    group.add_argument('--scenario-name', type=str, default='CartPole-v1',
                       help="the name of gym env")
    group.add_argument('--episode-length', type=int, default=1000,
                       help="the max length of an episode")
    group.add_argument('--num-agents', type=int, default=1,
                       help="number of agents controlled by RL policy")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = create_run_dir(all_args)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    with ExitStack() as stack:
        envs = make_train_env(all_args)
        stack.callback(envs.close)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        if eval_envs is not None:
            stack.callback(eval_envs.close)

        swanlab_run = stack.enter_context(track_swanlab(all_args, run_dir))

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "device": device,
            "run_dir": run_dir,
            "render_mode": "txt",
            "swanlab_run": swanlab_run,
        }

        if all_args.use_selfplay:
            from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            from runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)
        runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
