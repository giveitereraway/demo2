#!/usr/bin/env python
import sys
import os
import torch
import random
import logging
import numpy as np
from contextlib import ExitStack
import setproctitle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from runner.experiment_tracking import create_run_dir, track_swanlab
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.JSBSim.envs.singlecontrol_env import (
    HEADING_MEASUREMENT_NOISE_PROBABILITIES,
    HEADING_MEASUREMENT_NOISE_STD,
    HeadingMeasurementNoiseEnv,
    HeadingTrackingEvalEnv,
)
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
from runner.tacview import Tacview

HEADING_EVAL_NOISE_LEVEL = 2


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env_type = HeadingMeasurementNoiseEnv if all_args.heading_measurement_noise else SingleControlEnv
                env = env_type(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    else:
        if all_args.n_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                noise_level = HEADING_EVAL_NOISE_LEVEL if all_args.heading_measurement_noise else 0
                if all_args.heading_first_target:
                    env = HeadingTrackingEvalEnv(all_args.scenario_name, noise_level=noise_level)
                elif noise_level:
                    env = HeadingMeasurementNoiseEnv(all_args.scenario_name, noise_level=noise_level)
                else:
                    env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_eval_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    group.add_argument('--render-mode', type=str, default='txt',
                       help="txt or real_time")
    group.add_argument("--heading-measurement-noise", action="store_true",
                       help="Heading 训练采样随机加测量噪声，定期评估固定使用第二档噪声")
    all_args = parser.parse_known_args(args)[0]
    if all_args.heading_measurement_noise and (
        (all_args.env_name, all_args.scenario_name, all_args.algorithm_name) != ("SingleControl", "1/heading", "ppo")
        or all_args.use_selfplay
    ):
        parser.error("--heading-measurement-noise 仅支持 SingleControl / 1/heading 的 PPO 训练")
    if all_args.policy_loss == "spo" and (all_args.algorithm_name != "ppo"
            or all_args.env_name not in {"SingleControl", "SingleCombat"}):
        parser.error("SPO 仅支持 SingleControl / SingleCombat 的 PPO 训练，暂不支持 MAPPO / MultipleCombat")
    from algorithms.utils.temporal_attention import get_memory_encoder
    try:
        encoder = get_memory_encoder(all_args)
        if encoder == "temporal_attention" and all_args.env_name not in {"SingleControl", "SingleCombat"}:
            raise ValueError("时序注意力仅支持 SingleControl 和 SingleCombat 的 PPO 场景")
    except ValueError as error:
        parser.error(str(error))
    if all_args.heading_first_target:
        if (all_args.env_name, all_args.scenario_name) != ("SingleControl", "1/heading") or all_args.use_selfplay:
            parser.error("--heading-first-target 仅支持 SingleControl / 1/heading")
        if not all_args.use_eval:
            parser.error("--heading-first-target 需要 --use-eval")
        if len(all_args.heading_error_limits) not in (2, 3) or not all(
                np.isfinite(limit) and limit > 0 for limit in all_args.heading_error_limits):
            parser.error("--heading-error-limits 需要两个或三个有限正数（航向、高度、可选速度）")
        if not 0 < all_args.heading_target_pass_rate <= 1 or all_args.heading_target_eval_seed < 0:
            parser.error("达标率须在 (0, 1] 内，验证种子须为非负整数")
        if min(all_args.eval_episodes, all_args.n_eval_rollout_threads, all_args.eval_interval) < 1:
            parser.error("验证回合数、线程数和间隔须为正数")
    if all_args.heading_measurement_noise and all_args.use_eval:
        all_args.heading_eval_noise_std_m_deg_mps = HEADING_MEASUREMENT_NOISE_STD[HEADING_EVAL_NOISE_LEVEL]
    return all_args


def main(args):
    parser = get_config() # 构建解析器
    all_args = parse_args(args, parser) # 解析命令行参数
    if all_args.heading_measurement_noise:
        logging.info("Heading 测量噪声（高度 m、航向 °、前向速度 m/s）：标准差 %s，档位概率 %s",
                     HEADING_MEASUREMENT_NOISE_STD, HEADING_MEASUREMENT_NOISE_PROBABILITIES)
        if all_args.use_eval:
            logging.info("Heading 定期评估使用第二档测量噪声，标准差 %s",
                         all_args.heading_eval_noise_std_m_deg_mps)

    # seed
    np.random.seed(all_args.seed) # 设置NumPy随机种子，all_args.seed固定时，每次生成的随机数一样
    random.seed(all_args.seed) # 设置Python内置随机种子
    torch.manual_seed(all_args.seed) # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed_all(all_args.seed) # 设置PyTorch 所有GPU随机种子

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
            "render_mode": all_args.render_mode,
            "swanlab_run": swanlab_run,
        }

        if all_args.env_name == "MultipleCombat":
            Runner = ShareJSBSimRunner
        elif all_args.use_selfplay:
            from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            from runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)
        runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
    """
    sys.argv[1:]指命令行中第二个参数到最后一个参数
    sys.argv[0]指第一个参数，一般是python文件
    例如：python train.py --model CNN --size 128 ，"train.py"就是sys.argv[0]

    断点续训用法：
    1. 正常训练：
       python train_jsbsim.py ...
    2. 断点续训：
       python train_jsbsim.py --model-dir <上次训练的结果目录> ...
       只需指定--model-dir参数，程序会自动加载该目录下的actor_latest.pt和critic_latest.pt继续训练。
    """
