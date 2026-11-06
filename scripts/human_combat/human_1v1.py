#!/usr/bin/env python
import sys
import os
import time
import traceback
import torch
import random
import logging
import numpy as np
import setproctitle


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from runner.experiment_tracking import create_run_dir, track_swanlab
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv

from envs.JSBSim.human_agent.HumanAgent_1v1 import HumanAgent_1v1
from envs.JSBSim.human_agent.FollowAgent import FollowAgent
from envs.JSBSim.human_agent.PPO_FollowAgent import PPO_FollowAgent
from envs.JSBSim.test.test_baseline_use_env import BaselineAgent, PursueAgent
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from envs.JSBSim.human_task.HumanSingleCombatTask import HumanSingleCombatTask

from scripts.train.train_jsbsim import parse_args
from runner.tacview import Tacview
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

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    run_dir = create_run_dir(all_args)
    with track_swanlab(all_args, run_dir, job_type="human_loop"):
        run_simulation(all_args, device)


def run_simulation(all_args, device):
    # 你可以在这里传入你的配置或初始化环境

    tacview = Tacview()

    env = SingleCombatEnv(all_args.scenario_name)  # 初始化环境

    # 初始化 HumanAgent，直接传递 env
    human_agent_id = 0
    ai_agent_id = 1
    print("Initializing HumanAgent...")
    human_agent = HumanAgent_1v1(env, human_agent_id)
    #ai_agent = PursueAgent(ai_agent_id)
    #ai_agent = FollowAgent(env, ai_agent_id, human_agent_id)
    print("Initializing AI Agent...")
    ai_agent = PPO_FollowAgent(ai_agent_id, env, all_args, device)
    print("Agent initialization complete.")

   # 重置环境，获取初始观察状态
    obs = env.reset()
    #human_observation = human_agent.reset()
    #ai_observation = ai_agent.reset()

    done = np.array([False, False])  # 初始化 done 为 [False, False]（两个智能体）
    timestamp = 0 # use for tacview real time render
    
    print("Environment reset complete. Starting main loop...")
    while not np.array(done).all():
        try:
            # 获取动作
            human_action = human_agent.get_action()
            ai_action = ai_agent.get_action(env, env.task)
            ai_action = np.append(ai_action,0)
            print("AI Action:", ai_action)
            # 组合动作，顺序要和env.ego_ids + env.enm_ids一致
            actions = np.zeros((2, 4), dtype=np.float32)
            actions[0] = human_action
            actions[1] = ai_action
            # 执行一次 step
            obs, reward, done, info = env.step(actions)
            #env.render(mode="txt", filepath="agent_follow_human.acmi")
            #human_observation, human_reward, human_done, human_info = human_agent.step()
            #ai_observation, ai_reward, ai_done, ai_info = ai_agent.step()

            # real render with tacview
            render_data = [f"#{timestamp:.2f}\n"]
            for sim in env._jsbsims.values(): # _jsbsims是字典,[键是飞机标识符，值是飞机仿真器实例]
                log_msg = sim.log()
                if log_msg is not None:
                    render_data.append(log_msg + "\n")

            render_data_str = "".join(render_data)
            """with open ("agent_follow_human", "a") as f:
                f.write(render_data_str)"""
            try:
                tacview.send_data_to_client(render_data_str)
            except Exception as e:
                logging.error(f"Tacview rendering error: {e}")
                # 打印调用栈信息
                logging.error("".join(traceback.format_exc()))

            timestamp += 0.01  # step 0.2s
            # print(timestamp)

            # 可以加入适当的延时控制，避免过快执行
            time.sleep(0)  # 设置每一步之间的间隔时间（单位：秒），根据需求调整

        except KeyboardInterrupt:
            print("\nUser interrupted the program. Exiting...")
            break
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            # 打印完整的调用栈信息
            logging.error("".join(traceback.format_exc()))
            break  # 可选择退出循环
    
    print("\nSimulation ended. Final done status:", done)

if __name__ == "__main__":
    # 同时输出到控制台和文件
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log', mode='w'),
            logging.StreamHandler()  # 添加控制台输出
        ]
    )

    main(sys.argv[1:])