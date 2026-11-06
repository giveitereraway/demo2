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

from envs.JSBSim.human_agent.HumanAgent import HumanAgent
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from envs.JSBSim.tasks.heading_task import HeadingTask  

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
    
    tacview = Tacview() # 启动服务器
    
    env = SingleControlEnv(all_args.scenario_name)  # 初始化环境
    
    # 初始化 HumanAgent，直接传递 env
    agent = HumanAgent(env)  # 不需要显式传递 task，env 中已包含

   # 重置环境，获取初始观察状态
    observation = agent.reset()

    
    done = False  # 初始化 done 为 False，表示还没有结束
    timestamp = 0 # use for tacview real time render 
    while not done:
        try:
            # 执行一次 step
            observation, reward, done, info = agent.step()  # 确保调用 step 方法
            
            # real render with tacview 实时渲染
            render_data = [f"#{timestamp:.2f}\n"]
            for sim in env._jsbsims.values():
                log_msg = sim.log()
                if log_msg is not None:
                    render_data.append(log_msg + "\n")

            render_data_str = "".join(render_data)
            try:
                tacview.send_data_to_client(render_data_str)
            except Exception as e:
                logging.error(f"Tacview rendering error: {e}")
                # 打印调用栈信息
                logging.error("".join(traceback.format_exc()))

            timestamp += 0.2  # step 0.2s
            # print(timestamp)

            # 可以加入适当的延时控制，避免过快执行
            time.sleep(0.1)  # 设置每一步之间的间隔时间（单位：秒），根据需求调整

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            # 打印完整的调用栈信息
            logging.error("".join(traceback.format_exc()))
            break  # 可选择退出循环

if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    
    logging.basicConfig(
        level=logging.DEBUG,               # 设置日志级别为 DEBUG，意味着记录所有级别的日志
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        filename='debug.log',              # 指定日志文件名
        filemode='w'                        # 'w'表示写入模式，'a'表示追加模式
    )
    
    main(sys.argv[1:])