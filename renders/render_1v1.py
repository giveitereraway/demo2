import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True # 使用先验知识
    
def _t2n(x): # 将PyTorch张量（Tensor）转换为NumPy数组（ndarray）
    return x.detach().cpu().numpy()

num_agents = 2
render = True
ego_policy_index = 1040
enm_policy_index = 0
episode_rewards = 0
ego_run_dir = "E:/clone/demo2/scripts/results/SingleCombat/1v1/NoWeapon/TacticalHierarchySelfplay/ppo/1v1_follow_tactical_demo2/wandb/offline-run-20260602_155345-wfdbhlmv/files"
enm_run_dir = "E:/clone/demo1/scripts/results/SingleCombat/1v1/NoWeapon/TacticalHierarchySelfplay/ppo/1v1_tactical_hierarchy_2/wandb/offline-run-20260516_131027-v6k42xjz/files"
#experiment_name = ego_run_dir.split('/')[-4] # ego_run_dir路径中的倒数第4个目录
experiment_name = "selfplay"

env = SingleCombatEnv("1v1/NoWeapon/TacticalHierarchySelfplay")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval() # 设置为验证模式，不使用Dropout和BatchNorm，相当于.train(False)
enm_policy.eval()
#ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt")) # 将预训练的参数权重加载到新的模型之中
#enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))
ego_policy.load_state_dict(torch.load(ego_run_dir + "/actor_latest.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + "/actor_latest.pt"))


print("Start render")
obs = env.reset()
if render:
    env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1)) # 2 // 2 = 0,形状为（1，1）的数组
enm_obs =  obs[num_agents // 2:, :]
ego_obs =  obs[:num_agents // 2, :] # obs[:1, :],如果obs的形状是(n, m)，那么结果就是一个形状为(1, m)的数组
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True) # 调用PPOActor.forward()
    # torch.nn.Module中，forward()会被自动调用
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    # Obser reward and next obs
    obs, rewards, dones, infos = env.step(actions)
    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    if dones.all():
        print(infos)
        break
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods}")
    enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]

print(episode_rewards)