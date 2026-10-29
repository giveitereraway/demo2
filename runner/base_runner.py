import os
import sys
import wandb
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.utils.buffer import ReplayBuffer
import logging

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.render_mode = config['render_mode']
        
        # Tacview render obj
        self.tacview = None
        if self.render_mode == "real_time":
            from runner.tacview import Tacview
            self.tacview = Tacview()

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.buffer_size = self.all_args.buffer_size
        self.use_wandb = self.all_args.use_wandb

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes

        # 最低训练量之前也保存最佳模型，但不累计无提升次数。
        self.use_early_stop = getattr(self.all_args, 'use_early_stop', False)
        min_steps = getattr(self.all_args, 'early_stop_min_steps', None)
        self.early_stop_min_steps = int(self.num_env_steps * 0.2 if min_steps is None else min_steps)
        self.early_stop_patience = getattr(self.all_args, 'early_stop_patience', 10)
        if self.use_early_stop:
            if not self.use_eval or self.eval_envs is None:
                raise ValueError("早停必须开启 --use-eval 并提供验证环境")
            if self.early_stop_min_steps < 0 or self.early_stop_patience < 1:
                raise ValueError("早停最低步数必须非负，耐心次数必须大于 0")
        if self.use_eval and (self.eval_interval < 1 or self.eval_episodes < 1):
            raise ValueError("验证间隔和验证回合数必须大于 0")
        self.best_eval_reward = -np.inf
        self.no_improvement_evals = 0

        # dir
        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.save_dir = str(self.run_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.load()

    def load(self):
        # algorithm
        if self.algorithm_name == "ppo":
            from ..algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from ..algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args,
                             self.envs.observation_space,
                             self.envs.action_space,
                             device=self.device)
        self.trainer = Trainer(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args,
                                   self.envs.observation_space,
                                   self.envs.action_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def rollout(self):
        raise NotImplementedError

    @torch.no_grad() # 使得在本函数运行的代码都不计算梯度，也不会进行反向传播
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def train(self):
        self.policy.prep_training()
        train_infos = self.trainer.train(self.policy, self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        self.save_checkpoint("latest")

    def save_checkpoint(self, tag):
        torch.save(self.policy.actor.state_dict(), os.path.join(self.save_dir, f"actor_{tag}.pt"))
        torch.save(self.policy.critic.state_dict(), os.path.join(self.save_dir, f"critic_{tag}.pt"))

    def check_early_stopping(self, eval_reward, total_num_steps):
        """按验证平均回合奖励保存最佳模型，并判断是否满足工程早停条件。"""
        eval_reward = float(np.mean(eval_reward))
        if not np.isfinite(eval_reward):
            logging.warning("验证奖励不是有限值，跳过本次最佳模型和早停判断")
            return False
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.no_improvement_evals = 0
            self.save_checkpoint("best")
            logging.info(f"保存最佳 checkpoint：步数 {total_num_steps}，验证奖励 {eval_reward:.6g}")
        elif self.use_early_stop and total_num_steps >= self.early_stop_min_steps:
            self.no_improvement_evals += 1

        should_stop = (self.use_early_stop
                       and total_num_steps >= self.early_stop_min_steps
                       and self.no_improvement_evals >= self.early_stop_patience)
        if should_stop:
            logging.info(f"触发早停：步数 {total_num_steps}，连续 {self.no_improvement_evals} 次验证无提升，"
                         f"最佳验证奖励 {self.best_eval_reward:.6g}")
        return should_stop

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_info(self, infos, total_num_steps):
        if self.use_wandb:
            try:
                wandb.log(infos, step=total_num_steps)
            except Exception as e:
                logging.warning(f"WandB logging failed, disable WandB logging and continue training: {e}")
                self.use_wandb = False
        else:
            pass
        
    def render_with_tacview(self, data):
        """
        Send data to Tacview for real-time rendering.
        :param data: The data to be rendered, which can be a string or a specific structure.
        """
        if self.tacview:
            try:
                self.tacview.send_data_to_client(data)
            except Exception as e:
                logging.error(f"Tacview rendering error: {e}")
