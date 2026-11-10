import json
import time
import torch
import logging
import numpy as np
from typing import List
from pathlib import Path
from .base_runner import Runner, ReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class JSBSimRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay

        # policy & algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        self.warmup() # warmup() 为后续的 collect() 和 insert() 方法提供基础：collect() 方法会使用 self.buffer.obs[step] 来获取当前观测；insert() 方法会向缓冲区添加新的经验数据

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads # 1041步

        for episode in range(episodes):

            heading_turns_list = []

            """PPO算法需要“采集一段轨迹”后,统一进行训练,而不是每一步都立刻更新网络
            只有收集到足够多的经验数据后,才能用这些数据来计算Return、Advantag,并进行批量训练"""
            for step in range(self.buffer_size):
                # Sample actions 采样动作，获取状态价值
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs 获取新观测和奖励
                obs, rewards, dones, infos = self.envs.step(actions)

                # Extra recorded information
                for info in infos:
                    if 'heading_turn_counts' in info:
                        heading_turns_list.append(info['heading_turn_counts'])

                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # insert data into buffer 存储经验数据
                self.insert(data)

            # compute return and update network
            self.compute() # 计算回报函数
            train_infos = self.train() # 更新策略和价值网络

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episode,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                if len(heading_turns_list):
                    train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                    logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            eval_update = episode + 1 if getattr(self.all_args, "heading_first_target", False) else episode
            if self.use_eval and ((eval_update % self.eval_interval == 0 and eval_update != 0)
                                  or episode == episodes - 1):
                # 短训练也评估最后模型；自博弈首次评估前需有可加载的对手快照。
                if self.use_selfplay and not self.policy_pool:
                    self.save(episode)
                eval_reward = self.eval(self.total_num_steps)
                self.update_eval_checkpoint(eval_reward, episode)

            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout() # 策略网络设置为评估模式
        # 使用 np.concatenate() 将多个线程的观测数据合并为单一数组
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]), # (32, 1, 12) → (32, 12)
                                      np.concatenate(self.buffer.rnn_states_actor[step]), # (32, 1, 1, 128) → (32, 1, 128)
                                      np.concatenate(self.buffer.rnn_states_critic[step]), # (32, 1, 1, 128) → (32, 1, 128)
                                      np.concatenate(self.buffer.masks[step])) # (32, 1, 1) → (32, 1)
        # split parallel data [N*M, shape] => [N, M, shape]
        # 将一维数组按n_rollout_threads分割成多个子数组，将分割后的数组列表转换为二维NumPy数组
        values = np.array(np.split(_t2n(values), self.n_rollout_threads)) # (32, 1) → (32, 1, 1) 32 个环境 × 1 个智能体 × 1 维价值
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads)) # (32, 5) → (32, 1, 5) 32 个环境 × 1 个智能体 × 5 维动作
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads)) # (32, 1) → (32, 1, 1) 32 个环境 × 1 个智能体 × 1 维对数概率
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads)) # (32, 1, 128) → (32, 1, 1, 128) 32 个环境 × 1 个智能体 × 1 层 GRU × 128 维隐状态
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads)) # (32, 1, 128) → (32, 1, 1, 128) 32 个环境 × 1 个智能体 × 1 层 GRU × 128 维隐状态
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        tracking = getattr(self.all_args, "heading_first_target", False)
        if tracking:
            seed = self.all_args.heading_target_eval_seed
            ranks = np.arange(self.n_eval_rollout_threads)
            # 每个环境的回合配额固定，避免按完成先后选择验证场景。
            remaining = (self.eval_episodes + self.n_eval_rollout_threads - 1 - ranks) // self.n_eval_rollout_threads
            episode_ok = np.ones(self.n_eval_rollout_threads, dtype=bool)
            passed = 0
            limits = np.asarray(self.all_args.heading_error_limits)
            eval_obs = self.eval_envs.reset(seeds=[seed + int(rank) for rank in ranks])
        else:
            eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        self.timestamp = 0 # use for tacview real time render 
        if self.render_mode == "real_time" and self.tacview: #reconnect tacview to clear the telemetry
            print("reconnect tacview.....")
            self.tacview.reconnect()
        
        while total_episodes < self.eval_episodes:

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            # real render with tacview
            if self.render_mode == "real_time" and self.tacview:
                render_data = [f"#{self.timestamp:.2f}\n"]
                for sim in self.eval_envs.envs[0]._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                for sim in self.eval_envs.envs[0]._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                render_data_str = "".join(render_data)
                try:
                    self.tacview.send_data_to_client(render_data_str)
                except Exception as e:
                    logging.error(f"Tacview rendering error: {e}")
            self.timestamp += 0.2   # step 0.2s
            
            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            counted_dones = eval_dones_env
            if tracking:
                counted_dones = eval_dones_env & (remaining > 0)
                for rank, info in enumerate(eval_infos):
                    errors = info.get("heading_tracking_errors")
                    if errors is not None:
                        episode_ok[rank] &= bool(np.all(np.abs(errors[:limits.size]) <= limits))
                    if counted_dones[rank]:
                        passed += int(episode_ok[rank] and errors is not None
                                      and info.get("heading_tracking_timeout", False))
                remaining[counted_dones] -= 1
                episode_ok[eval_dones_env] = True
            total_episodes += np.sum(counted_dones)
            eval_episode_rewards.append(eval_cumulative_rewards[counted_dones])
            eval_cumulative_rewards[eval_dones_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        if tracking:
            eval_infos["heading_tracking_pass_rate"] = passed / self.eval_episodes
            self.record_heading_first_target(total_num_steps, passed)
        mean_reward = float(np.concatenate(eval_episode_rewards).mean(axis=1).mean())
        eval_infos['eval_average_episode_rewards'] = mean_reward
        logging.info(" eval average episode rewards: %s", mean_reward)
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")
        return mean_reward

    def record_heading_first_target(self, total_num_steps, passed):
        args = self.all_args
        pass_rate = passed / self.eval_episodes
        logging.info("heading 跟踪验证：步数 %d，通过 %d/%d，要求 %.1f%%",
                     total_num_steps, passed, self.eval_episodes, 100 * args.heading_target_pass_rate)
        path = Path(self.save_dir) / "heading_first_target.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))["first_env_steps"] is not None:
            return
        reached = pass_rate >= args.heading_target_pass_rate
        if reached:
            self.save_checkpoint("first_tracking_target")
            logging.info("保存 heading 首次达标模型：训练环境步数 %d", total_num_steps)
        record = {
            "first_env_steps": int(total_num_steps) if reached else None,
            "evaluated_env_steps": int(total_num_steps),
            "pass_rate": pass_rate,
            "passed_episodes": passed,
            "eval_episodes": self.eval_episodes,
            "eval_seed": args.heading_target_eval_seed,
            "eval_rollout_threads": self.n_eval_rollout_threads,
            "eval_interval": self.eval_interval,
            "error_limits_deg_m_mps": args.heading_error_limits,
            "required_pass_rate": args.heading_target_pass_rate,
            "training_seed": args.seed,
            "policy_loss": args.policy_loss,
        }
        if hasattr(args, "heading_eval_noise_std_m_deg_mps"):
            record["eval_measurement_noise_std_m_deg_mps"] = args.heading_eval_noise_std_m_deg_mps
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @torch.no_grad()
    def render(self):
        logging.info(f"\nStart render, render mode is {self.render_mode} ... ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode=self.render_mode, filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi',tacview=self.tacview)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        super().save()
