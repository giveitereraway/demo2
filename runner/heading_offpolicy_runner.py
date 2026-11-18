import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from algorithms.heading_baselines import BranchingDQN, FactorizedDiscreteSAC, HeadingReplay
from .jsbsim_runner import JSBSimRunner


class HeadingOffPolicyRunner(JSBSimRunner):
    def load(self):
        args = self.all_args
        self.num_agents = self.envs.num_agents
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        policy_type = BranchingDQN if self.algorithm_name == "bdq" else FactorizedDiscreteSAC
        self.policy = policy_type(args, self.obs_space, self.act_space, self.device)
        replay_seed, exploration_seed = np.random.SeedSequence(args.seed).spawn(2)
        self.replay = HeadingReplay(args.offpolicy_replay_size, self.obs_space.shape, self.act_space.shape, replay_seed)
        self.exploration_rng = np.random.default_rng(exploration_seed)
        self.model_config = {
            "algorithm_name": self.algorithm_name,
            "memory_encoder": self.policy.actor.memory_encoder,
            "state_shape": list(self.policy.actor.state_shape),
            "observation_space": str(self.obs_space),
            "action_space": str(self.act_space),
            "hidden_size": args.hidden_size,
            "act_hidden_size": args.act_hidden_size,
            "activation_id": args.activation_id,
            "use_feature_normalization": args.use_feature_normalization,
            "use_prior": False,
        }
        if self.model_dir is not None:
            self.restore()

    @torch.no_grad()
    def collect_actions(self, obs, steps):
        args = self.all_args
        if steps < args.offpolicy_warmup_steps:
            return self.exploration_rng.integers(self.act_space.nvec, size=(len(obs), *self.act_space.shape))
        actions = self.policy.actor.actions(obs, deterministic=False).cpu().numpy()
        if self.algorithm_name == "bdq":
            progress = min((steps - args.offpolicy_warmup_steps) / args.bdq_exploration_steps, 1.0)
            random_mask = self.exploration_rng.random(actions.shape) < 1.0 - 0.95 * progress
            random_actions = self.exploration_rng.integers(self.act_space.nvec, size=actions.shape)
            actions = np.where(random_mask, random_actions, actions)
        return actions

    def run(self):
        args = self.all_args
        rounds = self.heading_training_rounds()
        obs = self.envs.reset()[:, 0]
        self.total_num_steps = 0
        next_update = max(args.offpolicy_warmup_steps, args.offpolicy_batch_size, args.offpolicy_train_freq)
        gradient_updates = 0
        episode_returns = np.zeros(self.n_rollout_threads)
        for episode in range(rounds):
            round_start = time.perf_counter()
            train_stats = {}
            completed_returns = []
            for _ in range(self.buffer_size):
                actions = self.collect_actions(obs, self.total_num_steps)
                next_obs, rewards, done, _ = self.envs.step(actions[:, None, :])
                next_obs = next_obs[:, 0]
                rewards, done = rewards.reshape(-1), done.reshape(-1).astype(bool)
                # 回合结束时包装器返回新回合首帧，done 屏蔽对应转移的自举项。
                self.replay.insert(obs, actions, rewards, next_obs, done)
                obs = next_obs
                self.total_num_steps += self.n_rollout_threads
                episode_returns += rewards
                completed_returns.extend(episode_returns[done])
                episode_returns[done] = 0
                while self.total_num_steps >= next_update:
                    train_stats = self.policy.train_batch(self.replay.sample(args.offpolicy_batch_size, self.device))
                    next_update += args.offpolicy_train_freq
                    gradient_updates += 1

            metrics = {"gradient_updates": gradient_updates, **self.heading_training_metrics(round_start)}
            metrics.update({key: value.item() for key, value in train_stats.items()})
            if completed_returns:
                metrics["average_episode_rewards"] = np.mean(completed_returns)
            last_round = episode + 1 == rounds
            if episode % self.log_interval == 0 or last_round:
                logging.info("Heading %s 第 %d/%d 轮，环境步数 %d，梯度更新 %d，训练耗时 %.1f 秒",
                             self.algorithm_name, episode + 1, rounds, self.total_num_steps,
                             gradient_updates, metrics["training_elapsed_seconds"])
                self.log_info(metrics, self.total_num_steps)
            if self.use_eval and ((episode + 1) % self.eval_interval == 0 or last_round):
                reward = self.eval(self.total_num_steps)
                self.update_eval_checkpoint(reward, episode)
            if episode % self.save_interval == 0 or last_round:
                self.save(episode)

    def save_checkpoint(self, tag):
        directory = Path(self.save_dir)
        (directory / "model_config.json").write_text(json.dumps(self.model_config, indent=2), encoding="utf-8")
        torch.save(self.policy.actor.state_dict(), directory / f"actor_{tag}.pt")
        if self.algorithm_name == "sac":
            torch.save({"critic": self.policy.critic.state_dict(), "log_alpha": self.policy.log_alpha.detach()},
                       directory / f"critic_{tag}.pt")

    def restore(self):
        directory = Path(self.model_dir)
        saved_config = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
        if saved_config != self.model_config:
            raise ValueError("Heading 离策略 checkpoint 配置不匹配，请核对算法、网络和动作空间")
        self.policy.actor.load_state_dict(torch.load(directory / "actor_latest.pt", map_location=self.device,
                                                     weights_only=True))
        if self.algorithm_name == "bdq":
            self.policy.target.load_state_dict(self.policy.actor.state_dict())
        else:
            state = torch.load(directory / "critic_latest.pt", map_location=self.device, weights_only=True)
            self.policy.critic.load_state_dict(state["critic"])
            self.policy.target.load_state_dict(state["critic"])
            with torch.no_grad():
                self.policy.log_alpha.copy_(state["log_alpha"])
