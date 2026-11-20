import torch
import logging
import json
import shutil
from pathlib import Path
import numpy as np
from typing import List
from .base_runner import Runner, ReplayBuffer
from .jsbsim_runner import JSBSimRunner


def _t2n(x):
    return x.detach().cpu().numpy()


class SelfplayJSBSimRunner(JSBSimRunner):

    def __init__(self, config):
        self.no_improvement_evals = 0
        self.convergence_recorded = False
        super().__init__(config)

    def load(self):
        self.use_selfplay = self.all_args.use_selfplay 
        assert self.use_selfplay == True, "Only selfplay can use SelfplayRunner"
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.num_opponents = self.all_args.n_choose_opponents
        assert self.eval_episodes >= self.num_opponents, \
        f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.num_opponents}"
        self.init_elo = self.all_args.init_elo
        self.latest_elo = self.init_elo

        # policy & algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)

        # [Selfplay] allocate memory for opponent policy/data in training
        from algorithms.utils.selfplay import get_algorithm
        self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)
        self.mix_pfsp = self.all_args.selfplay_algorithm == "mix_pfsp"
        if self.mix_pfsp:
            args = self.all_args
            self.selfplay_algo = self.selfplay_algo(args.seed)
            self.mix_version = 0
            self.mix_next_id = 0
            self.mix_eval_rng = np.random.default_rng(args.seed + 1)
            from envs.JSBSim.utils.utils import parse_config
            scene = parse_config(args.scenario_name)
            self.mix_protocol = {"scenario": args.scenario_name,
                                 "use_artillery": getattr(scene, "use_artillery", False),
                                 "score_version": 1}

        assert self.num_opponents <= self.n_rollout_threads, \
            "Number of different opponents({}) must less than or equal to number of training threads({})!" \
            .format(self.num_opponents, self.n_rollout_threads)
        self.policy_pool = {}  # type: dict[str, float]
        self.opponent_policy = [
            Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
            for _ in range(self.num_opponents)]
        self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
        self.opponent_obs = np.zeros_like(self.buffer.obs[0])
        self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
        self.opponent_masks = np.ones_like(self.buffer.masks[0])

        if self.use_eval:
            self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)

        logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}.\n"
                        .format(self.all_args.selfplay_algorithm, self.num_opponents))

        if self.model_dir is not None:
            if not self.mix_pfsp or not self.restore_mix_state():
                self.restore()

    def warmup(self):
        if self.mix_pfsp:
            if not self.policy_pool:
                self.save(-1)
            self.reset_opponent()
            return
        # reset env
        obs = self.envs.reset()
        # [Selfplay] divide ego/opponent of initial obs
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        obs = obs[:, :self.num_agents // 2, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # [Selfplay] get actions of opponent policy
        opponent_actions = np.zeros_like(actions)
        for policy_idx, policy in enumerate(self.opponent_policy):
            env_idx = self.opponent_env_split[policy_idx]
            opponent_action, opponent_rnn_states \
                = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                                np.concatenate(self.opponent_rnn_states[env_idx]),
                                np.concatenate(self.opponent_masks[env_idx]))
            opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
            self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
        actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # [Selfplay] divide ego/opponent of collecting data
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        self.opponent_masks = masks[:, self.num_agents // 2:, ...]
        self.opponent_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        
        obs = obs[:, :self.num_agents // 2, ...]
        actions = actions[:, :self.num_agents // 2, ...]
        rewards = rewards[:, :self.num_agents // 2, ...]
        masks = masks[:, :self.num_agents // 2, ...]

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        self.policy.prep_rollout()
        total_episodes = 0
        episode_rewards, opponent_episode_rewards = [], []
        episode_scores = []
        cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
        opponent_cumulative_rewards= np.zeros_like(cumulative_rewards)

        # [Selfplay] Choose opponent policy for evaluation
        if self.mix_pfsp:
            ids, probabilities = self.selfplay_algo.probabilities(self.policy_pool, self.mix_version)
            eval_choose_opponents = self.mix_eval_rng.choice(ids, size=self.num_opponents, p=probabilities).tolist()
        else:
            eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.num_opponents)]
        quotas = np.full(self.num_opponents, self.eval_episodes // self.num_opponents)
        quotas[:self.eval_episodes % self.num_opponents] += 1
        boundaries = np.concatenate(([0], np.cumsum(quotas)))
        logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")

        eval_cur_opponent_idx = 0
        # use for tacview's timestamp
        self.timestamp = 0
        while total_episodes < self.eval_episodes:

            # [Selfplay] Load opponent policy
            if total_episodes == boundaries[eval_cur_opponent_idx]:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(torch.load(
                    Path(self.save_dir) / f"actor_{policy_idx}.pt", map_location=self.device, weights_only=True))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes + 1}/{self.eval_episodes})")

                # reset obs/rnn/mask
                obs = self.eval_envs.reset()
                cumulative_rewards.fill(0)
                opponent_cumulative_rewards.fill(0)
                masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
                rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
                opponent_obs = obs[:, self.num_agents // 2:, ...]
                obs = obs[:, :self.num_agents // 2, ...]
                opponent_masks = np.ones_like(masks, dtype=np.float32)
                opponent_rnn_states = np.zeros_like(rnn_states, dtype=np.float32)

            # [Selfplay] get actions
            actions, rnn_states = self.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks), deterministic=True)
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_eval_rollout_threads))

            opponent_actions, opponent_rnn_states \
                = self.eval_opponent_policy.act(np.concatenate(opponent_obs),
                                                np.concatenate(opponent_rnn_states),
                                                np.concatenate(opponent_masks), deterministic=True)
            opponent_rnn_states = np.array(np.split(_t2n(opponent_rnn_states), self.n_eval_rollout_threads))
            opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
            actions = np.concatenate((actions, opponent_actions), axis=1)

            # Obser reward and next obs
            obs, eval_rewards, dones, eval_infos = self.eval_envs.step(actions)
            dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
            # 同一步可能结束多局，只接收当前对手配额内的结果。
            remaining = boundaries[eval_cur_opponent_idx] - total_episodes
            counted = np.flatnonzero(dones_env)[:remaining]
            total_episodes += len(counted)
            if self.mix_pfsp:
                episode_scores.extend(eval_infos[index]["selfplay_result"]["score"] for index in counted)

            # [Selfplay] Reset obs, masks, rnn_states
            opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            # 仅在新回合第一帧清空历史，后续帧恢复记忆；双方分别维护 mask。
            masks.fill(1)
            opponent_masks.fill(1)
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), *masks.shape[1:]), dtype=np.float32)
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states.shape[1:]), dtype=np.float32)

            opponent_masks[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_masks.shape[1:]), dtype=np.float32)
            opponent_rnn_states[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_rnn_states.shape[1:]), dtype=np.float32)

            # [Selfplay] Get rewards
            opponent_rewards = eval_rewards[:, self.num_agents//2:, ...]
            opponent_cumulative_rewards += opponent_rewards
            opponent_episode_rewards.append(opponent_cumulative_rewards[counted])
            opponent_cumulative_rewards[dones_env == True] = 0

            eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
            cumulative_rewards += eval_rewards
            episode_rewards.append(cumulative_rewards[counted])
            cumulative_rewards[dones_env == True] = 0
            
            # Tacview real time render
            if self.render_mode == "real_time":
                render_data = [f"#{self.timestamp:.2f}\n"]
                for sim in self.eval_envs.envs[0]._jsbsims.values():
                    log_msg = sim.log()
                    
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                        
                render_data_str = "".join(render_data)
                self.tacview.send_data_to_client(render_data_str)
            self.timestamp += 0.2

        # Compute average episode rewards
        episode_rewards = np.concatenate(episode_rewards) # shape (self.eval_episodes, self.num_agents, 1)
        episode_rewards = episode_rewards.squeeze(-1).mean(axis=-1) # shape: (self.eval_episodes,)
        eval_average_episode_rewards = np.array([block.mean() for block in np.split(episode_rewards, boundaries[1:-1])])

        opponent_episode_rewards = np.concatenate(opponent_episode_rewards)
        opponent_episode_rewards = opponent_episode_rewards.squeeze(-1).mean(axis=-1)
        opponent_average_episode_rewards = np.array([
            block.mean() for block in np.split(opponent_episode_rewards, boundaries[1:-1])])

        # Update elo
        '''
        Elo Rating Algorithm:
            Players with higher ELO ratings have a higher probability of winning a game than players with lower ELO ratings.
            After each game, the ELO rating of players is updated.
            If a player with a higher ELO rating wins, only a few points are transferred from the lower-rated player.
            However if the lower-rated player wins, then the transferred points from a higher-rated player are far greater.

        ego_elo / opponent_elo: Elo scores for each agent
        expected_score: Probability of winning of `ego` agent, i.e. when ego_elo >> opponent_elo, expected_score -> 1.
        actual_score: Outcome for current match, 1: ego win, 0: ego lose, 0.5: tie.

        Update Rule:
            Ra = Ra + K * (outcome - Pa)
            Rb = Rb + K * ((1 - outcome) - Pb) = Rb + K * ((1 - outcome) - (1 - Pa)) = Rb + K * (Pa - outcome)
        '''
        #ego_elo = np.array([self.latest_elo for _ in range(self.n_eval_rollout_threads)])  # 当前策略的 Elo
        ego_elo = np.array([self.latest_elo for _ in range(self.num_opponents)])  # 当前策略的 Elo
        opponent_elo = np.array([self.policy_pool[key] for key in eval_choose_opponents])  # 对手的 Elo
        expected_score = 1 / (1 + 10 ** ((opponent_elo - ego_elo) / 400)) # 期望得分

        # 计算实际得分
        if self.mix_pfsp:
            actual_score = np.array([block.mean() for block in np.split(episode_scores, boundaries[1:-1])])
        else:
            diff = eval_average_episode_rewards - opponent_average_episode_rewards
            actual_score = np.where(diff > 100, 1.0, np.where(abs(diff) < 100, 0.5, 0.0))

        K = 32  # K 因子，决定 Elo 变化的幅度
        update_opponent_elo = opponent_elo + K * (expected_score - actual_score) # 更新对手的 Elo
        for i, key in enumerate(eval_choose_opponents):
            self.policy_pool[key] = float(update_opponent_elo[i])
        update_ego_elo = ego_elo + K * (actual_score - expected_score) # 更新当前策略的 Elo
        self.latest_elo = float(update_ego_elo.mean())

        # Logging
        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = eval_average_episode_rewards.mean()
        eval_infos['eval_opponent_average_episode_rewards'] = opponent_average_episode_rewards.mean()
        eval_infos['eval_elo_gain'] = (K * (actual_score - expected_score)).mean()
        eval_infos['latest_elo'] = self.latest_elo
        if self.mix_pfsp:
            eval_infos["eval_selfplay_score"] = float(np.mean(episode_scores))
            eval_infos["eval_episodes_completed"] = total_episodes
        logging.info(f" eval ego elo: {ego_elo.mean()} | opponent average elo: {opponent_elo.mean()}\n"
                     f" ego average episode rewards: {eval_infos['eval_average_episode_rewards']}\n"
                     f" opponent average episode rewards: {eval_infos['eval_opponent_average_episode_rewards']}\n"
                     f" elo gain: {(K * (actual_score - expected_score)).mean()} | latest elo score: {self.latest_elo}")
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

        # [Selfplay] Reset opponent for the following training
        if not self.mix_pfsp:
            self.reset_opponent()
        return float(eval_infos['eval_average_episode_rewards'])

    def update_eval_checkpoint(self, eval_reward, episode):
        improved = super().update_eval_checkpoint(eval_reward, episode)
        if self.convergence_recorded:
            return
        if improved is not False:
            self.no_improvement_evals = 0
            return
        self.no_improvement_evals += 1
        if self.no_improvement_evals < 10:
            return

        self.save_checkpoint("converged")
        record = {
            "training_round": episode + 1,
            "total_num_steps": self.total_num_steps,
            "eval_average_episode_rewards": float(eval_reward),
            "best_eval_average_episode_rewards": self.best_eval_reward,
            "no_improvement_evals": self.no_improvement_evals,
        }
        path = Path(self.save_dir) / "selfplay_convergence.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        self.convergence_recorded = True
        logging.info("记录首次收敛：训练轮次 %d，步数 %d", episode + 1, self.total_num_steps)

    def choose_opponent(self):
        if getattr(self, "mix_pfsp", False):
            return self.selfplay_algo.choose(self.policy_pool, self.mix_version)
        return self.selfplay_algo.choose(self.policy_pool)

    def record_mix_results(self, infos, dones):
        """只用随机动作训练对局更新采样统计，保留冻结对手的身份。"""
        finished = np.all(dones.reshape(self.n_rollout_threads, -1), axis=1)
        for opponent_id, env_indices in zip(self.training_opponent_ids, self.opponent_env_split):
            for env_index in env_indices[finished[env_indices]]:
                result = infos[env_index]["selfplay_result"]
                self.selfplay_algo.update(opponent_id, result["score"], self.mix_version)

    def restore_mix_state(self):
        path = Path(self.model_dir) / "mix_pfsp_state.pt"
        if not path.exists():
            logging.info("加载已有 actor/critic 开始新的 Mix-PFSP 阶段，对手统计从空池建立")
            return False
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state["format_version"] != 1 or state["protocol"] != self.mix_protocol:
            raise ValueError("Mix-PFSP 恢复文件的场景或计分口径与当前配置不一致")
        snapshot_ids = list(state["pool"])
        for opponent_id in snapshot_ids:
            if not opponent_id.isdecimal():
                raise ValueError("Mix-PFSP 快照 ID 必须为非负整数")
            source = Path(self.model_dir) / f"actor_{opponent_id}.pt"
            if not source.is_file():
                raise FileNotFoundError(f"恢复对手池缺少快照：{source}")
        self.selfplay_algo.load_state_dict(state["sampler"])
        for opponent_id in snapshot_ids:
            source = Path(self.model_dir) / f"actor_{opponent_id}.pt"
            target = Path(self.save_dir) / source.name
            if source.resolve() != target.resolve():
                if target.exists():
                    raise FileExistsError(f"恢复目录已存在同名快照：{target}")
                shutil.copy2(source, target)
        self.policy.actor.load_state_dict(state["actor"])
        self.policy.critic.load_state_dict(state["critic"])
        self.policy_pool = state["pool"]
        self.latest_elo = state["latest_elo"]
        self.mix_version = state["version"]
        self.mix_next_id = state["next_id"]
        self.mix_eval_rng.bit_generator.state = state["eval_rng"]
        logging.info("恢复 Mix-PFSP 对手池 %d 项及第 %d 轮统计", len(self.policy_pool), self.mix_version)
        return True

    def save(self, episode):
        super().save(episode)
        if getattr(self, "mix_pfsp", False):
            # 单调编号保证恢复训练后继续追加不可变对手快照。
            snapshot = Path(self.save_dir) / f"actor_{self.mix_next_id}.pt"
            while snapshot.exists():
                self.mix_next_id += 1
                snapshot = Path(self.save_dir) / f"actor_{self.mix_next_id}.pt"
            temporary = snapshot.with_suffix(".tmp")
            torch.save(self.policy.actor.state_dict(), temporary)
            temporary.replace(snapshot)
            self.policy_pool[str(self.mix_next_id)] = self.latest_elo
            self.mix_next_id += 1
            state = {
                "format_version": 1, "protocol": self.mix_protocol,
                "actor": self.policy.actor.state_dict(), "critic": self.policy.critic.state_dict(),
                "pool": self.policy_pool, "sampler": self.selfplay_algo.state_dict(),
                "version": self.mix_version, "next_id": self.mix_next_id,
                "latest_elo": self.latest_elo,
                "eval_rng": self.mix_eval_rng.bit_generator.state,
            }
            # 同一原子文件保存学习者权重和采样统计，避免半写入后恢复错配。
            path = Path(self.save_dir) / "mix_pfsp_state.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            return
        policy_actor_state_dict = self.policy.actor.state_dict()
        # [Selfplay] save policy & performance
        torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
        self.policy_pool[str(episode)] = self.latest_elo

    def reset_opponent(self):
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.choose_opponent()
            choose_opponents.append(choose_idx)
            policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{choose_idx}.pt', weights_only=True))
            policy.prep_rollout()
        logging.info(f" Choose opponents {choose_opponents} for training")
        self.training_opponent_ids = choose_opponents

        # clear buffer
        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)
        self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
        self.opponent_masks = np.ones_like(self.opponent_masks)

        # reset env
        obs = self.envs.reset()
        if self.num_opponents > 0:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def render(self):
        idx = self.all_args.render_index
        opponent_idx = self.all_args.render_opponent_index
        dir_list = str(self.run_dir).split('/')
        file_path = '/'.join(dir_list[:dir_list.index('results')+1])
        self.policy.actor.load_state_dict(torch.load(str(self.model_dir)+ f'/actor_{idx}.pt'))
        self.policy.prep_rollout()
        self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.model_dir) + f'/actor_{opponent_idx}.pt',weights_only=True))
        self.eval_opponent_policy.prep_rollout()
        logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
        render_obs = render_obs[:, :self.num_agents // 2, ...]
        render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
        render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            render_opponent_actions, render_opponent_rnn_states \
                = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs),
                                                np.concatenate(render_opponent_rnn_states),
                                                np.concatenate(render_opponent_masks),
                                                deterministic=True)
            render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
            render_opponent_rnn_states = np.expand_dims(_t2n(render_opponent_rnn_states), axis=0)
            render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
            render_obs = render_obs[:, :self.num_agents // 2, ...]
        print(render_episode_rewards)
