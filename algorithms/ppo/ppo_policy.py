import json
from pathlib import Path

import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.obs_space, self.device)

        # 结构单独保存，保留裸 state_dict 格式与旧 GRU checkpoint 的兼容性。
        self.model_config = {
            "memory_encoder": self.actor.memory_encoder,
            "state_shape": list(self.actor.state_shape),
            "observation_space": str(obs_space),
            "action_space": str(act_space),
            "hidden_size": args.hidden_size,
            "act_hidden_size": args.act_hidden_size,
            "activation_id": args.activation_id,
            "use_feature_normalization": args.use_feature_normalization,
            "use_prior": args.use_prior,
        }
        if self.actor.memory_encoder == "temporal_attention":
            self.model_config.update(history_length=args.history_length,
                                     attention_hidden_size=args.attention_hidden_size,
                                     attention_heads=args.attention_heads)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def save_model_config(self, directory):
        with (Path(directory) / "model_config.json").open("w", encoding="utf-8") as stream:
            json.dump(self.model_config, stream, ensure_ascii=False, indent=2)

    def validate_model_config(self, directory):
        path = Path(directory) / "model_config.json"
        if not path.exists():
            if self.actor.memory_encoder == "temporal_attention":
                raise ValueError("注意力模型缺少 model_config.json，无法核对网络结构")
            return
        with path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved != self.model_config:
            raise ValueError("checkpoint 网络配置不匹配，请核对编码器、窗口、网络维数和观测/动作空间")

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
