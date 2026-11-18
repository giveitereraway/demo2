from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .utils.mlp import MLPBase, MLPLayer


HEADING_BASELINES = ("sac", "bdq")


class HeadingBaselineActor(nn.Module):
    """副翼、升降舵、方向舵和油门分支共享 MLP 特征。"""

    memory_encoder = "mlp"
    state_shape = (1, 1)
    use_prior = False

    def __init__(self, args, obs_space, act_space, device, *, dueling=None):
        super().__init__()
        self.dueling = args.algorithm_name == "bdq" if dueling is None else dueling
        self.action_dims = tuple(int(n) for n in act_space.nvec)
        self.base = MLPBase(obs_space, args.hidden_size, args.activation_id, args.use_feature_normalization)
        self.features = nn.Identity()
        width = self.base.output_size
        if args.act_hidden_size:
            self.features = MLPLayer(width, args.act_hidden_size, args.activation_id)
            width = self.features.output_size
        self.branches = nn.Linear(width, sum(self.action_dims))
        if self.dueling:
            self.value = nn.Linear(width, 1)
        else:
            nn.init.orthogonal_(self.branches.weight, gain=args.gain)
            nn.init.zeros_(self.branches.bias)
        self.to(device)

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.branches.weight.device)
        features = self.features(self.base(obs))
        branches = self.branches(features).split(self.action_dims, dim=-1)
        if self.dueling:
            value = self.value(features)
            return tuple(value + branch - branch.mean(dim=-1, keepdim=True) for branch in branches)
        return branches

    def actions(self, obs, deterministic):
        branches = self(obs)
        if self.dueling or deterministic:
            return torch.stack([branch.argmax(dim=-1) for branch in branches], dim=-1)
        return torch.stack([torch.distributions.Categorical(logits=branch).sample() for branch in branches], dim=-1)

    def act(self, obs, state, masks, deterministic=False):
        actions = self.actions(obs, deterministic)
        return actions, torch.as_tensor(state, device=actions.device)


class HeadingBatch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class HeadingReplay:
    def __init__(self, capacity, obs_shape, action_shape, seed):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self.obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.done = np.empty(capacity, dtype=np.float32)

    def insert(self, obs, actions, rewards, next_obs, done):
        count = len(obs)
        indices = (self.position + np.arange(count)) % self.capacity
        self.obs[indices] = obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_obs[indices] = next_obs
        self.done[indices] = done
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size, device):
        indices = self.rng.integers(self.size, size=batch_size)
        arrays = (self.obs, self.actions, self.rewards, self.next_obs, self.done)
        return HeadingBatch(*(torch.as_tensor(array[indices], device=device) for array in arrays))


def selected_values(branches, actions):
    return torch.cat([branch.gather(1, index[:, None]) for branch, index in zip(branches, actions.T)], dim=-1)


def soft_update(source, target, tau):
    with torch.no_grad():
        for current, delayed in zip(source.parameters(), target.parameters()):
            delayed.lerp_(current, tau)


class HeadingOffPolicy:
    def __init__(self, args, obs_space, act_space, device):
        self.actor = HeadingBaselineActor(args, obs_space, act_space, device)
        self.args = args

    def act(self, obs, state, masks, deterministic=False):
        return self.actor.act(obs, state, masks, deterministic)

    def prep_rollout(self):
        self.actor.eval()


class BranchingDQN(HeadingOffPolicy):
    def __init__(self, args, obs_space, act_space, device):
        super().__init__(args, obs_space, act_space, device)
        self.target = deepcopy(self.actor).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)

    def train_batch(self, batch):
        with torch.no_grad():
            greedy = self.actor.actions(batch.next_obs, deterministic=True)
            next_q = selected_values(self.target(batch.next_obs), greedy).mean(dim=-1)
            # 各分支共用一个由双网络选择和估值的均值 TD 目标。
            target = batch.rewards + self.args.gamma * (1 - batch.done) * next_q
        q = selected_values(self.actor(batch.obs), batch.actions)
        loss = F.mse_loss(q, target[:, None].expand_as(q))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        soft_update(self.actor, self.target, self.args.offpolicy_tau)
        return {"td_loss": loss.detach()}

class FactorizedDiscreteSAC(HeadingOffPolicy):
    def __init__(self, args, obs_space, act_space, device):
        super().__init__(args, obs_space, act_space, device)
        # 两个可加性 Q 网络，每个网络的联合 Q 为各动作分支 Q 的均值。
        self.critic = nn.ModuleList([
            HeadingBaselineActor(args, obs_space, act_space, device, dueling=True) for _ in range(2)
        ])
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(args.sac_initial_alpha), dtype=torch.float32, device=device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = args.sac_target_entropy_ratio * float(np.log(act_space.nvec).sum())

    def _soft_terms(self, obs, critics):
        log_probs = [F.log_softmax(logits, dim=-1) for logits in self.actor(obs)]
        probs = [log_prob.exp() for log_prob in log_probs]
        with torch.no_grad():
            q1, q2 = (critic(obs) for critic in critics)
        # 逐分支取双 Q 下界，使联合策略下的期望可按维度精确求和。
        q = torch.stack([(p * torch.minimum(q_a, q_b)).sum(dim=-1)
                         for p, q_a, q_b in zip(probs, q1, q2)]).mean(dim=0)
        entropy = -torch.stack([(p * log_p).sum(dim=-1) for p, log_p in zip(probs, log_probs)]).sum(dim=0)
        return q, entropy

    def train_batch(self, batch):
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            next_q, next_entropy = self._soft_terms(batch.next_obs, self.target)
            target = batch.rewards + self.args.gamma * (1 - batch.done) * (next_q + alpha * next_entropy)
        critic_loss = sum(F.mse_loss(selected_values(critic(batch.obs), batch.actions).mean(dim=-1), target)
                          for critic in self.critic)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        self.critic_optimizer.step()

        q, entropy = self._soft_terms(batch.obs, self.critic)
        actor_loss = -(q + alpha * entropy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.actor_optimizer.step()

        alpha_loss = self.log_alpha * (entropy.detach().mean() - self.target_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        soft_update(self.critic, self.target, self.args.offpolicy_tau)
        return {"actor_loss": actor_loss.detach(), "critic_loss": critic_loss.detach(),
                "policy_entropy": entropy.detach().mean(), "entropy_alpha": alpha}
