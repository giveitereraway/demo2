import math

import numpy as np
import torch
import torch.nn as nn
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.temporal_attention import get_memory_encoder
from ..utils.utils import check, get_gard_norm


class PPOTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.policy_loss = getattr(args, "policy_loss", "clip")
        if self.policy_loss not in {"clip", "spo"}:
            raise ValueError("--policy-loss 只支持 clip 或 spo")
        if self.policy_loss == "spo" and (not math.isfinite(self.clip_param) or self.clip_param <= 0):
            raise ValueError("SPO 要求 --clip-param 为有限正数")
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = get_memory_encoder(args) == "gru"
        self.data_chunk_length = args.data_chunk_length

    def ppo_update(self, policy: PPOPolicy, sample):

        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        # Critic 预测的状态价值（values）；当前策略对这些动作的对数概率（action_log_probs）；当前策略的动作分布熵（dist_entropy）
        values, action_log_probs, dist_entropy = policy.evaluate_actions(obs_batch,
                                                                         rnn_states_actor_batch,
                                                                         rnn_states_critic_batch,
                                                                         actions_batch,
                                                                         masks_batch)

        # Obtain the loss function loss=策略损失+价值损失+熵损失
        # 完整动作的联合 log probability 必须保持 [batch, 1]，避免意外广播。
        if (action_log_probs.ndim != 2 or action_log_probs.shape[-1] != 1
                or action_log_probs.shape != old_action_log_probs_batch.shape
                or action_log_probs.shape != advantages_batch.shape):
            raise ValueError("新旧动作 log probability 和优势必须同为 [batch, 1]")
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch) # 新旧策略比
        surr1 = ratio * advantages_batch # 概率比乘以优势函数
        if self.policy_loss == "spo":
            # Xie 等，Simple Policy Optimization，ICML 2025，公式 (16)。
            surrogate = surr1 - advantages_batch.abs() / (2 * self.clip_param) * (ratio - 1).square()
        else:
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
            surrogate = torch.min(surr1, surr2)
        policy_loss = torch.sum(surrogate, dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2) #returns_batch是回报(累计奖励),values是Critic预测的状态价值
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()

        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

        # Optimize the loss function
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['ratio_deviation'] = 0
        train_info['approx_kl'] = 0
        train_info['ratio_out_of_bounds'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                data_generator = ReplayBuffer.feed_forward_generator(buffer, self.num_mini_batch)

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()
                with torch.no_grad():
                    train_info['ratio_deviation'] += (ratio - 1).abs().mean().item()
                    train_info['approx_kl'] += ((ratio - 1) - ratio.log()).mean().item()
                    train_info['ratio_out_of_bounds'] += (
                        (ratio < 1 - self.clip_param) | (ratio > 1 + self.clip_param)
                    ).float().mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        # 使用 rollout 时的价值预测；回报无方差时解释方差没有定义，记录 NaN。
        buffers = [buffer] if isinstance(buffer, ReplayBuffer) else buffer
        returns = np.concatenate([b.returns[:-1].reshape(-1) for b in buffers])
        predictions = np.concatenate([b.value_preds[:-1].reshape(-1) for b in buffers])
        return_variance = np.var(returns)
        train_info['explained_variance'] = float(1 - np.var(returns - predictions) / return_variance) \
            if return_variance > 0 else float('nan')

        return train_info
