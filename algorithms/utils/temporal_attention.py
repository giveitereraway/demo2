import math

import torch
import torch.nn as nn


def get_memory_encoder(args):
    """显式选择编码器，同时兼容旧命令中反向的 GRU 开关。"""
    encoder = getattr(args, "memory_encoder", None)
    if encoder is None:
        return "gru" if args.use_recurrent_policy else "mlp"
    if encoder != "mlp" and not args.use_recurrent_policy:
        raise ValueError("--memory-encoder 与关闭 GRU 的旧参数 --use-recurrent-policy 冲突，请移除旧参数")
    if encoder == "temporal_attention" and getattr(args, "algorithm_name", "ppo") != "ppo":
        raise ValueError("时序注意力仅支持 PPO，暂不支持 MAPPO")
    return encoder


def get_policy_state_shape(args, obs_space):
    """注意力状态保存原始观测及 padding 标记，沿用现有状态传递接口。"""
    if get_memory_encoder(args) == "temporal_attention":
        if len(obs_space.shape) != 1 or args.history_length < 1:
            raise ValueError("时序注意力要求一维观测，且 --history-length 至少为 1")
        return args.history_length, obs_space.shape[0] + 1
    return args.recurrent_hidden_layers, args.recurrent_hidden_size


class TemporalAttention(nn.Module):
    """单层有限窗口注意力：只计算当前 query，所有 key 都来自当前及过去。"""

    def __init__(self, args, obs_space):
        super().__init__()
        self.state_shape = get_policy_state_shape(args, obs_space)
        self.output_size = args.attention_hidden_size
        heads = args.attention_heads
        if heads < 1 or self.output_size < 1 or self.output_size % heads:
            raise ValueError("注意力维数必须为正数，且能被头数整除")
        obs_size = obs_space.shape[0]
        self.feature_norm = nn.LayerNorm(obs_size) if args.use_feature_normalization else nn.Identity()
        self.embedding = nn.Linear(obs_size, self.output_size)
        self.attention_norm = nn.LayerNorm(self.output_size)
        self.attention = nn.MultiheadAttention(self.output_size, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(self.output_size)
        self.ffn = nn.Sequential(nn.Linear(self.output_size, 2 * self.output_size), nn.ReLU(),
                                 nn.Linear(2 * self.output_size, self.output_size))
        self.output_norm = nn.LayerNorm(self.output_size)

        # 相对于当前时刻的位置编码只表达顺序，不包含传感器质量或时间龄期。
        positions = torch.arange(1 - args.history_length, 1, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, self.output_size, 2) * (-math.log(10000.0) / self.output_size))
        pe = torch.zeros(args.history_length, self.output_size)
        pe[:, 0::2] = torch.sin(positions * frequencies)
        pe[:, 1::2] = torch.cos(positions * frequencies[:pe[:, 1::2].shape[1]])
        self.register_buffer("position_encoding", pe)

    def forward(self, obs, history, masks):
        if history.shape != (obs.shape[0], *self.state_shape):
            raise ValueError("注意力状态形状不匹配，请按策略的 state_shape 初始化历史")
        # mask=0 开始新回合；历史原始值重新编码，避免复用旧参数生成的特征。
        history = torch.where(masks.bool().unsqueeze(-1), history, torch.zeros_like(history))
        current = torch.cat((obs, torch.ones_like(obs[:, :1])), dim=-1).unsqueeze(1)
        window = torch.cat((history[:, 1:], current), dim=1)
        valid = window[:, :, -1].bool()
        raw_obs = torch.where(valid.unsqueeze(-1), window[:, :, :-1], torch.zeros_like(window[:, :, :-1]))
        tokens = self.embedding(self.feature_norm(raw_obs)) + self.position_encoding
        normalized = self.attention_norm(tokens)
        # 当前 token 始终有效；仅查询最后一帧，无未来 token，等价于因果层的最后一行。
        attended, _ = self.attention(normalized[:, -1:], normalized, normalized,
                                     key_padding_mask=~valid, need_weights=False)
        features = tokens[:, -1] + attended.squeeze(1)
        features = features + self.ffn(self.ffn_norm(features))
        return self.output_norm(features), window.detach()
