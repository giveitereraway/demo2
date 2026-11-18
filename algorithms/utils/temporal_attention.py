import math

import torch
import torch.nn as nn

from .gru import GRULayer, InnovationGatedResidualGRU
from .mlp import MLPBase


def get_memory_encoder(args):
    """显式选择编码器，同时兼容旧命令中反向的 GRU 开关。"""
    encoder = getattr(args, "memory_encoder", None)
    if encoder is None:
        return "gru" if args.use_recurrent_policy else "mlp"
    if encoder != "mlp" and not args.use_recurrent_policy:
        raise ValueError("--memory-encoder 与关闭 GRU 的旧参数 --use-recurrent-policy 冲突，请移除旧参数")
    if encoder in {"temporal_attention", "gru_attention", "igru"} and getattr(args, "algorithm_name", "ppo") != "ppo":
        raise ValueError("扩展时序编码器仅支持 PPO，暂不支持 MAPPO")
    return encoder


def is_recurrent_sequence_encoder(args) -> bool:
    """判断编码器是否通过 recurrent_generator 按时间块训练。"""
    return get_memory_encoder(args) in {"gru", "gru_attention", "igru"}


def get_policy_state_shape(args, obs_space):
    """返回策略记忆状态形状，注意力与改进 GRU 编码器复用现有 rnn_states 通道。"""
    encoder = get_memory_encoder(args)
    if encoder == "temporal_attention":
        if len(obs_space.shape) != 1 or args.history_length < 1:
            raise ValueError("时序注意力要求一维观测，且 --history-length 至少为 1")
        return args.history_length, obs_space.shape[0] + 1
    if encoder == "gru_attention":
        if len(obs_space.shape) != 1 or args.history_length < 1:
            raise ValueError("门控递归注意力要求一维观测，且 --history-length 至少为 1")
        if args.recurrent_hidden_layers < 1 or args.recurrent_hidden_size < 1:
            raise ValueError("GRU 层数和隐状态维数必须为正整数")
        gru_dim = args.recurrent_hidden_layers * args.recurrent_hidden_size
        hist_dim = args.history_length * (obs_space.shape[0] + 1)
        return 1, gru_dim + hist_dim
    if encoder == "igru":
        if len(obs_space.shape) != 1:
            raise ValueError("新息门控残差 GRU 要求一维观测")
        if args.recurrent_hidden_layers < 1 or args.recurrent_hidden_size < 1:
            raise ValueError("GRU 层数和隐状态维数必须为正整数")
        gru_dim = args.recurrent_hidden_layers * args.recurrent_hidden_size
        return 1, gru_dim + obs_space.shape[0] + 1
    return args.recurrent_hidden_layers, args.recurrent_hidden_size


def _build_relative_position_encoding(history_length: int, dim: int) -> torch.Tensor:
    """构造相对于当前时刻的正弦位置编码，仅表达帧间相对顺序。"""
    positions = torch.arange(1 - history_length, 1, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(history_length, dim)
    pe[:, 0::2] = torch.sin(positions * frequencies)
    pe[:, 1::2] = torch.cos(positions * frequencies[:pe[:, 1::2].shape[1]])
    return pe


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
        self.register_buffer("position_encoding", _build_relative_position_encoding(args.history_length, self.output_size))

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


class GatedAttentionGRU(nn.Module):
    """门控递归-因果注意力混合编码器：融合 MLP+GRU 递归低通滤波与帧间差分感知的历史回溯。"""

    def __init__(self, args, obs_space):
        super().__init__()
        self.state_shape = get_policy_state_shape(args, obs_space)
        self.obs_dim = obs_space.shape[0]
        self.history_length = args.history_length
        self._gru_layers = args.recurrent_hidden_layers
        self._gru_hidden = args.recurrent_hidden_size
        self._gru_flat_dim = self._gru_layers * self._gru_hidden
        self._hist_flat_dim = self.history_length * (self.obs_dim + 1)
        self.output_size = self._gru_hidden

        attn_dim = args.attention_hidden_size
        heads = args.attention_heads
        if heads < 1 or attn_dim < 1 or attn_dim % heads:
            raise ValueError("注意力维数必须为正数，且能被头数整除")
        self.attn_dim = attn_dim

        # 路径 A：与 baseline 完全一致的非线性特征提取 + GRU 递归滤波主干。
        self.mlp_base = MLPBase(obs_space, args.hidden_size, args.activation_id, args.use_feature_normalization)
        self.rnn = GRULayer(self.mlp_base.output_size, self._gru_hidden, self._gru_layers)

        # 路径 B：拼接 [o_k, o_k - o_{k-1}] 感知零阶保持丢帧冻结与变化率，经非线性编码后做因果注意力。
        act_cls = [nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.ELU][args.activation_id]
        self.obs_norm = nn.LayerNorm(self.obs_dim) if args.use_feature_normalization else nn.Identity()
        self.token_encoder = nn.Sequential(
            nn.Linear(2 * self.obs_dim, attn_dim),
            act_cls(),
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, attn_dim),
            act_cls(),
            nn.LayerNorm(attn_dim),
        )
        self.register_buffer("position_encoding", _build_relative_position_encoding(self.history_length, attn_dim))

        # 使用 GRU 滤波后的平滑隐状态生成去噪 Query，避免单帧强噪声污染注意力权重。
        self.query_proj = nn.Linear(self._gru_hidden, attn_dim)
        self.query_norm = nn.LayerNorm(attn_dim)
        self.kv_norm = nn.LayerNorm(attn_dim)
        self.attention = nn.MultiheadAttention(attn_dim, heads, dropout=0.0, batch_first=True)

        self.context_proj = nn.Linear(attn_dim, self.output_size)
        self.ffn_norm = nn.LayerNorm(self.output_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.output_size, 2 * self.output_size),
            act_cls(),
            nn.Linear(2 * self.output_size, self.output_size),
        )
        self.context_out_norm = nn.LayerNorm(self.output_size)

        # Sigmoid 残差门控初始偏置设为 -2.0（约 0.12），保证初始阶段与干净观测下对齐 GRU 下限。
        self.gate_proj = nn.Linear(2 * self.output_size, self.output_size)
        nn.init.orthogonal_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        self.output_norm = nn.LayerNorm(self.output_size)

        self.record_attention = False
        self.last_gate_mean = 0.0
        self.last_attention_weights = None
        self.last_gate_values = None

    def _roll_windows(self, obs: torch.Tensor, history: torch.Tensor, masks: torch.Tensor):
        """在单步采样 (T=1) 与序列分块训练 (T=L) 下以完全一致的语义滑动并重置历史窗口。"""
        batch_envs = history.shape[0]
        total_steps = obs.shape[0]
        if total_steps == batch_envs:
            reset_hist = torch.where(masks.bool().unsqueeze(-1), history, torch.zeros_like(history))
            current = torch.cat((obs, torch.ones_like(obs[:, :1])), dim=-1).unsqueeze(1)
            next_history = torch.cat((reset_hist[:, 1:], current), dim=1)
            return next_history, next_history

        chunk_len = total_steps // batch_envs
        obs_seq = obs.view(chunk_len, batch_envs, self.obs_dim)
        mask_seq = masks.view(chunk_len, batch_envs, 1)
        cur_hist = history
        windows = []
        for t in range(chunk_len):
            cur_hist = torch.where(mask_seq[t].bool().unsqueeze(-1), cur_hist, torch.zeros_like(cur_hist))
            current = torch.cat((obs_seq[t], torch.ones_like(obs_seq[t, :, :1])), dim=-1).unsqueeze(1)
            cur_hist = torch.cat((cur_hist[:, 1:], current), dim=1)
            windows.append(cur_hist)
        stacked = torch.stack(windows, dim=0).view(total_steps, self.history_length, self.obs_dim + 1)
        return stacked, cur_hist

    def forward(self, obs: torch.Tensor, state: torch.Tensor, masks: torch.Tensor):
        if state.ndim != 3 or state.shape[1:] != self.state_shape:
            raise ValueError("门控递归注意力状态形状不匹配，请按策略的 state_shape 初始化状态")
        batch_envs = state.shape[0]
        if obs.shape[0] < batch_envs or obs.shape[0] % batch_envs != 0 or masks.shape[0] != obs.shape[0]:
            raise ValueError("观测与掩码批次维数必须是状态环境数的正整数倍")

        hxs = state[:, 0, :self._gru_flat_dim].reshape(batch_envs, self._gru_layers, self._gru_hidden)
        history = state[:, 0, self._gru_flat_dim:].reshape(batch_envs, self.history_length, self.obs_dim + 1)

        # 路径 A：GRU 递归特征
        mlp_features = self.mlp_base(obs)
        gru_features, next_hxs = self.rnn(mlp_features, hxs, masks)

        # 路径 B：帧间差分感知的历史因果注意力
        windows, next_history = self._roll_windows(obs, history, masks)
        valid = windows[:, :, -1].bool()
        raw_obs = torch.where(valid.unsqueeze(-1), windows[:, :, :-1], torch.zeros_like(windows[:, :, :-1]))

        delta_obs = torch.zeros_like(raw_obs)
        if self.history_length > 1:
            pair_valid = valid[:, 1:] & valid[:, :-1]
            diff = raw_obs[:, 1:] - raw_obs[:, :-1]
            delta_obs[:, 1:] = torch.where(pair_valid.unsqueeze(-1), diff, torch.zeros_like(diff))

        token_inputs = torch.cat((self.obs_norm(raw_obs), delta_obs), dim=-1)
        tokens = self.token_encoder(token_inputs) + self.position_encoding
        query = self.query_norm(self.query_proj(gru_features)).unsqueeze(1)
        kv = self.kv_norm(tokens)
        need_weights = bool(self.record_attention)
        attended, attn_weights = self.attention(
            query, kv, kv, key_padding_mask=~valid, need_weights=need_weights, average_attn_weights=True
        )

        context = self.context_proj(attended.squeeze(1))
        context = self.context_out_norm(context + self.ffn(self.ffn_norm(context)))

        gate = torch.sigmoid(self.gate_proj(torch.cat((gru_features, context), dim=-1)))
        features = self.output_norm(gru_features + gate * context)

        self.last_gate_mean = float(gate.detach().mean().item())
        if need_weights and attn_weights is not None:
            self.last_attention_weights = attn_weights.detach()
            self.last_gate_values = gate.detach()

        next_state = torch.cat(
            (
                next_hxs.reshape(batch_envs, 1, self._gru_flat_dim),
                next_history.detach().reshape(batch_envs, 1, self._hist_flat_dim),
            ),
            dim=-1,
        )
        return features, next_state

