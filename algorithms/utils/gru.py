
import torch
import torch.nn as nn

from .mlp import MLPBase
# 当 mask=0 时，表示当前状态是一个episode的结束，隐藏状态会被重置；当 mask=1 时，隐藏状态会从之前的时间步传递过来
# 这确保了智能体不会在不同episode之间混淆信息

class GRULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(GRULayer, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)
        # NOTE: self.gru(x, hxs) needs x=[T, N, input_size] and hxs=[L, N, hidden_size]

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor):
        # NOTE: N = mini_batch_size; T = recurrent chunk length; L = gru layers

        # (T=1) x: [N, input_size], hxs: [N, L, hidden_size], masks: [N, 1]
        if x.size(0) == hxs.size(0):
            # masks: [N, 1] => [N, L] => [N, L, 1]
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks.repeat(1, self._num_layers).unsqueeze(-1)).transpose(0, 1).contiguous())

            x = x.squeeze(0)            # [1, N, input_size] => [N, input_size]
            hxs = hxs.transpose(0, 1)   # [L, N, hidden_size] => [N, L, hidden_size]

        # (T>1): x=[T * N, input_size], hxs=[N, L, hidden_size], masks=[T * N, 1]
        else:
            # Mannual reset hxs to zero at ternimal states might be too slow to calculate
            # We need to tackle the problem more efficiently

            # x is a (T, N, input_size) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0) # 2400
            T = int(x.size(0) / N) # 19200 / 2400 = 8
            # unflatten x and masks
            x = x.view(T, N, x.size(1))  # [T * N, input_size] => [T, N, input_size]
            masks = masks.view(T, N)     # [T * N, 1] => [T, N]

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)#判断是否有true # [T, N] => [T, 1]
                         .nonzero(as_tuple=False) # 找出True值的索引
                         .squeeze(dim=-1)   # [T, 1] => [T]
                         .cpu())
            # 例如：has_zeros = [10,20] 表示第10步和第20步有智能体完成了一个episode
            # +1 to correct the masks[1:]
            has_zeros = (has_zeros + 1).numpy().tolist()
            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)   # [N, L, hidden_size] => [L, N, hidden_size]
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                # masks[start_idx]: [N] => [1, N, 1] => [L, N, 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._num_layers, 1, 1)).contiguous()
                rnn_scores, hxs = self.gru(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.view(T * N, -1)       # [T, N, input_size] => [T * N, input_size]
            hxs = hxs.transpose(0, 1)   # [L, N, hidden_size] => [N, L, hidden_size]

        x = self.norm(x)
        return x, hxs

    @property
    def output_size(self):
        return self._hidden_size


class InnovationGatedResidualGRU(nn.Module):
    """新息门控残差 GRU（IGRU）：仅缓存上一帧观测，融合一阶差分/冻结感知调制、原生 GRU 滤波与先验新息残差直通。"""

    DELTA_CLIP = 2.0

    def __init__(self, args, obs_space):
        super().__init__()
        if len(obs_space.shape) != 1:
            raise ValueError("新息门控残差 GRU 要求一维观测空间")
        if args.recurrent_hidden_layers < 1 or args.recurrent_hidden_size < 1:
            raise ValueError("GRU 层数和隐状态维数必须为正整数")

        self.obs_dim = obs_space.shape[0]
        self._gru_layers = args.recurrent_hidden_layers
        self._gru_hidden = args.recurrent_hidden_size
        self._gru_flat_dim = self._gru_layers * self._gru_hidden
        self._prev_flat_dim = self.obs_dim + 1
        self.state_shape = (1, self._gru_flat_dim + self._prev_flat_dim)
        self.output_size = self._gru_hidden

        act_cls = [nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.ELU][args.activation_id]

        # 主干瞬时特征提取与原生 cuDNN GRU 递归滤波，与 baseline 结构严格对齐。
        self.mlp_base = MLPBase(obs_space, args.hidden_size, args.activation_id, args.use_feature_normalization)
        base_dim = self.mlp_base.output_size
        self.rnn = GRULayer(base_dim, self._gru_hidden, self._gru_layers)
        self.inst_proj = nn.Linear(base_dim, self._gru_hidden) if base_dim != self._gru_hidden else nn.Identity()

        # 前端：由 [o_t, Δo_t, |Δo_t|, valid_t] 编码运动导数并感知零阶保持丢帧冻结（|Δo_t|=0）与异常跳变。
        self.obs_norm = nn.LayerNorm(self.obs_dim) if args.use_feature_normalization else nn.Identity()
        self.diff_encoder = nn.Sequential(
            nn.Linear(3 * self.obs_dim + 1, base_dim),
            act_cls(),
            nn.LayerNorm(base_dim),
        )
        in_bottleneck = max(16, base_dim // 4)
        in_gate_out = nn.Linear(in_bottleneck, base_dim)
        nn.init.orthogonal_(in_gate_out.weight, gain=0.1)
        nn.init.constant_(in_gate_out.bias, -2.0)
        self.in_gate_net = nn.Sequential(
            nn.Linear(2 * base_dim, in_bottleneck),
            act_cls(),
            in_gate_out,
        )

        # 后端：由上一时刻先验隐状态 h_{t-1} 预测瞬时特征，依据新息残差控制干净高频特征的跳跃直通。
        self.prior_pred = nn.Linear(self._gru_hidden, self._gru_hidden)
        res_bottleneck = max(16, self._gru_hidden // 4)
        res_gate_out = nn.Linear(res_bottleneck, self._gru_hidden)
        nn.init.orthogonal_(res_gate_out.weight, gain=0.1)
        nn.init.constant_(res_gate_out.bias, -2.0)
        self.res_gate_net = nn.Sequential(
            nn.Linear(4 * self._gru_hidden, res_bottleneck),
            act_cls(),
            res_gate_out,
        )
        self.output_norm = nn.LayerNorm(self.output_size)

        self.last_in_gate_mean = 0.0
        self.last_gate_mean = 0.0

    def _compute_step_diff(self, obs: torch.Tensor, prev_state: torch.Tensor, masks: torch.Tensor):
        """在单步采样 (T=1) 与序列分块训练 (T=L) 下无循环计算帧间差分与有效帧对标记。"""
        batch_envs = prev_state.shape[0]
        total_steps = obs.shape[0]
        prev_obs_init = prev_state[:, :self.obs_dim]
        prev_valid_init = prev_state[:, self.obs_dim:]

        if total_steps == batch_envs:
            pair_valid = (masks > 0.5) & (prev_valid_init > 0.5)
            delta = torch.where(pair_valid, obs - prev_obs_init, torch.zeros_like(obs))
            next_prev = torch.cat((obs.detach(), torch.ones_like(prev_valid_init)), dim=-1)
            return delta, pair_valid.to(obs.dtype), next_prev

        chunk_len = total_steps // batch_envs
        obs_seq = obs.view(chunk_len, batch_envs, self.obs_dim)
        mask_seq = masks.view(chunk_len, batch_envs, 1)

        prev_obs_seq = torch.cat((prev_obs_init.unsqueeze(0), obs_seq[:-1]), dim=0)
        ones_valid = torch.ones((chunk_len - 1, batch_envs, 1), dtype=torch.bool, device=obs.device)
        prev_exists_seq = torch.cat((prev_valid_init.unsqueeze(0) > 0.5, ones_valid), dim=0)
        pair_valid_seq = (mask_seq > 0.5) & prev_exists_seq

        delta_seq = torch.where(pair_valid_seq, obs_seq - prev_obs_seq, torch.zeros_like(obs_seq))
        next_prev = torch.cat((obs_seq[-1].detach(), torch.ones_like(prev_valid_init)), dim=-1)
        return (
            delta_seq.reshape(total_steps, self.obs_dim),
            pair_valid_seq.reshape(total_steps, 1).to(obs.dtype),
            next_prev,
        )

    def _extract_prior_hidden(
        self,
        hxs: torch.Tensor,
        gru_features: torch.Tensor,
        pair_valid_float: torch.Tensor,
    ) -> torch.Tensor:
        """提取每步更新前的归一化顶层先验隐状态 h_{t-1}，保证 T=1 与 T=L 严格一致。"""
        batch_envs = hxs.shape[0]
        total_steps = gru_features.shape[0]
        prior_0 = self.rnn.norm(hxs[:, -1, :])
        if total_steps == batch_envs:
            return prior_0 * pair_valid_float

        chunk_len = total_steps // batch_envs
        feat_seq = gru_features.view(chunk_len, batch_envs, self._gru_hidden)
        prior_seq = torch.cat((prior_0.unsqueeze(0), feat_seq[:-1]), dim=0)
        return prior_seq.reshape(total_steps, self._gru_hidden) * pair_valid_float

    def forward(self, obs: torch.Tensor, state: torch.Tensor, masks: torch.Tensor):
        if state.ndim != 3 or state.shape[1:] != self.state_shape:
            raise ValueError("IGRU 状态形状不匹配，请按策略的 state_shape 初始化状态")
        batch_envs = state.shape[0]
        if obs.shape[0] < batch_envs or obs.shape[0] % batch_envs != 0 or masks.shape[0] != obs.shape[0]:
            raise ValueError("观测与掩码批次维数必须是状态环境数的正整数倍")

        hxs = state[:, 0, :self._gru_flat_dim].reshape(batch_envs, self._gru_layers, self._gru_hidden)
        prev_state = state[:, 0, self._gru_flat_dim:]

        # 1. 无循环一阶差分与冻结/跳变幅值特征提取
        delta_raw, pair_valid_float, next_prev = self._compute_step_diff(obs, prev_state, masks)
        delta_clamped = torch.clamp(delta_raw, -self.DELTA_CLIP, self.DELTA_CLIP)
        diff_inputs = torch.cat((self.obs_norm(obs), delta_clamped, delta_clamped.abs(), pair_valid_float), dim=-1)

        # 2. 前端时序导数门控调制并送入原生并行 GRU
        z_t = self.mlp_base(obs)
        diff_feat = self.diff_encoder(diff_inputs)
        in_gate = torch.sigmoid(self.in_gate_net(torch.cat((z_t, diff_feat), dim=-1)))
        gru_in = z_t + in_gate * diff_feat
        gru_features, next_hxs = self.rnn(gru_in, hxs, masks)

        # 3. 后端先验新息残差计算与瞬时特征门控直通
        inst_feat = self.inst_proj(z_t)
        prior_h = self._extract_prior_hidden(hxs, gru_features, pair_valid_float)
        pred_inst = self.prior_pred(prior_h)
        innovation = pair_valid_float * (inst_feat - pred_inst)

        res_gate_in = torch.cat((gru_features, inst_feat, innovation, innovation.abs()), dim=-1)
        res_gate = torch.sigmoid(self.res_gate_net(res_gate_in))
        features = self.output_norm(gru_features + res_gate * inst_feat)

        self.last_in_gate_mean = float(in_gate.detach().mean().item())
        self.last_gate_mean = float(res_gate.detach().mean().item())

        next_state = torch.cat(
            (
                next_hxs.reshape(batch_envs, 1, self._gru_flat_dim),
                next_prev.reshape(batch_envs, 1, self._prev_flat_dim),
            ),
            dim=-1,
        )
        return features, next_state
