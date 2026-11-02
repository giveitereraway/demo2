import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.temporal_attention import TemporalAttention, get_memory_encoder, get_policy_state_shape
from ..utils.act import ACTLayer
from ..utils.utils import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cuda:0")):
        super(PPOActor, self).__init__()
        # network config         args.gain这些参数是在render_*.py中设置的
        self.gain = args.gain # 默认0.01
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.memory_encoder = get_memory_encoder(args)
        self.use_recurrent_policy = self.memory_encoder == "gru"
        self.state_shape = get_policy_state_shape(args, obs_space)
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_prior = args.use_prior
        # (1) feature extraction module 特征提取网络，把原始观测转为高维特征 activation_id默认为1
        self.base = (TemporalAttention(args, obs_space) if self.memory_encoder == "temporal_attention"
                     else MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization))
        input_size = self.base.output_size
        # (2) rnn module 可选的GRU网络，把高维特征转化为低维特征
        if self.use_recurrent_policy: # use_recurrent_policy 默认为True
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) act module 动作输出层，根据特征输出动作分布
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)
        # 顺序：观测值obs_space→MLP特征提取→（GRU可选）→act动作输出层→动作分布
        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        # obs: (T*N,obs_space); rnn_states: (T*N,num_layers,hidden_size); masks: (T*N, 1)
        # obs, rnn_states, masks被存储在ReplayBuffer里；collect方法中准备传递给policy；
        # 在 PPOPolicy 的 get_actions 方法中，这些数据被传递给actor的 forward 方法
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.use_prior:
            # prior knowledage for controling shoot missile
            attack_angle = torch.rad2deg(obs[:, 11]) # unit degree
            distance = obs[:, 13] * 10000 # unit m
            alpha0 = torch.full(size=(obs.shape[0],1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0],1), fill_value=10).to(**self.tpdv)
            alpha0[distance<=12000] = 6 # 原本是6 90最佳
            alpha0[distance<=8000] = 10 # 原本是10 95最佳
            beta0[attack_angle<=45] = 6
            beta0[attack_angle<=22.5] = 3
        if self.memory_encoder == "temporal_attention":
            actor_features, rnn_states = self.base(obs, rnn_states, masks)
        else:
            actor_features = self.base(obs)

        if self.use_recurrent_policy: # 然后经过GRU层（可选）
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            #print("GRU OK")
        if self.use_prior: # 最后通过动作输出层
            actions, action_log_probs = self.act(actor_features, deterministic=True, alpha0=alpha0, beta0=beta0)
        else:
            actions, action_log_probs = self.act(actor_features, deterministic)
        #print("ACT OK")

        return actions, action_log_probs, rnn_states
        # obs(任意维)→2层MLP(128，128)→1层GRU(128)→1层ACTLayer→actions

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        """评估已有动作 的对数概率和计算分布熵"""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.use_prior:
            # prior knowledage for controling shoot missile
            attack_angle = torch.rad2deg(obs[:, 11]) # unit degree
            distance = obs[:, 13] * 10000 # unit m
            alpha0 = torch.full(size=(obs.shape[0], 1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0], 1), fill_value=10).to(**self.tpdv)
            alpha0[distance<=12000] = 6
            alpha0[distance<=8000] = 10
            beta0[attack_angle<=45] = 6
            beta0[attack_angle<=22.5] = 3

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.memory_encoder == "temporal_attention":
            actor_features, rnn_states = self.base(obs, rnn_states, masks)
        else:
            actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_prior:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks, alpha0=alpha0, beta0=beta0)
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy # 动作的对数概率（用于PPO损失计算），分布的熵（用于正则化）

"""
1、输入观测 obs
2、预处理（to device, check）
3、（可选）用先验知识调整分布参数
4、MLPBase 特征提取
5、（可选）GRU时序处理
6、ACTLayer 输出动作分布，采样动作
7、返回动作、log_prob、rnn状态；损失值
"""
