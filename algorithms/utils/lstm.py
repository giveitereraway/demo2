import torch
from torch import nn


class LSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_size = hidden_size

    def forward(self, x: torch.Tensor, state: torch.Tensor, masks: torch.Tensor):
        # 缓冲区沿最后一维拼接 h 和 c，保留环境、层、特征三维布局。
        batch_size = state.size(0)
        sequence = x.reshape(-1, batch_size, x.size(-1))
        masks = masks.reshape(-1, batch_size)
        hidden, cell = state.transpose(0, 1).split(self.output_size, dim=-1)

        # 在任一环境重置的位置分段，其余时间步使用一次 LSTM 调用。
        boundaries = [0, len(sequence)]
        if len(sequence) > 1:
            resets = (masks[1:] == 0).any(dim=1).nonzero(as_tuple=True)[0] + 1
            boundaries[1:1] = resets.tolist()
        outputs = []
        for start, stop in zip(boundaries, boundaries[1:]):
            keep = masks[start].view(1, batch_size, 1)
            initial = ((hidden * keep).contiguous(), (cell * keep).contiguous())
            output, (hidden, cell) = self.lstm(sequence[start:stop], initial)
            outputs.append(output)

        features = torch.cat(outputs, dim=0).reshape(-1, self.output_size)
        state = torch.cat((hidden, cell), dim=-1).transpose(0, 1)
        return self.norm(features), state
