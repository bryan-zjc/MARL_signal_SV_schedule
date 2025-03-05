import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(MLP, self).__init__()
        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, layer_size)
        self.ff_3 = nn.Linear(layer_size, action_size)

    def forward(self, input):
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        x = torch.relu(self.ff_2(x))
        out = self.ff_3(x)
        
        return out

