import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Any, Optional, Type, Sequence
ModuleType = Type[nn.Module]
class Actor(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 device: Union[str, int, torch.device] = 'cpu',
                 norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
                 activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
                 flatten_input: bool = False,
                 ):
        super().__init__()

        self.device = device

        self.input_dim = state_shape
        self.output_dim = action_shape
        if norm_layer:
            self.model = nn.Sequential(*[
                norm_layer(normalized_shape=np.prod(state_shape),device=self.device),
                nn.Linear(self.input_dim, 128), activation(),
                nn.Linear(128, 128), activation(),
                nn.Linear(128, 64), activation(),
                nn.Linear(64, self.output_dim)
            ])
        else:
            self.model = nn.Sequential(*[
                nn.Linear(self.input_dim, 128), activation(),
                nn.Linear(128, 128), activation(),
                nn.Linear(128, 64), activation(),
                nn.Linear(64, self.output_dim)
            ])

        self.flatten_input = flatten_input

    def forward(self,
                obs: Union[np.array, torch.Tensor],
                state: None) -> Tuple[torch.Tensor, Any]:
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        act_n = self.model(obs)
        return act_n


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2