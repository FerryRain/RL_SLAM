import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Any, Optional, Type, Sequence
ModuleType = Type[nn.Module]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 60)
        self.layer_2 = nn.Linear(60, 40)
        self.layer_3 = nn.Linear(40, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # 2023年1月20日 修改 提高训练速度以及准确率 网络宽度
        self.layer_1 = nn.Linear(state_dim, 80)
        self.layer_2_s = nn.Linear(80, 60)
        self.layer_2_a = nn.Linear(action_dim, 60)
        self.layer_3 = nn.Linear(60, 1)

        self.layer_4 = nn.Linear(state_dim, 80)
        self.layer_5_s = nn.Linear(80, 60)
        self.layer_5_a = nn.Linear(action_dim, 60)
        self.layer_6 = nn.Linear(60, 1)

        # self.layer_1 = nn.Linear(state_dim, 60)
        # self.layer_2_s = nn.Linear(60, 40)
        # self.layer_2_a = nn.Linear(action_dim, 40)
        # self.layer_3 = nn.Linear(40, 1)
        #
        # self.layer_4 = nn.Linear(state_dim, 60)
        # self.layer_5_s = nn.Linear(60, 40)
        # self.layer_5_a = nn.Linear(action_dim, 40)
        # self.layer_6 = nn.Linear(40, 1)

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

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    a = torch.from_numpy(np.array([1]))
    a = Actor(1, 2)
    print(a(torch.tensor([1.0])))
