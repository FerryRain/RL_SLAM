"""
# File       : PPO_policy.py
# Time       ：2022/12/17 22:45
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
from Net.PPO_Net import Actor, Critic, ActorCritic
from typing import Union, Tuple, Any, Optional, Type, Sequence

import torch, numpy as np
from torch import nn
ModuleType = Type[nn.Module]
class PPO_policy():
    def __init__(self,
                 state_shape,
                 action_shape,
                 device: Union[str, int, torch.device] = 'cuda',
                 norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
                 activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
                 flatten_input: bool = False,
                 lr: float = 0.0001,
                 lr_decay: bool = True
                 ):
        self.device = device
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.lr = lr

        actor = Actor(self.state_shape,self.action_shape, self.device, norm_layer=norm_layer, activation=activation,flatten_input=False)
        critic = Critic(self.action_shape, self.device, norm_layer=norm_layer, activation=activation, flatten_input=flatten_input)
        optim = torch.optim.Adam(
            ActorCritic(actor, critic).parameters(), lr=lr, eps=1e-5
        )

        lr_scheduler = None
        if lr_decay:
            # decay learning rate to 0 linearly
            max_update_num = np.ceil(
                args.step_per_epoch / args.step_per_collect
            ) * args.epoch

            lr_scheduler = LambdaLR(
                optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
            )