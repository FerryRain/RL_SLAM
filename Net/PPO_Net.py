"""
# File       : PPO_Net.py
# Time       ：2022/12/17 16:16
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
from typing import Union, Tuple, Any, Optional, Type, Sequence

import torch, numpy as np
from torch import nn
ModuleType = Type[nn.Module]

class Actor(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 device: Union[str, int, torch.device] = 'cuda',
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
                norm_layer(normalized_shape=state_shape,device=self.device),
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
        return act_n, state

class Critic(nn.Module):
    def __init__(self,
                 action_shape,
                 device: Union[str, int, torch.device] = 'cuda',
                 norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
                 activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
                 flatten_input: bool = True,
                 ):
        super().__init__()

        self.device = device
        self.output_dim = 1
        self.input_dim = action_shape

        if norm_layer:
            self.model = nn.Sequential(*[
                norm_layer(normalized_shape=self.input_dim, device=self.device),
                nn.Linear(self.input_dim, 128), activation(),
                nn.Linear(128, 64), activation(),
                nn.Linear(64, self.output_dim), activation()
            ])
        else:
            self.model = nn.Sequential(*[
                nn.Linear(self.input_dim, 128), activation(),
                nn.Linear(128, 64), activation(),
                nn.Linear(64, self.output_dim), activation()
            ])

        self.flatten_input = flatten_input

    def forward(self,
                obs: Union[np.array, torch.Tensor],
                act: Optional[Union[np.ndarray, torch.Tensor]] = None,
                ):
        if self.flatten_input:
            obs = torch.as_tensor(
                obs,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)

        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        Value = self.last(obs)
        return Value

class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.

    Using ``actor_critic.parameters()`` instead of set.union or list+list to avoid
    issue #449.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """

    def __init__(self, actor: nn.Module, critic: nn.Module) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic