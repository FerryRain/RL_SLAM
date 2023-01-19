"""
# File       : PPO_policy.py
# Time       ：2022/12/17 22:45
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import os
import pprint

from torch.utils.tensorboard import SummaryWriter

# from Net.PPO_Net import Actor, Critic, ActorCritic
from typing import Union, Tuple, Any, Optional, Type, Sequence
# from Thirdpart.tianshou.examples.atari.atari_network import DQN, layer_init, scale_obs
import torch, numpy as np
import datetime
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
ModuleType = Type[nn.Module]

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import ICMPolicy, PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
# from tianshou.utils.net.common import ActorCritic
# from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
class PPO_policy():
    def __init__(self,
                 state_shape,
                 action_shape,
                 step_per_epoch,
                 step_per_collect,
                 epoch,
                 env,
                 gamma=0.99,
                 buffer_size: int = 10000,
                 repeat_per_collect: int = 4,
                 batch_size: int = 256,
                 training_num:int = 1,
                 test_num: int = 1,
                 rew_norm: bool = False,
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.01,
                 gae_lambda: float = 0.95,
                 max_grad_norm: float = 0.5,
                 eps_clip: float = 0.1,
                 dual_clip: float = None,
                 value_clip: bool = True,
                 norm_adv: bool = True,
                 recompute_adv: bool = False,
                 frames_stack: int = 1,
                 device: Union[str, int, torch.device] = 'cuda',
                 norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
                 activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
                 flatten_input: bool = False,
                 lr: float = 0.0001,
                 lr_decay: bool = True,

                 logdir: str = "log",
                 resume_path: str = None
                 ):
        self.device = device
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.lr = lr
        self.batch_size = batch_size
        self.training_num = training_num
        self.epoch = epoch
        self.step_per_epoch = step_per_epoch
        self.repeat_per_collect = repeat_per_collect
        self.test_num = test_num
        self.step_per_collect = step_per_collect

        actor = Actor(self.state_shape,self.action_shape, self.device, norm_layer=norm_layer, activation=activation,flatten_input=False)
        critic = Critic(self.action_shape, self.device, norm_layer=norm_layer, activation=activation, flatten_input=flatten_input)
        # net_cls = scale_obs(DQN) if scale_obs else DQN
        # net = net_cls(
        #
        #     state_shape,
        #     action_shape,
        #     device=device,
        #     features_only=True,
        #     output_dim=3,
        #     layer_init=layer_init,
        # )
        # actor = Actor(net, action_shape, device=device, softmax_output=False)
        # critic = Critic(net, device=device)
        optim = torch.optim.Adam(
            ActorCritic(actor, critic).parameters(), lr=lr, eps=1e-5
        )

        lr_scheduler = None
        if lr_decay:
            # decay learning rate to 0 linearly
            max_update_num = np.ceil(
                step_per_epoch / step_per_collect
            ) * epoch

            lr_scheduler = LambdaLR(
                optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
            )

        def dist(p):
            return torch.distributions.Categorical(logits=p)

        # Todo:修改Policy的foreward、update、reset
        self.policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=gamma,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            reward_normalization=rew_norm,
            action_scaling=False,
            lr_scheduler=lr_scheduler,
            action_space=action_shape,
            eps_clip=eps_clip,
            value_clip=value_clip,
            dual_clip=dual_clip,
            advantage_normalization=norm_adv,
            recompute_advantage=recompute_adv,
        ).to(device)

        if resume_path:
            self.policy.load_state_dict(torch.load(resume_path, map_location=device))
            print("Loaded agent from: ", resume_path)

        # buffer
        self.buffer = VectorReplayBuffer(
            buffer_size,
            buffer_num=training_num,
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=frames_stack,
        )

        # collector
        self.train_collector = Collector(self.policy, env, self.buffer, exploration_noise=True)
        self.test_collector = Collector(self.policy, env, exploration_noise=True)

        # log
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        algo_name = "ppo"
        self.log_name = os.path.join("UAV", algo_name, now)
        self.log_path = os.path.join(logdir, self.log_name)

        self.writer = SummaryWriter(self.log_path)
        self.logger = TensorboardLogger(self.writer)

    def save_best_fn(self, policy):
        torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

    def stop_fn(self, mean_rewards):
        return mean_rewards >= 2000000000   #TODO

    def save_checkpoint_fn(self, epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(self.log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": self.policy.state_dict()}, ckpt_path)
        return ckpt_path

    def train(self):
        #Todo: 重写collect
        self.train_collector.collect(n_step=self.batch_size * self.training_num)

        # Todo:
        result = onpolicy_trainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            self.epoch,
            self.step_per_epoch,
            self.repeat_per_collect,
            self.test_num,
            self.batch_size,
            step_per_collect=self.step_per_collect,
            stop_fn=self.stop_fn,
            save_best_fn=self.save_best_fn,
            logger=self.logger,
            test_in_train=False,
            save_checkpoint_fn=self.save_checkpoint_fn,
        )

        pprint.pprint(result)
