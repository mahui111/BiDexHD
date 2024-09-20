import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional
from ..pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet, getPointNetWithInstanceInfo
from typing import List, Optional, Tuple


class PointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet({"input_feature_dim": self.pc_dim, "feat_dim": self.feature_dim})
        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(pretrained_model_path, map_location="cpu")["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(state_dict,strict=False,)
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)

    def forward(self, input_obs):
        if 'mask' in input_obs:
            return self.backbone(torch.cat([input_obs['pc'], input_obs['mask']], dim=-1))
        else:
            return self.backbone(input_obs['pc'])

class TransPointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        state_dim: int,
        use_seg: bool = True,
    ):
        super().__init__()

        cfg = {}
        cfg["state_dim"] = state_dim
        cfg["feature_dim"] = feature_dim
        cfg["pc_dim"] = pc_dim
        cfg["output_dim"] = feature_dim
        if use_seg:
            cfg["mask_dim"] = 2
        else:
            cfg["mask_dim"] = 0

        self.transpn = getPointNetWithInstanceInfo(cfg)

    def forward(self, input_obs):
        input_obs["pc"] = torch.cat([input_obs["pc"], input_obs["mask"]], dim=-1)
        return self.transpn(input_obs)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        model_cfg,
        asymmetric=False,
        **kwargs,
    ):
        super(ActorCritic, self).__init__()

        self.use_pc = len(kwargs["pointcloud_indices"]) > 0
        self.use_objlabel = len(kwargs["objlabel_indices"]) > 0
        if "futureobjps_indices" in kwargs and len(kwargs["futureobjps_indices"]) > 0:
            self.futureobjps_indices = kwargs["futureobjps_indices"]
            self.use_futureobjps = 1
        else:
            self.use_futureobjps = 0

        if model_cfg is None:  # default
            actor_hidden_dim = [1024, 1024, 512, 512]
            critic_hidden_dim = [1024, 1024, 512, 512]
            activation = get_activation("elu")
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation = get_activation(model_cfg["activation"])

        self.num_obs = obs_shape[0]
        self.robostate_indices = kwargs["robostate_indices"]
        self.pointcloud_indices = kwargs.get("pointcloud_indices", [])
        self.objlabel_indices = kwargs.get("objlabel_indices", [])

        if self.use_pc:
            assert model_cfg is not None
            self.use_seg = int(model_cfg["useSeg"])
            self.num_downsample = model_cfg["numDownsample"]
            self.pc_emb_dim = model_cfg["pcEmbDim"]
            self.each_point_dim = model_cfg["numEachPoint"]
            self.num_pc_flatten = self.num_downsample * self.each_point_dim
            self.backbone_type = model_cfg["backbone_type"]
            if self.backbone_type == "PointNetBackbone":
                self.backbone = PointNetBackbone(
                    pc_dim=self.each_point_dim + 2 * self.use_seg,
                    feature_dim=self.pc_emb_dim if 'centralize' in kwargs else self.pc_emb_dim // 2
                )
            elif self.backbone_type == "TransPointNetBackbone":
                self.backbone = TransPointNetBackbone(
                    pc_dim=self.each_point_dim,
                    feature_dim=self.pc_emb_dim,
                    state_dim=self.num_robot_state,
                    use_seg=self.use_seg,
                )
            else:
                raise ValueError(f"Invalid backbone type: {self.backbone_type}")
            assert len(self.pointcloud_indices) % self.num_pc_flatten == 0
        else:
            self.pc_emb_dim = 0

        if self.use_objlabel:
            self.objlabel_dim = len(self.robostate_indices) + self.pc_emb_dim
            self.objlabel_emb = nn.Embedding(256, self.objlabel_dim)
            nn.init.xavier_uniform_(self.objlabel_emb.weight)


        self.input_dim = len(self.robostate_indices)
        if self.use_pc:
            self.input_dim += self.pc_emb_dim
        if self.use_futureobjps:
            self.input_dim += len(self.futureobjps_indices)

        actor_layers = []
        critic_layers = []

        actor_layers.append(nn.Linear(self.input_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers.append(nn.Linear(self.input_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def forward(self):
        raise NotImplementedError
    
    def get_pc_observation(self, observations):
        robot_state = observations[:, self.robostate_indices]
        pc = observations[:, self.pointcloud_indices].reshape(-1, self.num_downsample, self.each_point_dim)
        input_data = dict(pc=pc)
        if self.use_seg:
            raise NotImplementedError   
            mask = observations[:, -2 * self.num_downsample-1:-1].reshape(-1, self.num_downsample, 2)
            input_data.update(dict(mask=mask,))
        if self.backbone_type == "TransPointNetBackbone":
            raise NotImplementedError 
            input_data.update(dict(state=robot_state,))
        pc_feature = self.backbone(input_data).reshape(len(observations), -1)
        return pc_feature

    def get_all_observation(self, observations):
        all_observation = observations[:, self.robostate_indices]
        if self.use_pc:
            pc_feature = self.get_pc_observation(observations)
            all_observation = torch.cat([all_observation, pc_feature], dim=1)
        if self.use_objlabel:
            obj_label = observations[:, self.objlabel_indices] * 255
            objlabel_feature = self.objlabel_emb(obj_label.long()).mean(dim=1)
            all_observation = all_observation + objlabel_feature
        if self.use_futureobjps:
            future_objps = observations[:, self.futureobjps_indices]
            all_observation = torch.cat([all_observation, future_objps], dim=1)
        return all_observation

    def act(self, observations, grad=False):
        observations = self.get_all_observation(observations)
        actions_mean = self.actor(observations)#.clamp(-1.1, 1.1)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())

        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        value = self.critic(observations)

        if not grad:
            actions_mean = actions_mean.detach()
        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean,
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
        )

    def act_inference(self, observations):
        observations = self.get_all_observation(observations)
        actions_mean = self.actor(observations)#.clamp(-1.1, 1.1)
        return actions_mean.detach()

    def evaluate(self, observations, actions):
        observations = self.get_all_observation(observations)
        actions_mean = self.actor(observations)#.clamp(-1.1, 1.1)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(observations)

        return (
            actions_log_prob,
            entropy,
            value,
            actions_mean,
            self.log_std.repeat(actions_mean.shape[0], 1),
        )


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
