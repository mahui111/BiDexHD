from datetime import datetime
import os
import os.path as osp
from pickle import FALSE
import time
from turtle import done

from matplotlib.patches import FancyArrow

from gym.spaces import Space

import numpy as np
import statistics
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .storage import RolloutStorage


class IPPOAgent(nn.Module):
    def __init__(
        self,
        vec_env,
        actor_critic_class,
        train_param,
        is_vision=False,
    ):
        super(IPPOAgent, self).__init__()
        # PPO parameters
        self.clip_param = train_param["cliprange"]
        self.num_learning_epochs = train_param["noptepochs"]
        self.num_mini_batches = train_param["nminibatches"]
        
        self.num_transitions_per_env = train_param["nsteps"]
        self.value_loss_coef = train_param.get("value_loss_coef", 2.0)
        self.entropy_coef = train_param["ent_coef"]
        self.gamma = train_param["gamma"]
        self.lam = train_param["lam"]
        self.max_grad_norm = train_param.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = train_param.get("use_clipped_value_loss", False)
        self.init_noise_std = train_param.get("init_noise_std", 0.3)

        self.model_cfg = train_param.policy
        self.sampler = train_param.get("sampler", "sequential")
        self.is_vision = is_vision

        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = vec_env.device
        self.asymmetric = vec_env.num_states > 0

        self.desired_kl = train_param.get("desired_kl", None)
        self.schedule = train_param.get("schedule", "fixed")
        self.step_size = train_param["optim_stepsize"]

        # PPO components
        self.vec_env = vec_env
        self.single_observation_space_shape = (self.observation_space.shape[0]//2,)
        self.single_action_space_shape = (self.action_space.shape[0]//2,)
        self.actor_critic = actor_critic_class(
            self.single_observation_space_shape,
            self.state_space.shape,
            self.single_action_space_shape,
            self.init_noise_std,
            self.model_cfg,
            asymmetric=self.asymmetric,
            use_pc=self.is_vision,
        )
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(
            self.vec_env.num_envs,
            self.num_transitions_per_env,
            self.single_observation_space_shape,
            self.state_space.shape,
            self.single_action_space_shape,
            self.device,
            self.sampler,
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size)        


   
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices] if self.asymmetric else None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                (
                    actions_log_prob_batch,
                    entropy_batch,
                    value_batch,
                    mu_batch,
                    sigma_batch,
                ) = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == "adaptive":
                    kl = torch.sum(
                        sigma_batch - old_sigma_batch 
                        + (torch.square(old_sigma_batch.exp())+ torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss


class IPPO(nn.Module):
    '''indepenent PPO'''
    def __init__(
        self,
        vec_env,
        actor_critic_class,
        train_param,
        log_dir="run",
        apply_reset=False,
        is_vision=False,
    ):
        super(IPPO, self).__init__()
        # environment parameters
        self.vec_env = vec_env
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.device = vec_env.device
        # agent
        self.left_agent = IPPOAgent(
            vec_env,
            actor_critic_class,
            train_param,
            is_vision,
        )
        self.right_agent = IPPOAgent(
            vec_env,
            actor_critic_class,
            train_param,
            is_vision,
        )
        # self.left_obs_indices = list(range(0,22))+list(range(44,66))+list(range(88,100))+list(range(112,134))+list(range(156, 169)) + list(range(182,189)) + list(range(196,211))
        # self.right_obs_indices = list(range(22,44))+list(range(66,88))+list(range(100,112))+list(range(134,156))+list(range(169,182)) + list(range(189,196)) + list(range(211,226))
        self.left_obs_indices, self.right_obs_indices = vec_env.get_obs_idx()
        assert self.left_agent.single_observation_space_shape[0] == len(self.left_obs_indices)
        assert self.right_agent.single_observation_space_shape[0] == len(self.right_obs_indices)

        # training params
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.apply_reset = apply_reset
        self.is_testing = train_param["test"]
        self.num_learning_iterations = train_param["max_iterations"]
        self.print_log = train_param["print_log"]
        self.num_transitions_per_env = train_param["nsteps"]
        self.save_interval = train_param["save_interval"]
        
        # Log
        self.log_dir = log_dir
        if not self.is_testing:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def test(self, path):
        self.load(path)
        self.eval()

    def load(self, path):
        assert os.path.isfile(path) or os.path.isdir(path), f"{path} is not a valid path"
        if os.path.isdir(path):
            from glob import glob
            from natsort import natsorted
            paths = natsorted(glob(os.path.join(path, "*.pt")))
            if len(paths):
                path = paths[-1]
            else:
                print(f"Checkpoint directory {path} is empty")
                self.current_learning_iteration = 0
                self.train()
                return
            
        saved_ckpt = torch.load(path, map_location=self.device)
        if 'optimizer_state_dict' in saved_ckpt:
            left_optimizer_state_dict, right_optimizer_state_dict = saved_ckpt["optimizer_state_dict"]
            self.left_agent.optimizer.load_state_dict(left_optimizer_state_dict)
            self.right_agent.optimizer.load_state_dict(right_optimizer_state_dict)
        self.load_state_dict(saved_ckpt["model_state_dict"])
        self.current_learning_iteration = int(left_optimizer_state_dict['state'][1]['step'].item()//20)
        self.train()
        print(f"Loaded checkpoint from {path}")

    def save(self, path):
        torch.save(dict(
            model_state_dict=self.state_dict(),
            optimizer_state_dict=[self.left_agent.optimizer.state_dict(), self.right_agent.optimizer.state_dict()],
        ), path)

    def run(self):
        num_learning_iterations = self.num_learning_iterations
        if self.is_testing:
            self.vec_env.random_time = False
        current_obs = self.vec_env.reset()["obs"]
        current_states = self.vec_env.get_state()

        if self.is_testing:
            for i in range(self.vec_env.max_episode_length):
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()["obs"]
                    # Compute the action
                    left_actions = self.left_agent.actor_critic.act_inference(current_obs[:, self.left_obs_indices])
                    right_actions = self.right_agent.actor_critic.act_inference(current_obs[:, self.right_obs_indices])
                    left_actions = torch.cat((left_actions, right_actions), dim=1)
                    # Step the vec_environment
                    next_obs_dict, rews, dones, infos = self.vec_env.step(left_actions)
                    next_obs = next_obs_dict["obs"]
                    current_obs.copy_(next_obs)
                if i == self.vec_env.max_episode_length - 2:
                    success_rate = self.vec_env.successes.sum() / self.vec_env.num_envs
            print("success_rate:", success_rate.item())
            exit()

        else:
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(
                self.vec_env.num_envs, dtype=torch.float, device=self.device
            )
            cur_episode_length = torch.zeros(
                self.vec_env.num_envs, dtype=torch.float, device=self.device
            )

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()["obs"]
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    left_obs = current_obs[:, self.left_obs_indices]
                    right_obs = current_obs[:, self.right_obs_indices]
                    left_actions, left_actions_log_prob, left_values, left_mu, left_sigma = self.left_agent.actor_critic.act(left_obs, current_states)
                    right_actions, right_actions_log_prob, right_values, right_mu, right_sigma = self.right_agent.actor_critic.act(right_obs, current_states)
                    actions = torch.cat((left_actions, right_actions), dim=1)
                    # Step the vec_environment
                    with torch.no_grad():
                        next_obs_dict, rews, dones, infos = self.vec_env.step(actions)
                        next_obs = next_obs_dict["obs"]
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    left_rews = infos["left_reward"]
                    self.left_agent.storage.add_transitions(
                        left_obs,
                        current_states,
                        left_actions,
                        left_rews,
                        dones,
                        left_values,
                        left_actions_log_prob,
                        left_mu,
                        left_sigma,
                    )
                    right_rews = infos["right_reward"]
                    self.right_agent.storage.add_transitions(
                        right_obs,
                        current_states,
                        right_actions,
                        right_rews,
                        dones,
                        right_values,
                        right_actions_log_prob,
                        right_mu,
                        right_sigma,
                    )
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)
                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, left_last_values, _, _ = self.left_agent.actor_critic.act(left_obs, current_states)
                _, _, right_last_values, _, _ = self.right_agent.actor_critic.act(right_obs, current_states)
                stop = time.time()
                collection_time = stop - start
                left_mean_trajectory_length, left_mean_reward = self.left_agent.storage.get_statistics()
                right_mean_trajectory_length, right_mean_reward = self.right_agent.storage.get_statistics()

                # Learning step
                start = stop
                self.left_agent.storage.compute_returns(left_last_values, self.left_agent.gamma, self.left_agent.lam)
                self.right_agent.storage.compute_returns(right_last_values, self.right_agent.gamma, self.right_agent.lam)
                left_mean_value_loss, left_mean_surrogate_loss = self.left_agent.update()
                right_mean_value_loss, right_mean_surrogate_loss = self.right_agent.update()
                self.left_agent.storage.clear()
                self.right_agent.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )
        for side in ['left', 'right']:
            self.writer.add_scalar(f"IPPO/{side}_mean_reward", locs[f"{side}_mean_reward"], locs["it"])
            self.writer.add_scalar(f"IPPO/{side}_mean_episode_length",locs[f"{side}_mean_trajectory_length"],locs["it"],)
            self.writer.add_scalar(f"IPPO/{side}_value_function", locs[f"{side}_mean_value_loss"], locs["it"])
            self.writer.add_scalar(f"IPPO/{side}_mean_surrogate_loss", locs[f"{side}_mean_surrogate_loss"], locs["it"])

        fps = int(
            self.num_transitions_per_env
            * self.vec_env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )
        self.writer.add_scalar("Computation/fps", fps, locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)
    