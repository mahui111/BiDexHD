import numpy as np
import torch
from torch.utils.data.sampler import (
    BatchSampler,
    SequentialSampler,
    SubsetRandomSampler,
)


class RolloutStorage:

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        states_shape,
        actions_shape,
        device="cpu",
        sampler="sequential",
    ):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        self.states = torch.zeros(
            num_transitions_per_env, num_envs, *states_shape, device=self.device
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(
        self,
        observations,
        states,
        actions,
        rewards,
        dones,
        values,
        actions_log_prob,
        mu,
        sigma,
    ):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch

class DaggerStorage:
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        states_shape,
        actions_shape,
        device="cpu",
        sampler="sequential",
    ):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        self.states = torch.zeros(
            num_transitions_per_env, num_envs, *states_shape, device=self.device
        )
        self.left_rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.right_rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.left_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.right_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For PPO
        self.left_values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.left_returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.right_values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.right_returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        # For Dagger
        self.expert_left_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.expert_right_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.expert_left_values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.expert_right_values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(
        self,
        observations,
        states,
        left_actions, right_actions,
        left_rewards, right_rewards,
        dones,
        left_values, right_values,
        expert_left_actions, expert_right_actions,
        expert_left_values, expert_right_values,
    ):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.left_actions[self.step].copy_(left_actions)
        self.right_actions[self.step].copy_(right_actions)
        self.left_rewards[self.step].copy_(left_rewards.view(-1, 1))
        self.right_rewards[self.step].copy_(right_rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.left_values[self.step].copy_(left_values)
        self.right_values[self.step].copy_(right_values)
        self.expert_left_actions[self.step].copy_(expert_left_actions)
        self.expert_right_actions[self.step].copy_(expert_right_actions)
        self.expert_left_values[self.step].copy_(expert_left_values)
        self.expert_right_values[self.step].copy_(expert_right_values)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_left_values, last_right_values, gamma, lam):
        left_advantage, right_advantage = 0, 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_left_values = last_left_values if step == self.num_transitions_per_env - 1 else self.left_values[step + 1]
            next_right_values = last_right_values if step == self.num_transitions_per_env - 1 else self.right_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            left_delta = self.left_rewards[step] + next_is_not_terminal * gamma * next_left_values - self.left_values[step]
            right_delta = self.right_rewards[step] + next_is_not_terminal * gamma * next_right_values - self.right_values[step]
            left_advantage = left_delta + next_is_not_terminal * gamma * lam * left_advantage
            right_advantage = right_delta + next_is_not_terminal * gamma * lam * right_advantage
            self.left_returns[step] = left_advantage + self.left_values[step]
            self.right_returns[step] = right_advantage + self.right_values[step]

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((
            flat_dones.new_tensor([-1], dtype=torch.int64),
            flat_dones.nonzero(as_tuple=False)[:, 0],
        ))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.left_rewards.mean(), self.right_rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch

class M2DaggerStorage:
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        states_shape,
        actions_shape,
        device="cpu",
        sampler="sequential",
    ):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        self.states = torch.zeros(
            num_transitions_per_env, num_envs, *states_shape, device=self.device
        )
        self.left_rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.right_rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.left_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.right_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        self.expert_ids = torch.zeros(
            num_transitions_per_env, num_envs, device=self.device
        ).long()

        # For PPO
        self.left_values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.left_returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.right_values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.right_returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(
        self,
        observations,
        states,
        left_actions, right_actions,
        left_rewards, right_rewards,
        dones,
        left_values, right_values,
        expert_id
    ):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.left_actions[self.step].copy_(left_actions)
        self.right_actions[self.step].copy_(right_actions)
        self.left_rewards[self.step].copy_(left_rewards.view(-1, 1))
        self.right_rewards[self.step].copy_(right_rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.left_values[self.step].copy_(left_values)
        self.right_values[self.step].copy_(right_values)
        self.expert_ids[self.step].copy_(expert_id)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_left_values, last_right_values, gamma, lam):
        left_advantage, right_advantage = 0, 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_left_values = last_left_values if step == self.num_transitions_per_env - 1 else self.left_values[step + 1]
            next_right_values = last_right_values if step == self.num_transitions_per_env - 1 else self.right_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            left_delta = self.left_rewards[step] + next_is_not_terminal * gamma * next_left_values - self.left_values[step]
            right_delta = self.right_rewards[step] + next_is_not_terminal * gamma * next_right_values - self.right_values[step]
            left_advantage = left_delta + next_is_not_terminal * gamma * lam * left_advantage
            right_advantage = right_delta + next_is_not_terminal * gamma * lam * right_advantage
            self.left_returns[step] = left_advantage + self.left_values[step]
            self.right_returns[step] = right_advantage + self.right_values[step]

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((
            flat_dones.new_tensor([-1], dtype=torch.int64),
            flat_dones.nonzero(as_tuple=False)[:, 0],
        ))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.left_rewards.mean(), self.right_rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
