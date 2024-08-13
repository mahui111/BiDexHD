import os
import os.path as osp
import time
from gym import spaces
from gym.spaces import Space
import numpy as np
import statistics
import yaml
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ..common import DaggerStorage, ActorCritic

class DaggerValue(nn.Module):
    def __init__(
        self,
        vec_env,
        train_param,
        expert_class,
        log_dir='run',
    ):
        super().__init__()
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        
        self.vec_env = vec_env
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        # TODO: dagger obs:'dofps+dofvel+ftps+lastact+objstate+palmpose+relps+meshpc'
        self.num_pc_flatten = train_param.policy['numDownsample'] * train_param.policy['numEachPoint']
        self.expert_left_obs_indices = list(range(0, 22)) + list(range(44, 66)) + list(range(88, 100)) + list(range(112, 134)) + list(range(156, 169)) + list(range(182, 189)) + list(range(196, 211))
        self.expert_right_obs_indices = list(range(22, 44)) + list(range(66, 88)) + list(range(100, 112)) + list(range(134, 156)) + list(range(169, 182)) + list(range(189, 196)) + list(range(211, 226))
        self.student_left_robostate_indices = list(range(0, 22)) + list(range(88, 100)) + list(range(112, 134)) + list(range(182, 185))
        self.student_left_pointcloud_indices = list(range(226, 226 + self.num_pc_flatten))
        self.student_left_obs_indices = self.student_left_robostate_indices + self.student_left_pointcloud_indices
        self.student_right_robostate_indices = list(range(22, 44)) + list(range(100, 112)) + list(range(134, 156)) + list(range(189, 192))
        self.student_right_pointcloud_indices = list(range(226 + self.num_pc_flatten, 226 + self.num_pc_flatten * 2))
        self.student_right_obs_indices = self.student_right_robostate_indices + self.student_right_pointcloud_indices
        assert 226 + self.num_pc_flatten * 2 == self.observation_space.shape[0]
        assert len(self.expert_left_obs_indices) == len(self.expert_right_obs_indices) and len(self.student_left_obs_indices) == len(self.student_right_obs_indices)
        self.single_observation_space_shape = (len(self.student_left_obs_indices),)
        self.single_action_space_shape = (self.action_space.shape[0] // 2,)
        
        # DAGGER parameters
        self.value_loss_cfg = train_param['value_loss']
        self.num_learning_epochs = train_param["noptepochs"]
        self.num_mini_batches = train_param["nminibatches"]
        self.num_transitions_per_env = train_param["nsteps"]
        self.num_learning_iterations = train_param["max_iterations"]
        self.save_interval = train_param["save_interval"]
        self.gamma = train_param["gamma"]
        self.lam = train_param["lam"]
        self.sampler = train_param.get("sampler", "sequential")
        
        self.device = vec_env.device
        
        # Log
        self.log_dir = log_dir
        self.print_log = train_param["print_log"]
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = train_param["test"]
        self.checkpoint_path = train_param["checkpoint"]
        self.current_learning_iteration = 0
        
        # student 
        init_noise_std = train_param["init_noise_std"]
        self.left_actor_critic = ActorCritic(self.single_observation_space_shape, self.state_space.shape, 
                                             self.single_action_space_shape, init_noise_std, train_param.policy, use_pc=True,
                                             robostate_indices=self.student_left_robostate_indices, pointcloud_indices=self.student_left_pointcloud_indices)
        self.left_actor_critic.to(self.device)
        self.left_optimizer = optim.Adam(self.left_actor_critic.parameters(), lr=train_param["optim_stepsize"])
        self.right_actor_critic = ActorCritic(self.single_observation_space_shape, self.state_space.shape, 
                                              self.single_action_space_shape, init_noise_std, train_param.policy, use_pc=True,
                                              robostate_indices=self.student_right_robostate_indices, pointcloud_indices=self.student_right_pointcloud_indices)
        self.right_actor_critic.to(self.device)
        self.right_optimizer = optim.Adam(self.right_actor_critic.parameters(), lr=train_param["optim_stepsize"])

        if not self.is_testing:
            # multi_expert
            self.expert_list = []
            for expert_cfg in train_param['expert']:
                expert = expert_class(vec_env, train_param, None, obs_type=train_param['expertObservationType'])
                expert.to(self.device)
                expert.load(expert_cfg['path'])
                self.expert_list.append(expert)
            self.num_task = len(self.expert_list)
            self.storage = DaggerStorage(self.vec_env.num_envs, self.num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.single_action_space_shape, self.device, self.sampler)
    
    def test(self, path):
        self.load(path)
        self.eval()

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def run(self,):
        self.train()
        num_learning_iterations = self.num_learning_iterations
        if self.is_testing:
            self.vec_env.random_time = False
        current_obs = self.vec_env.reset()["obs"]
        current_states = self.vec_env.get_state()

        if self.is_testing:
            eplen = self.vec_env.max_episode_length
            for i in range(eplen):
                with torch.no_grad():
                    # Compute the action
                    stu_actions = torch.cat([
                        self.left_actor_critic.act_inference(current_obs),
                        self.right_actor_critic.act_inference(current_obs),
                    ],dim=1)
                    # Step the vec_environment
                    next_obs_dict, rews, dones, infos = self.vec_env.step(stu_actions)
                    next_obs = next_obs_dict["obs"]
                    current_obs.copy_(next_obs)
                # if i == self.vec_env.max_episode_length - 2:
                #     print('stage 1 left success:', self.vec_env.stage1_left_successes.mean().item())
                #     print('stage 1 right success:', self.vec_env.stage1_right_successes.mean().item())
                #     print('stage 1 success:', self.vec_env.stage1_successes.mean().item())
                #     print('stage 2 left success:', self.vec_env.stage2_left_successes.mean().item())
                #     print('stage 2 right success:', self.vec_env.stage2_right_successes.mean().item())
            exit()
        else:
            retbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            episode_return = []
            episode_length = []
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            for it in range(1 + self.current_learning_iteration, 1 + num_learning_iterations):
                start = time.time()
                ep_infos = []
                for i in range(self.num_transitions_per_env):
                    # Compute expert action
                    expert_left_actions, expert_left_values, expert_right_actions, expert_right_values = self.expert_batch_act(current_obs)
                    # Compute the action
                    stu_left_actions, _, stu_left_values, _, _ = self.left_actor_critic.act(current_obs)
                    stu_right_actions, _, stu_right_values, _, _ = self.right_actor_critic.act(current_obs)
                    stu_actions = torch.cat([stu_left_actions, stu_right_actions], dim=1)
                    # Step the vec_environment
                    with torch.no_grad():
                        next_obs_dict, rews, dones, infos = self.vec_env.step(stu_actions)
                        next_obs = next_obs_dict["obs"]
                    current_obs.copy_(next_obs)
                    current_states.copy_(self.vec_env.get_state())
                    left_rews, right_rews = infos["left_reward"], infos["right_reward"]
                    # Record the transition
                    self.storage.add_transitions(
                        current_obs, current_states, 
                        stu_left_actions, stu_right_actions,
                        left_rews, right_rews, dones,
                        stu_left_values, stu_right_values,
                        expert_left_actions, expert_right_actions,
                        expert_left_values, expert_right_values
                    )

                    # Book keeping
                    ep_infos.append(infos)
                    if self.print_log:  # calculate metrics
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        episode_return.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                if self.print_log:
                    retbuffer.extend(episode_return)
                    lenbuffer.extend(episode_length)
                _, _, last_left_values, _, _ = self.left_actor_critic.act(next_obs)
                _, _, last_right_values, _, _ = self.right_actor_critic.act(next_obs)
                stop = time.time()
                collection_time = stop - start
                mean_trajectory_length, left_mean_reward, right_mean_reward = self.storage.get_statistics()
                # Learning step
                start = stop
                self.storage.compute_returns(last_left_values, last_right_values, self.gamma, self.lam)
                mean_policy_loss, mean_value_loss, loss_info_dict = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                # save
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "dagger_{}.pt".format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, "dagger_{}.pt".format(num_learning_iterations)))

    def update(self):
        mean_policy_loss = 0
        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        mean_value_loss = 0
        for _ in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                expert_left_actions_batch = self.storage.expert_left_actions.view(-1, self.storage.left_actions.size(-1))[indices]
                expert_right_actions_batch = self.storage.expert_right_actions.view(-1, self.storage.right_actions.size(-1))[indices]
                # Policy loss
                cur_left_actions_batch = self.left_actor_critic.act(obs_batch, grad=True)[3]
                cur_right_actions_batch = self.right_actor_critic.act(obs_batch, grad=True)[3]
                left_action_loss = F.huber_loss(cur_left_actions_batch, expert_left_actions_batch)
                right_action_loss = F.huber_loss(cur_right_actions_batch, expert_right_actions_batch)
                # Value loss
                if self.value_loss_cfg['apply']:
                    left_action_batch = self.storage.left_actions.view(-1, self.storage.left_actions.size(-1))[indices]
                    right_action_batch = self.storage.right_actions.view(-1, self.storage.right_actions.size(-1))[indices]
                    left_returns_batch = self.storage.left_returns.view(-1, 1)[indices]
                    right_returns_batch = self.storage.right_returns.view(-1, 1)[indices]
                    expert_left_values_batch = self.storage.expert_left_values.view(-1, 1)[indices]
                    expert_right_values_batch = self.storage.expert_right_values.view(-1, 1)[indices]
                    cur_left_value_batch = self.left_actor_critic.evaluate(obs_batch, None, left_action_batch)[2]
                    cur_right_value_batch = self.right_actor_critic.evaluate(obs_batch, None, right_action_batch)[2]
                    if self.value_loss_cfg['use_clipped_value_loss']:
                        left_value_clipped = expert_left_values_batch + (cur_left_value_batch - expert_left_values_batch).clamp(-self.value_loss_cfg['clip_range'], self.value_loss_cfg['clip_range'])
                        left_value_losses = (cur_left_value_batch - left_returns_batch).pow(2)
                        left_value_losses_clipped = (left_value_clipped - left_returns_batch).pow(2)
                        left_value_loss = torch.max(self.symlog(left_value_losses), self.symlog(left_value_losses_clipped)).mean()
                        right_value_clipped = expert_right_values_batch + (cur_right_value_batch - expert_right_values_batch).clamp(-self.value_loss_cfg['clip_range'], self.value_loss_cfg['clip_range'])
                        right_value_losses = (cur_right_value_batch - right_returns_batch).pow(2)
                        right_value_losses_clipped = (right_value_clipped - right_returns_batch).pow(2)
                        right_value_loss = torch.max(self.symlog(right_value_losses), self.symlog(right_value_losses_clipped)).mean()
                    else:
                        left_value_loss = (left_returns_batch - cur_left_value_batch).pow(2).mean()
                        right_value_loss = (right_returns_batch - cur_right_value_batch).pow(2).mean()
                else:
                    left_value_loss = torch.zeros_like(left_action_loss)
                    right_value_loss = torch.zeros_like(right_action_loss)
                # Gradient step
                left_loss = left_action_loss + left_value_loss * self.value_loss_cfg['value_loss_coef']
                self.left_optimizer.zero_grad()
                left_loss.backward()
                self.left_optimizer.step()
                right_loss = right_action_loss + right_value_loss * self.value_loss_cfg['value_loss_coef']
                self.right_optimizer.zero_grad()
                right_loss.backward()
                self.right_optimizer.step()

                mean_policy_loss += 0.5 * (left_action_loss.item() + right_action_loss.item())
                mean_value_loss += 0.5 * (left_value_loss.item() + right_value_loss.item())
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_policy_loss /= num_updates
        mean_value_loss /= num_updates
        return mean_policy_loss, mean_value_loss, dict(left_action_loss=left_action_loss.item(), right_action_loss=right_action_loss.item(), left_value_loss=left_value_loss.item(), right_value_loss=right_value_loss.item())

    def expert_batch_act(self, current_obs):
        with torch.no_grad():
            batch_expert_left_actions, batch_expert_left_values = torch.zeros(current_obs.shape[:1] + self.single_action_space_shape, device=self.device), torch.zeros(current_obs.shape[:1] + (1,), device=self.device)
            batch_expert_right_actions, batch_expert_right_values = torch.zeros(current_obs.shape[:1] + self.single_action_space_shape, device=self.device), torch.zeros(current_obs.shape[:1] + (1,), device=self.device)
            for i_task in range(len(self.expert_list)):
                batch_expert_left_actions[i_task::self.num_task], _, batch_expert_left_values[i_task::self.num_task], _, _ = self.expert_list[i_task].left_agent.actor_critic.act(current_obs[i_task::self.num_task, self.expert_left_obs_indices])
                batch_expert_right_actions[i_task::self.num_task], _, batch_expert_right_values[i_task::self.num_task], _, _ = self.expert_list[i_task].right_agent.actor_critic.act(current_obs[i_task::self.num_task, self.expert_right_obs_indices])
        return batch_expert_left_actions, batch_expert_left_values, batch_expert_right_actions, batch_expert_right_values
    
    @staticmethod
    def symlog(x):
        return x.sign() * x.abs().log1p()

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/policy', locs['mean_policy_loss'], locs['it'])
        for k, v in locs['loss_info_dict'].items():
            self.writer.add_scalar(f'Loss/{k}', v, locs['it'])

        if len(locs['retbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['retbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['retbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/left_mean_reward/step', locs['left_mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/right_mean_reward/step', locs['right_mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))
        self.writer.add_scalar("Computation/fps", fps, locs["it"])
        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['retbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['retbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean left reward/step:':>{pad}} {locs['left_mean_reward']:.2f}\n"""
                          f"""{'Mean right reward/step:':>{pad}} {locs['right_mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Mean left reward/step:':>{pad}} {locs['left_mean_reward']:.2f}\n"""
                          f"""{'Mean right reward/step:':>{pad}} {locs['right_mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)  
