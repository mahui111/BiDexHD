import os, pickle
import os.path as osp
import time
import open3d as o3d
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

from ..common import BCStorage, ActorCritic


class Visualizer3D:
    def reset(self):
        #create a default coordinate frame  
        self.geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]

    def __init__(self):
        self.reset()

    def visualize_point_clouds(self, points=None, poses=None, colors=None):
        assert points is not None or poses is not None
        # Extract the translation components from the (4, 4) transformation matrices
        positions = np.array([pose[:3, 3] for pose in poses]) if points is None else points
        
        # Create the point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(positions)
        
        # Create colors array
        num_points = len(positions)
        # Assign (N,3) colors based on the point index
        if colors is None:
            colors = np.zeros((num_points, 3))
            colors[:5] = [0, 0, 1]  # Blue for the first 5 points
            colors[-5:] = [1, 0, 0]  # Red for the last 5 points
            colors[5:-5] = [0.1, 0.7, 0.1]  # Green for the middle points
        
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.geometries.append(point_cloud)

    def visualize_poses(self, poses, size=0.01):
        self.geometries.extend([o3d.geometry.TriangleMesh.create_coordinate_frame(size=size).transform(pose) for pose in poses])

    def draw(self, discard=True):
        o3d.visualization.draw_geometries(self.geometries)
        if discard:
            self.reset()


class BCPPO(nn.Module):
    def __init__(
        self,
        vec_env,
        train_param,
        dataset_file,
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
        # teacher obs: 'dofps+dofvel+ftps+lastact+objstate+palmpose+relps+objlabel'
        self.num_pc_flatten = train_param.policy['numDownsample'] * train_param.policy['numEachPoint']
        self.expert_left_robostate_indices = list(range(0, 22)) + list(range(44, 66)) + list(range(88, 100)) + list(range(112, 134)) + list(range(156, 169)) + list(range(182, 189)) + list(range(196, 211))
        self.expert_left_pointcloud_indices = []
        self.expert_left_objlabel_indices = list(range(226 + self.num_pc_flatten * 2, 227 + self.num_pc_flatten * 2))
        self.expert_right_robostate_indices = list(range(22, 44)) + list(range(66, 88)) + list(range(100, 112)) + list(range(134, 156)) + list(range(169, 182)) + list(range(189, 196)) + list(range(211, 226))
        self.expert_right_pointcloud_indices = []
        self.expert_right_objlabel_indices = list(range(227 + self.num_pc_flatten * 2, self.num_pc_flatten * 2 + 228))
        # student obs: 'dofps+ftps+lastact+palmpose+meshpc+objlabel'
        self.student_left_robostate_indices = list(range(0, 22)) + list(range(88, 100)) + list(range(112, 134)) + list(range(182, 189))
        self.student_left_pointcloud_indices = list(range(226, 226 + self.num_pc_flatten))
        # self.student_left_objlabel_indices = list(range(226 + self.num_pc_flatten * 2, 226 + self.num_pc_flatten * 2 + 1))
        self.student_left_instrlabel_indices = []  
        self.student_left_futureobjps_indices = list(range(self.num_pc_flatten * 2 + 228, self.num_pc_flatten * 2 + 243))
        self.student_left_obs_indices = self.student_left_robostate_indices + self.student_left_pointcloud_indices + self.student_left_instrlabel_indices
        self.student_right_robostate_indices = list(range(22, 44)) + list(range(100, 112)) + list(range(134, 156)) + list(range(189, 196))
        self.student_right_pointcloud_indices = list(range(226 + self.num_pc_flatten, 226 + self.num_pc_flatten * 2))
        # self.student_right_objlabel_indices = list(range(226 + self.num_pc_flatten * 2 + 1, self.num_pc_flatten * 2 + 228))
        self.student_right_instrlabel_indices = []  
        self.student_right_futureobjps_indices = list(range(self.num_pc_flatten * 2 + 243, self.num_pc_flatten * 2 + 258))
        self.student_right_obs_indices = self.student_right_robostate_indices + self.student_right_pointcloud_indices + self.student_right_instrlabel_indices
        assert len(self.expert_left_robostate_indices) == len(self.expert_right_robostate_indices) and len(self.student_left_obs_indices) == len(self.student_right_obs_indices)
        self.stu_observation_space_shape = (len(self.student_left_obs_indices) + len(self.student_right_obs_indices),)
        self.stu_action_space_shape = self.action_space.shape
        
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
        self.max_grad_norm = train_param.get("max_grad_norm", 0.5)
        # self.clip_action = train_param.get("clip_action", False)
        self.device = vec_env.device
        
        # Log
        self.log_dir = log_dir
        self.print_log = False
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = train_param["test"]
        self.checkpoint_path = train_param["checkpoint"]
        self.current_learning_iteration = 0
        
        # student 
        init_noise_std = train_param["init_noise_std"]
        self.actor_critic = ActorCritic(self.stu_observation_space_shape, self.state_space.shape, 
                                        self.stu_action_space_shape, init_noise_std, train_param.policy,
                                        robostate_indices=self.student_left_robostate_indices + self.student_right_robostate_indices, 
                                        pointcloud_indices=self.student_left_pointcloud_indices + self.student_right_pointcloud_indices,
                                        objlabel_indices=[],
                                        futureobjps_indices=self.student_left_futureobjps_indices + self.student_right_futureobjps_indices,
                                        )
        
        self.actor_critic.to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=train_param["optim_stepsize"])

        # loss
        if train_param['lossFunc'] == 'huber':
            self.criterion = nn.HuberLoss(delta=0.01)
        elif train_param['lossFunc'] == 'l1':
            self.criterion = nn.SmoothL1Loss()

        if not self.is_testing:
            with open(dataset_file, 'rb') as f:
                BCdataset = pickle.load(f)
                # pad the dataset: (N, maxLen, D) -> (N * maxLen, D)
                maxlen = max([len(BCdataset['obs'][e]) for e in vec_env.train_task_ids])
                self.observations = torch.tensor([BCdataset['obs'][e] + [BCdataset['obs'][e][-1]] * (maxlen - len(BCdataset['obs'][e])) for e in vec_env.train_task_ids], device=self.device, dtype=torch.float, requires_grad=False).flatten(0,1)
                self.actions = torch.tensor([BCdataset['act'][e] + [BCdataset['act'][e][-1]] * (maxlen - len(BCdataset['act'][e])) for e in vec_env.train_task_ids], device=self.device, dtype=torch.float, requires_grad=False).flatten(0,1)
                # visualizer = Visualizer3D()
                # visualizer.visualize_point_clouds(self.observations[2,0,self.student_left_pointcloud_indices].detach().cpu().numpy().reshape(-1,3))
                # visualizer.visualize_point_clouds(self.observations[2,0,self.student_right_pointcloud_indices].detach().cpu().numpy().reshape(-1,3))
                # visualizer.draw(True)
                # visualizer.visualize_point_clouds(self.observations[5,0,self.student_left_pointcloud_indices].detach().cpu().numpy().reshape(-1,3))
                # visualizer.visualize_point_clouds(self.observations[5,0,self.student_right_pointcloud_indices].detach().cpu().numpy().reshape(-1,3))
                # visualizer.draw(True)
                # breakpoint()
            self.storage = BCStorage(self.observations, self.actions, self.device, self.sampler)
    
    def test(self, path):
        self.load(path)
        self.eval()

    def load(self, path):
        ckpt = torch.load(path)
        self.actor_critic.load_state_dict(ckpt)

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self,):
        self.train()
        num_learning_iterations = self.num_learning_iterations
        if self.is_testing:
            self.vec_env.random_time = False
        current_obs = self.vec_env.reset()["obs"]
        # current_obs = torch.cat([current_obs, self.vec_env.verb_category.view(-1,1) / 255.], dim=-1)
        current_states = self.vec_env.get_state()

        if self.is_testing:
            eplen = self.vec_env.max_episode_length
            sr1, sr2, ne = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device), torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device), 1e-8+torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            for i in range(1, 1 + eplen * 10):
                with torch.no_grad():
                    # Compute the action
                    stu_actions = self.actor_critic.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs_dict, rews, dones, infos = self.vec_env.step(stu_actions)
                    next_obs = next_obs_dict["obs"]
                    current_obs.copy_(next_obs)

                termination = torch.logical_or(self.vec_env.progress_buf >= self.vec_env.max_episode_length, self.vec_env.is_expect_end)  # self.vec_env.reset_buf > 0
                ne = torch.where(termination, ne + 1, ne)
                sr1 = torch.where(termination, sr1 + self.vec_env.stage1_successes, sr1)
                sr2 = torch.where(termination, sr2 + self.vec_env.stage2_successes, sr2)
                if i % eplen == 0: 
                    print(f"step {i}")
                    # 1. log success rate for train & test
                    print('-'*90 + f'\nTrain & Test')
                    train_env_ids = [l for l in range(self.vec_env.num_envs) if l % self.vec_env.num_task in self.vec_env.train_task_ids]
                    test_env_ids = [l for l in range(self.vec_env.num_envs) if l % self.vec_env.num_task in self.vec_env.test_task_ids]
                    print(f"training set\t| stage1_successes: {sr1[train_env_ids].sum() / ne[train_env_ids].sum()}\t| stage2_successes: {sr2[train_env_ids].sum() / ne[train_env_ids].sum()}")
                    # print(f"testing set\t| stage1_successes: {avgsr1[test_env_ids].mean()}\t| stage2_successes: {avgsr2[test_env_ids].mean()}")
                    print(f"testing seen\t| stage1_successes: {sr1[self.vec_env.types == 1].sum() / ne[self.vec_env.types == 1].sum()}\t| stage2_successes: {sr2[self.vec_env.types == 1].sum() / ne[self.vec_env.types == 1].sum()}")
                    print(f"testing unseen\t| stage1_successes: {sr1[self.vec_env.types == 2].sum() / ne[self.vec_env.types == 2].sum()}\t| stage2_successes: {sr2[self.vec_env.types == 2].sum() / ne[self.vec_env.types == 2].sum()}")
                    # log total
                    # print('-'*90 + f'\n Total')
                    # print(f"Average\t| stage1_successes: {avgsr1.mean()}\t| stage2_successes: {avgsr2.mean()}")
                    # print(f"Average Nonzero\t| stage1_successes: {avgsr1[avgsr1>0].mean()}\t| stage2_successes: {avgsr2[avgsr2>0].mean()}")
                    # print('-'*90)
            exit()
        else:
            for it in range(1 + self.current_learning_iteration, 1 + num_learning_iterations):
                start = time.time()
                mean_policy_loss = self.update()
                stop = time.time()
                learn_time = stop - start
                self.log(locals())
                # save
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "dagger_{}.pt".format(it)))
            self.save(os.path.join(self.log_dir, "dagger_{}.pt".format(num_learning_iterations)))

    def update(self):
        mean_policy_loss = 0
        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for _ in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations[indices]
                act_batch = self.storage.actions[indices]
                # Policy loss
                cur_actions_batch = self.actor_critic.act(obs_batch, grad=True)[3]
                action_loss = self.criterion(cur_actions_batch, act_batch)
                # Gradient step
                self.optimizer.zero_grad()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                action_loss.backward()
                self.optimizer.step()

                mean_policy_loss += action_loss.item()
                
        mean_policy_loss /= (self.num_learning_epochs * self.num_mini_batches)
        return mean_policy_loss
    
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['learn_time']
        iteration_time = locs['learn_time']

        ep_string = f''
        self.writer.add_scalar('Loss/policy', locs['mean_policy_loss'], locs['it'])

        fps = int(self.num_learning_epochs * self.num_mini_batches / (locs['learn_time']))
        self.writer.add_scalar("Computation/fps", fps, locs["it"])
        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        
        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)  
