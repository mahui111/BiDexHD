import os, json
import random
import pickle
import torch
import numpy as np
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class BiLeapHandGrasp(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]

        # NOTE don't use, scale factor of velocity based observations
        self.vel_obs_scale = 0.2
        # NOTE don't use, scale factor of force and torque based observations
        self.force_torque_obs_scale = 10.0

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.arm_controller = self.cfg["env"]["armController"]
        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_vis = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)
        self.goal_height = self.cfg["env"].get("goalHeight", 0.6)

        self.palm_offset = self.cfg["env"]["palm_offset"]
        self.fingertip_offset = self.cfg["env"]["finger_offset"]
        self.thumb_offset = self.cfg["env"]["thumb_offset"]

        self.obs_type = self.cfg["env"]["observationType"]
        self.multi_task = self.cfg["env"]["multiTask"]

        assert self.arm_controller in ["ik", "qpos"]
        assert self.obs_type in ["full_no_vel", "full", "full_state"]

        # need to set the number of observations according to the robot
        self.num_obs_dict = {
            "full": 262,
        }

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = self.num_obs_dict[self.obs_type] if self.asymmetric_obs else 0
        self.cfg["env"]["numActions"] = 44

        if self.arm_controller == "ik":  # use rotation 6D representation
            self.cfg["env"]["numObservations"] += 3
            self.cfg["env"]["numStates"] += 3 if self.asymmetric_obs else 0
            self.cfg["env"]["numActions"] += 3


        # need to set the names according to the robot
        self.palm = "palm_lower"
        self.fingertips = [
            "thumb_fingertip",
            "fingertip",
            "fingertip_2",
            "fingertip_3",
        ]
        self.arm_dof_names = [
            "arm_joint1",
            "arm_joint2",
            "arm_joint3",
            "arm_joint4",
            "arm_joint5",
            "arm_joint6",
        ]
        super().__init__(
            self.cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (control_freq_inv * self.dt))
            )
            print("Reset time: ", self.reset_time)
            print("Max episode length: ", self.max_episode_length)

        # viewer camera setup
        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
                self.num_envs, len(self.fingertips) * 6
            )
            # force tensor for each robot dof
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_robot_dofs
            )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.left_j_eef = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "left"))[:, self.left_eef_index - 1, :, :6]
        self.right_j_eef = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "right"))[:, self.right_eef_index - 1, :, :6]

        # create some wrapper tensors for different slices
        self.robot_default_dof_pos = torch.zeros(self.num_robot_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_robot_dofs]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.cfg["sim"]["up_axis"] == "z" else 1

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

        # if randamizing, apply once immediately on startup before the first sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _generate_urdf(self, object_dict):
        assert "id" in object_dict
        link_id = object_dict["id"]
        xyz = ' '.join(map(str, object_dict["xyz"])) if "xyz" in object_dict else '0 0 0'
        rpy = ' '.join(map(str, object_dict["rpy"])) if "rpy" in object_dict else '0 0 0'
        # TODO: modify the scale of object here
        scale = ' '.join(map(str, object_dict["scale"])) if "scale" in object_dict else "0.01 0.01 0.01"
        link_section = f"""
<robot name="objects_{link_id}">
<link name="link_{link_id}">
    <visual>
    <origin xyz="{xyz}" rpy="{rpy}"/>
    <geometry>
        <mesh filename="{link_id}_cm.obj" scale="{scale}"/>
    </geometry>
    <material name="">
        <color rgba="0.75 0.75 0.75 1"/>
    </material>
    </visual>
    <collision>
    <origin xyz="{xyz}" rpy="{rpy}"/>
    <geometry>
        <mesh filename="{link_id}_cm.obj" scale="{scale}"/>
    </geometry>
    </collision>
</link>
</robot>
"""
        return link_section
    
    def _create_urdf(self, tool_id, target_id, save_path):
        with open(os.path.join(save_path, "tool.urdf"), 'w') as urdf_file:
            urdf_file.write(self._generate_urdf(dict(id=tool_id)))
        
        with open(os.path.join(save_path, "object.urdf"), 'w') as urdf_file:
            urdf_file.write(self._generate_urdf(dict(id=target_id)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)


        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        left_asset, left_dof_props, self.left_palm_handle, self.left_fingertip_handles, self.left_eef_index, \
        self.left_arm_dof_indices, self.left_fingers_dof_indices, self.left_robot_dof_indices, \
        self.left_robot_dof_lower_limits, self.left_robot_dof_upper_limits, self.left_robot_dof_default_pos, self.left_robot_dof_default_vel = self._prepare_robot_asset(asset_root, self.cfg["env"]["asset"]["leftAssetFile"])
        right_asset, right_dof_props, self.right_palm_handle, self.right_fingertip_handles, self.right_eef_index,\
        self.right_arm_dof_indices, self.right_fingers_dof_indices, self.right_robot_dof_indices, \
        self.right_robot_dof_lower_limits, self.right_robot_dof_upper_limits, self.right_robot_dof_default_pos, self.right_robot_dof_default_vel = self._prepare_robot_asset(asset_root, self.cfg["env"]["asset"]["rightAssetFile"])    


        self.robot_dof_lower_limits = to_torch(self.left_robot_dof_lower_limits + self.right_robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.left_robot_dof_upper_limits + self.right_robot_dof_upper_limits, device=self.device)
        self.robot_dof_default_pos = to_torch(self.left_robot_dof_default_pos + self.right_robot_dof_default_pos, device=self.device)
        self.robot_dof_default_vel = to_torch(self.left_robot_dof_default_vel + self.right_robot_dof_default_vel, device=self.device)


        # get hand asset info
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(left_asset) + self.gym.get_asset_rigid_body_count(right_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(left_asset) + self.gym.get_asset_rigid_shape_count(right_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(left_asset) + self.gym.get_asset_dof_count(right_asset)
        # update right handles & dof_indices
        self.right_palm_handle += self.num_robot_bodies//2
        self.right_fingertip_handles = [i + self.num_robot_bodies//2 for i in self.right_fingertip_handles]
        self.right_arm_dof_indices = [i + self.num_robot_dofs//2 for i in self.right_arm_dof_indices]
        self.right_fingers_dof_indices = [i + self.num_robot_dofs//2 for i in self.right_fingers_dof_indices]
        self.right_robot_dof_indices = [i + self.num_robot_dofs//2 for i in self.right_robot_dof_indices]
        self.both_arm_dof_indices = to_torch(self.left_arm_dof_indices + self.right_arm_dof_indices, dtype=torch.long, device=self.device)
        self.both_fingers_dof_indices = to_torch(self.left_fingers_dof_indices + self.right_fingers_dof_indices, dtype=torch.long, device=self.device)
        self.both_robot_dof_indices = to_torch(self.left_robot_dof_indices + self.right_robot_dof_indices, dtype=torch.long, device=self.device)
        
        # object
        # self._prepare_dataset()
        object_asset = self._prepare_object_asset(asset_root, self.cfg["env"]["asset"]["objectAssetFile"])
        tool_asset = self._prepare_object_asset(asset_root, self.cfg["env"]["asset"]["toolAssetFile"])

        # get object asset info
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset) + self.gym.get_asset_rigid_body_count(tool_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset) + self.gym.get_asset_rigid_shape_count(tool_asset)
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset) + self.gym.get_asset_dof_count(tool_asset)

        # table
        table_asset, self.table_start_pose, side_panel_asset, side_panel_start_pose = self._prepare_table_asset()

        # initialize pose
        left_robot_start_pose = gymapi.Transform()
        left_robot_start_pose.p = gymapi.Vec3(-0.5, 0.3, 0.82)
        left_robot_start_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, np.pi)
        right_robot_start_pose = gymapi.Transform()
        right_robot_start_pose.p = gymapi.Vec3(-0.5, -0.3, 0.82)
        right_robot_start_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, np.pi)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0, 0.3, 0.3)
        object_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)
        tool_start_pose = gymapi.Transform()
        tool_start_pose.p = gymapi.Vec3(0, -0.3, 0.3)
        tool_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)


        self.envs = []
        self.left_robot_indices, self.right_robot_indices = [], []
        self.object_indices, self.tool_indices = [], []
        self.left_start_states, self.right_start_states = [], []
        self.object_init_states, self.tool_init_states = [], []

        if self.arm_controller == "ik":
            self.eef_idx = []
        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # aggregate size
            max_agg_bodies = self.num_robot_bodies + self.num_object_bodies + 2
            max_agg_shapes = self.num_robot_shapes + self.num_object_shapes + 2

            if self.aggregate_mode > 0:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # create robot actor
            left_robot_actor = self.gym.create_actor(env_ptr, left_asset, left_robot_start_pose, "left", i, -1, 0)
            right_robot_actor = self.gym.create_actor(env_ptr, right_asset, right_robot_start_pose, "right", i, -1, 0)
            self.left_start_states.append(
                [
                    left_robot_start_pose.p.x,
                    left_robot_start_pose.p.y,
                    left_robot_start_pose.p.z,
                    left_robot_start_pose.r.x,
                    left_robot_start_pose.r.y,
                    left_robot_start_pose.r.z,
                    left_robot_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            self.right_start_states.append(
                [
                    right_robot_start_pose.p.x,
                    right_robot_start_pose.p.y,
                    right_robot_start_pose.p.z,
                    right_robot_start_pose.r.x,
                    right_robot_start_pose.r.y,
                    right_robot_start_pose.r.z,
                    right_robot_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            self.gym.set_actor_dof_properties(env_ptr, left_robot_actor, left_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, right_robot_actor, right_dof_props)
            self.left_robot_indices.append(self.gym.get_actor_index(env_ptr, left_robot_actor, gymapi.DOMAIN_SIM))
            self.right_robot_indices.append(self.gym.get_actor_index(env_ptr, right_robot_actor, gymapi.DOMAIN_SIM))

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1, 0)
            self.object_init_states.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add tool
            tool_handle = self.gym.create_actor(env_ptr, tool_asset, tool_start_pose, "tool", i, -1, 0)
            self.tool_init_states.append(
                [
                    tool_start_pose.p.x,
                    tool_start_pose.p.y,
                    tool_start_pose.p.z,
                    tool_start_pose.r.x,
                    tool_start_pose.r.y,
                    tool_start_pose.r.z,
                    tool_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            tool_idx = self.gym.get_actor_index(env_ptr, tool_handle, gymapi.DOMAIN_SIM)
            self.tool_indices.append(tool_idx)


            # add table
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, self.table_start_pose, "table", i, -1, 0
            )
            side_panel_actor = self.gym.create_actor(
                env_ptr, side_panel_asset, side_panel_start_pose, "side_panel", i, -1, 0
            )

            # enable DOF force sensors, if needed
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, right_robot_actor)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

            if self.arm_controller == "ik":
                eef_idx = self.gym.find_actor_rigid_body_index(env_ptr, right_robot_actor, self.palm, gymapi.DOMAIN_SIM)
                self.eef_idx.append(eef_idx)

        self.left_start_states = to_torch(self.left_start_states, device=self.device).view(num_envs, 13)
        self.right_start_states = to_torch(self.right_start_states, device=self.device).view(num_envs, 13)
        self.object_init_states = to_torch(self.object_init_states, device=self.device).view(num_envs, 13)
        self.tool_init_states = to_torch(self.tool_init_states, device=self.device).view(num_envs, 13)
        self.left_fingertip_handles = to_torch(self.left_fingertip_handles, dtype=torch.long, device=self.device)
        self.right_fingertip_handles = to_torch(self.right_fingertip_handles, dtype=torch.long, device=self.device)
        self.left_palm_handle = to_torch(self.left_palm_handle, dtype=torch.long, device=self.device)
        self.right_palm_handle = to_torch(self.right_palm_handle, dtype=torch.long, device=self.device)
        self.left_robot_indices = to_torch(self.left_robot_indices, dtype=torch.long, device=self.device)
        self.right_robot_indices = to_torch(self.right_robot_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.tool_indices = to_torch(self.tool_indices, dtype=torch.long, device=self.device)

        if self.arm_controller == "ik":
            self.eef_idx = to_torch(self.eef_idx, dtype=torch.long, device=self.device)

    def _prepare_robot_asset(self, asset_root, asset_file):
        # load arm hand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # hand dof names
        self.hand_dof_names = []
        num_single_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        for i in range(num_single_robot_dofs):
            joint_name = self.gym.get_asset_dof_name(robot_asset, i)
            if joint_name not in self.arm_dof_names:
                self.hand_dof_names.append(joint_name)
        

        palm_handle = self.gym.find_asset_rigid_body_index(robot_asset, self.palm)
        fingertip_handles = [self.gym.find_asset_rigid_body_index(robot_asset, fingertip) for fingertip in self.fingertips]
    
        arm_dof_indices = [self.gym.find_asset_dof_index(robot_asset, name) for name in self.arm_dof_names]
        hand_dof_indices = [self.gym.find_asset_dof_index(robot_asset, name) for name in self.hand_dof_names]
        # print("fingers dof names: ", self.hand_dof_names, 'indices:', hand_dof_indices)
        robot_dof_indices = arm_dof_indices + hand_dof_indices

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(robot_asset, ft_handle, sensor_pose)

        # get eef index
        robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        arm_eef_index = robot_link_dict[self.palm]

        # set and get robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i in range(num_single_robot_dofs):
            if i in arm_dof_indices:
                robot_dof_props["driveMode"][i] = 1
                robot_dof_props["stiffness"][i] = 1000
                robot_dof_props["damping"][i] = 20
                robot_dof_props["friction"][i] = 0.01
                robot_dof_props["armature"][i] = 0.001
            elif i in hand_dof_indices:
                robot_dof_props["driveMode"][i] = 1
                robot_dof_props["stiffness"][i] = 3
                robot_dof_props["damping"][i] = 0.5
                robot_dof_props["friction"][i] = 0.01
                robot_dof_props["armature"][i] = 0.001
        
        robot_dof_lower_limits = [robot_dof_props["lower"][i] for i in range(num_single_robot_dofs)]
        robot_dof_upper_limits = [robot_dof_props["upper"][i] for i in range(num_single_robot_dofs)]
        robot_dof_default_pos = [0.0] * num_single_robot_dofs
        robot_dof_default_vel = [0.0] * num_single_robot_dofs
        
        return robot_asset, robot_dof_props, palm_handle, fingertip_handles, arm_eef_index, \
                arm_dof_indices, hand_dof_indices, robot_dof_indices, \
                robot_dof_lower_limits, robot_dof_upper_limits, robot_dof_default_pos, robot_dof_default_vel

    def _prepare_object_asset(self, asset_root, asset_file):
        # load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0
        object_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return object_asset

    def _prepare_dataset(self):
        with open(self.cfg['dataset']['meta_data_path'], 'r') as f:
            self.sampled_taco_task_data = json.load(f)[0]
        self.init_timestep = self.sampled_taco_task_data['key_steps']['init']
        self.dataset_end = self.sampled_taco_task_data['key_steps']['end']
        # objects
        self.dataset_object_pose = np.array(self.sampled_taco_task_data['object']['T'])
        self.dataset_object_pos = self.dataset_object_pose[:,:3,3]
        self.dataset_object_quat = R.from_matrix(self.dataset_object_pose[:,:3,:3]).as_quat()
        self.dataset_object_pos = to_torch(self.dataset_object_pos, device=self.device, dtype=torch.float)
        self.dataset_object_quat = to_torch(self.dataset_object_quat, device=self.device, dtype=torch.float)
        self.dataset_tool_pose = np.array(self.sampled_taco_task_data['tool']['T'])
        self.dataset_tool_pos = self.dataset_tool_pose[:,:3,3]
        self.dataset_tool_quat = R.from_matrix(self.dataset_tool_pose[:,:3,:3]).as_quat()
        self.dataset_tool_pos = to_torch(self.dataset_tool_pos, device=self.device, dtype=torch.float)  # just put here
        self.dataset_tool_quat = to_torch(self.dataset_tool_quat, device=self.device, dtype=torch.float)
        # hand poses and finger joints
        dataset_left_dof = self.sampled_taco_task_data['left']
        dataset_left_finger_dof = dataset_left_dof['qpos']
        dataset_left_quat = dataset_left_dof['q']
        dataset_left_pos = dataset_left_dof['p']
        dataset_right_dof = self.sampled_taco_task_data['right']
        dataset_right_finger_dof = dataset_right_dof['qpos']
        dataset_right_quat = dataset_right_dof['r']
        dataset_right_pos = dataset_right_dof['q']
        self.both_finger_dof = torch.from_numpy(np.concatenate([
            dataset_left_finger_dof,
            dataset_right_finger_dof,
        ], axis=-1)).to(self.device).float()
        self.target_left_pose = torch.from_numpy(np.concatenate([
            dataset_left_pos,
            dataset_left_quat,
        ], axis=-1)).to(self.device).float()
        self.target_right_pose = torch.from_numpy(np.concatenate([
            dataset_right_pos,
            dataset_right_quat,
        ], axis=-1)).to(self.device).float()

    def _prepare_object_tool_pair(self, asset_root):  
        assert isinstance(self.sampled_taco_task_data, dict), "Please load the dataset first!"
        object_mesh_path = os.path.join(asset_root, 'TACOobjects')
        self._create_urdf(self.sampled_taco_task_data['tool']['id'], self.sampled_taco_task_data['target']['id'], object_mesh_path)
        object_asset = self._prepare_object_asset(object_mesh_path, 'object.urdf')
        tool_asset = self._prepare_object_asset(object_mesh_path, 'tool.urdf')
        return object_asset, tool_asset

    def _prepare_table_asset(self):
        # create table asset
        table_dims = gymapi.Vec3(1, 1.5, 0.3)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options
        )

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z / 2)

        side_panel_dims = gymapi.Vec3(0.06, 1.5, 1.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        side_panel_asset = self.gym.create_box(
            self.sim,
            side_panel_dims.x,
            side_panel_dims.y,
            side_panel_dims.z,
            asset_options,
        )

        side_panel_start_pose = gymapi.Transform()
        side_panel_start_pose.p = gymapi.Vec3(-0.53, 0.0, side_panel_dims.z / 2)

        return table_asset, table_start_pose, side_panel_asset, side_panel_start_pose

    def compute_reward(self):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.current_successes[:],
            self.consecutive_successes[:],
            reward_info,
        ) = compute_task_rewards(
            self.reset_buf,
            self.progress_buf,
            self.successes,
            self.current_successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos, self.tool_pos,
            self.goal_height,
            self.left_palm_center_pos, self.right_palm_center_pos,
            self.left_fingertip_center_pos, self.right_fingertip_center_pos,
            self.dist_reward_scale,
            self.object_init_states, self.tool_init_states,
            self.action_penalty_scale,
            self.success_tolerance,
            self.av_factor,
            self.table_start_pose.p.z*2,
            self.actions,
        )

        self.extras.update(reward_info)
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["consecutive_successes"] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(
                        self.total_successes / self.total_resets
                    )
                )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        # get refreshed objects
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        self.tool_pose = self.root_state_tensor[self.tool_indices, 0:7]
        self.tool_pos = self.root_state_tensor[self.tool_indices, 0:3]
        self.tool_rot = self.root_state_tensor[self.tool_indices, 3:7]
        self.tool_linvel = self.root_state_tensor[self.tool_indices, 7:10]
        self.tool_angvel = self.root_state_tensor[self.tool_indices, 10:13]


        self.left_palm_state = self.rigid_body_states[:, self.left_palm_handle][..., :13]
        self.left_palm_pos = self.left_palm_state[..., :3]
        self.left_palm_rot = self.left_palm_state[..., 3:7]
        self.left_palm_center_pos = self.left_palm_pos + quat_apply(self.left_palm_rot, to_torch(self.palm_offset).repeat(self.num_envs, 1))
        self.right_palm_state = self.rigid_body_states[:, self.right_palm_handle][..., :13]
        self.right_palm_pos = self.right_palm_state[..., :3]
        self.right_palm_rot = self.right_palm_state[..., 3:7]
        self.right_palm_center_pos = self.right_palm_pos + quat_apply(self.right_palm_rot, to_torch(self.palm_offset).repeat(self.num_envs, 1))

        self.left_fingertip_state = self.rigid_body_states[:, self.left_fingertip_handles][..., :13]
        self.left_fingertip_pose = self.left_fingertip_state[..., :7]
        self.left_fingertip_pos = self.left_fingertip_state[..., :3]
        self.left_fingertip_rot = self.left_fingertip_state[..., 3:7]
        self.left_fingertip_center_pos = torch.zeros_like(self.left_fingertip_pos)
        for i in range(len(self.fingertips)):
            if i == 0:
                self.left_fingertip_center_pos[:, i, :] = self.left_fingertip_pos[:, i, :] + quat_apply(self.left_fingertip_rot[:, i, :],to_torch(self.thumb_offset).repeat(self.num_envs, 1),)
            else:
                self.left_fingertip_center_pos[:, i, :] = self.left_fingertip_pos[:, i, :] + quat_apply(self.left_fingertip_rot[:, i, :],to_torch(self.fingertip_offset).repeat(self.num_envs, 1),)

        self.right_fingertip_state = self.rigid_body_states[:, self.right_fingertip_handles][..., :13]
        self.right_fingertip_pose = self.right_fingertip_state[..., :7]
        self.right_fingertip_pos = self.right_fingertip_state[..., :3]
        self.right_fingertip_rot = self.right_fingertip_state[..., 3:7]
        self.right_fingertip_center_pos = torch.zeros_like(self.right_fingertip_pos)
        for i in range(len(self.fingertips)):
            if i == 0:
                self.right_fingertip_center_pos[:, i, :] = self.right_fingertip_pos[:, i, :] + quat_apply(self.right_fingertip_rot[:, i, :],to_torch(self.thumb_offset).repeat(self.num_envs, 1),)
            else:
                self.right_fingertip_center_pos[:, i, :] = self.right_fingertip_pos[:, i, :] + quat_apply(self.right_fingertip_rot[:, i, :],to_torch(self.fingertip_offset).repeat(self.num_envs, 1),)

        if self.obs_type == "full_no_vel":
            self.compute_full_observations()
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()  # useless

    def compute_full_observations(self, no_vel=False):
        # dof state, pos vel 44 * 2 
        cnt = 0
        self.obs_buf[:, cnt : cnt + self.num_robot_dofs] = unscale(
            self.robot_dof_pos,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )
        self.obs_buf[:, self.num_robot_dofs : 2 * self.num_robot_dofs] = self.vel_obs_scale * self.robot_dof_vel

        # fingertip state, 13 * 4 * 2 
        cnt += 2 * self.num_robot_dofs
        num_ft_states = len(self.fingertips) * 13
        self.obs_buf[:, cnt : cnt + num_ft_states] = self.left_fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, cnt + num_ft_states : cnt + 2 * num_ft_states] = self.right_fingertip_state.reshape(self.num_envs, num_ft_states)

        # action observations, 44
        cnt += 2 * num_ft_states
        self.obs_buf[:, cnt : cnt + self.num_actions] = self.actions


        # object state, pose, linvel, angvel. 13 
        # tool state, pose, linvel, angvel. 13
        cnt += self.num_actions
        self.obs_buf[:, cnt : cnt + 7] = self.object_pose
        self.obs_buf[:, cnt + 7 : cnt + 10] = self.object_linvel
        self.obs_buf[:, cnt + 10 : cnt + 13] = self.object_angvel
        self.obs_buf[:, cnt + 13 : cnt + 20] = self.tool_pose
        self.obs_buf[:, cnt + 20 : cnt + 23] = self.tool_linvel
        self.obs_buf[:, cnt + 23 : cnt + 26] = self.tool_angvel

    def calculate_ik(self, target_left_pose, target_right_pose):
        '''
        target_left_pose: (num_envs, 7)
        target_right_pose: (num_envs, 7)     
        '''
        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # set ik target pose
        target_left_pos = target_left_pose[:, :3]
        target_left_rot = target_left_pose[:, 3:7]
        target_right_pos = target_right_pose[:, :3]
        target_right_rot = target_right_pose[:, 3:7]
        
        cur_left_pose = self.rigid_body_states.view(self.num_envs, -1, 13)[:,self.left_eef_index, 0:7]
        cur_right_pose = self.rigid_body_states.view(self.num_envs, -1, 13)[:,self.right_eef_index, 0:7]
        # print('target_left_pos:', target_left_pos[0], 'target_right_pos:', target_right_pos[0])
        # print('target_left_rot:', target_left_rot[0], 'target_right_rot:', target_left_rot[0])
        # print('cur_left_pos:', cur_left_pose[0,:3], 'cur_right_pos:', cur_right_pose[0,:3])
        # print('cur_left_rot:', cur_left_pose[0,3:7], 'cur_right_rot:', cur_right_pose[0,3:7])
        # print(self.rigid_body_states.view(self.num_envs, -1, 13)[0,:,:3])

        left_pos_err = target_left_pos - cur_left_pose[:,:3]
        left_rot_err = orientation_error(target_left_rot,cur_left_pose[:,3:7])
        left_delta_qpos = self._control_ik(torch.cat([left_pos_err, left_rot_err], -1).unsqueeze(-1), self.left_j_eef, self.num_envs)
        right_pos_err = target_right_pos - cur_right_pose[:,:3]
        right_rot_err = orientation_error(target_right_rot,cur_right_pose[:,3:7])
        right_delta_qpos = self._control_ik(torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1), self.right_j_eef, self.num_envs)

        # print(left_pos_err, right_pos_err, left_rot_err, right_rot_err)
        return self.robot_dof_pos[:, self.both_arm_dof_indices] + torch.cat([left_delta_qpos, right_delta_qpos], -1)

    def compute_full_state(self):
        if self.asymmetric_obs:
            # dof state: pos, vel, force. 3 * 29 = 87
            self.states_buf[:, 0 : self.num_robot_dofs] = unscale(
                self.robot_dof_pos,
                self.robot_dof_lower_limits,
                self.robot_dof_upper_limits,
            )
            self.states_buf[:, self.num_robot_dofs : 2 * self.num_robot_dofs] = (
                self.vel_obs_scale * self.robot_dof_vel
            )
            self.states_buf[:, 2 * self.num_robot_dofs : 3 * self.num_robot_dofs] = (
                self.force_torque_obs_scale * self.dof_force_tensor
            )

            # object state: pos, rot, linvel, angvel. 13 (87+13=100)
            obj_obs_start = 3 * self.num_robot_dofs
            self.states_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = (
                self.object_linvel
            )
            self.states_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = (
                self.object_angvel
            )

            # fingertip observations, state(pose and vel) + force-torque sensors 13 * 5 + 6 * 5 = 95 (100+95=195)
            num_ft_states = len(self.fingertips) * 13
            num_ft_force_torques = len(self.fingertips) * 6

            ft_obs_start = obj_obs_start + 13
            self.states_buf[:, ft_obs_start : ft_obs_start + num_ft_states] = (
                self.left_fingertip_state.reshape(self.num_envs, num_ft_states)
            )
            self.states_buf[
                :,
                ft_obs_start
                + num_ft_states : ft_obs_start
                + num_ft_states
                + num_ft_force_torques,
            ] = self.vec_sensor_tensor

            # action observations, 29 (195+29=224)
            obs_end = ft_obs_start + num_ft_states + num_ft_force_torques
            self.states_buf[:, obs_end : obs_end + self.num_actions] = self.actions

            if self.use_contact_feat:
                self.states_buf[:, obs_end + self.num_actions :] = (
                    self.contact_point_pos
                )
        else:
            # dof state: pos, vel, force. 3 * 29 = 87
            self.obs_buf[:, 0 : self.num_robot_dofs] = unscale(
                self.robot_dof_pos,
                self.robot_dof_lower_limits,
                self.robot_dof_upper_limits,
            )
            self.obs_buf[:, self.num_robot_dofs : 2 * self.num_robot_dofs] = (
                self.vel_obs_scale * self.robot_dof_vel
            )
            self.obs_buf[:, 2 * self.num_robot_dofs : 3 * self.num_robot_dofs] = (
                self.force_torque_obs_scale * self.dof_force_tensor
            )
            # object state: pos, rot, linvel, angvel. 13 (87+13=100)
            obj_obs_start = 3 * self.num_robot_dofs
            self.obs_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = (
                self.object_angvel
            )

            # fingertip observations, state(pose and vel) + force-torque sensors 13 * 5 + 6 * 5 = 95 (100+95=195)
            num_ft_states = len(self.fingertips) * 13
            num_ft_force_torques = len(self.fingertips) * 6

            ft_obs_start = obj_obs_start + 13
            self.obs_buf[:, ft_obs_start : ft_obs_start + num_ft_states] = (
                self.left_fingertip_state.reshape(self.num_envs, num_ft_states)
            )
            self.obs_buf[
                :,
                ft_obs_start
                + num_ft_states : ft_obs_start
                + num_ft_states
                + num_ft_force_torques,
            ] = self.vec_sensor_tensor

            # action observations, 29 (195+29=224)
            obs_end = ft_obs_start + num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end : obs_end + self.num_actions] = self.actions

    def reset_idx(self, env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_robot_dofs * 2), device=self.device)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_states[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_states[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_states[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.tool_indices[env_ids]] = self.tool_init_states[env_ids].clone()
        self.root_state_tensor[self.tool_indices[env_ids], 0:2] = self.tool_init_states[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.tool_indices[env_ids], self.up_axis_idx] = self.tool_init_states[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        objects_indices = torch.cat([self.object_indices[env_ids], self.tool_indices[env_ids]])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(objects_indices.to(torch.int32)),
            len(objects_indices),
        )

        # reset robot
        delta_max = self.robot_dof_upper_limits - self.robot_dof_default_pos
        delta_min = self.robot_dof_lower_limits - self.robot_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, : self.num_robot_dofs] + 1.0)
        pos = self.robot_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.robot_dof_pos[env_ids, :] = pos
        self.robot_dof_vel[env_ids, :] = (
            self.robot_dof_default_vel
            + self.reset_dof_vel_noise
            * rand_floats[:, self.num_robot_dofs : 2 * self.num_robot_dofs]
        )
        self.prev_targets[env_ids, : self.num_robot_dofs] = pos
        self.cur_targets[env_ids, : self.num_robot_dofs] = pos

        left_robot_indices = self.left_robot_indices[env_ids]
        right_robot_indices = self.right_robot_indices[env_ids]
        self.root_state_tensor[left_robot_indices] = self.left_start_states[env_ids].clone()
        self.root_state_tensor[right_robot_indices] = self.right_start_states[env_ids].clone()

        both_hand_indices = torch.cat([left_robot_indices, right_robot_indices])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(both_hand_indices.to(torch.int32)),
            len(both_hand_indices),
        )
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
        )
        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.robot_dof_state),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = (
                self.prev_targets[:, self.both_fingers_dof_indices]
                + self.dof_speed_scale
                * self.dt
                * self.actions[:, self.both_fingers_dof_indices]
            )
            self.cur_targets[:, self.both_fingers_dof_indices] = tensor_clamp(
                targets,
                self.robot_dof_lower_limits[self.both_fingers_dof_indices],
                self.robot_dof_upper_limits[self.both_fingers_dof_indices],
            )
        else:
            self.cur_targets[:, self.both_fingers_dof_indices] = scale(
                self.actions[:, self.both_fingers_dof_indices],
                self.robot_dof_lower_limits[self.both_fingers_dof_indices],
                self.robot_dof_upper_limits[self.both_fingers_dof_indices],
            )
            self.cur_targets[:, self.both_fingers_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.both_fingers_dof_indices]
                + (1.0 - self.act_moving_average)
                * self.prev_targets[:, self.both_fingers_dof_indices]
            )
            self.cur_targets[:, self.both_fingers_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.both_fingers_dof_indices],
                self.robot_dof_lower_limits[self.both_fingers_dof_indices],
                self.robot_dof_upper_limits[self.both_fingers_dof_indices],
            )

        if self.arm_controller == "qpos":
            if self.use_relative_control:
                targets = (
                    self.prev_targets[:, self.both_arm_dof_indices]
                    + self.dof_speed_scale
                    * self.dt
                    * self.actions[:, self.both_arm_dof_indices]
                )
                self.cur_targets[:, self.both_arm_dof_indices] = tensor_clamp(
                    targets,
                    self.robot_dof_lower_limits[self.both_arm_dof_indices],
                    self.robot_dof_upper_limits[self.both_arm_dof_indices],
                )
            else:
                self.cur_targets[:, self.both_arm_dof_indices] = scale(
                    self.actions[:, self.both_arm_dof_indices],
                    self.robot_dof_lower_limits[self.both_arm_dof_indices],
                    self.robot_dof_upper_limits[self.both_arm_dof_indices],
                )
                self.cur_targets[:, self.both_arm_dof_indices] = tensor_clamp(
                    self.cur_targets[:, self.both_arm_dof_indices],
                    self.robot_dof_lower_limits[self.both_arm_dof_indices],
                    self.robot_dof_upper_limits[self.both_arm_dof_indices],
                )
        elif self.arm_controller == "ik":  # direct qpos control
            pos_err = (
                self.actions[:, 0:3]
                - self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:3]
            )
            orn_err = orientation_error(
                matrix_to_quaternion(
                    rotation_6d_to_matrix(
                        torch.torch.concat(
                            (self.actions[:, 3:6], self.actions[:, -3:]),
                            dim=-1,
                        )
                    )
                ),
                self.rigid_body_states.view(-1, 13)[self.eef_idx, 3:7],
            )
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            self.cur_targets[:, self.both_arm_dof_indices] = self.robot_dof_pos[:, self.both_arm_dof_indices] + self._control_ik(dpose)

        self.prev_targets[:, self.both_robot_dof_indices] = self.cur_targets[:, self.both_robot_dof_indices]
        both_hand_indices = torch.cat([self.left_robot_indices, self.right_robot_indices])
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(both_hand_indices.to(torch.int32)), len(both_hand_indices))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward()

        if self.viewer and self.debug_vis:
            # draw axes to debug
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            object_state = self.root_state_tensor[self.object_indices, :]
            tool_state = self.root_state_tensor[self.tool_indices, :]
            for i in range(self.num_envs):
                self._add_debug_lines(self.envs[i], object_state[i, :3], object_state[i, 3:7])
                self._add_debug_lines(self.envs[i], tool_state[i, :3], tool_state[i, 3:7])
                self._add_debug_lines(self.envs[i], self.left_palm_center_pos[i], self.left_palm_rot[i])
                self._add_debug_lines(self.envs[i], self.right_palm_center_pos[i], self.right_palm_rot[i])
                for j in range(len(self.fingertips)):
                    self._add_debug_lines(self.envs[i],self.left_fingertip_center_pos[i][j],self.left_fingertip_rot[i][j])
                    self._add_debug_lines(self.envs[i],self.right_fingertip_center_pos[i][j],self.right_fingertip_rot[i][j])

    def _add_debug_lines(self, env, pos, rot, line_len=0.2):
        posx = (
            (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * line_len))
            .cpu()
            .numpy()
        )
        posy = (
            (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * line_len))
            .cpu()
            .numpy()
        )
        posz = (
            (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * line_len))
            .cpu()
            .numpy()
        )

        p0 = pos.cpu().numpy()
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]],
            [0.85, 0.1, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]],
            [0.1, 0.85, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]],
            [0.1, 0.1, 0.85],
        )

    def _control_ik(self, dpose):
        damping = 0.1
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping**2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        return u




@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


@torch.jit.script
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    out = standardize_quaternion(out)
    return torch.cat([out[..., 3:4], out[..., 0:3]], dim=-1)

@torch.jit.script
def compute_task_rewards(
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos, tool_pos,
    goal_height: float,
    left_palm_pos, right_palm_pos,
    left_fingertip_pos, right_fingertip_pos,
    dist_reward_scale: float,
    object_init_states, tool_init_states,
    action_penalty_scale: float,
    success_tolerance: float,
    av_factor: float,
    table_height: float,
    actions
):
    info = {}
    goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
    gool_tool_dist = torch.abs(goal_height - tool_pos[:, 2])
    left_palm_object_dist = torch.norm(object_pos - left_palm_pos, dim=-1)
    left_palm_object_dist = torch.where(left_palm_object_dist >= 0.5, 0.5, left_palm_object_dist)
    right_palm_object_dist = torch.norm(tool_pos - right_palm_pos, dim=-1)
    right_palm_object_dist = torch.where(right_palm_object_dist >= 0.5, 0.5, right_palm_object_dist)
    object_offset = torch.norm(object_pos[:, 0:2] - object_init_states[:, 0:2], dim=-1)
    tool_offset = torch.norm(tool_pos[:, 0:2] - tool_init_states[:, 0:2], dim=-1)

    num_fingers = left_fingertip_pos.shape[1]
    left_fingertips_object_dist = torch.zeros_like(goal_object_dist)
    for i in range(num_fingers):
        left_fingertips_object_dist += torch.norm(
            left_fingertip_pos[:, i, :] - object_pos, dim=-1
        )
    left_fingertips_object_dist = torch.where(
        left_fingertips_object_dist >= 3.0, 3.0, left_fingertips_object_dist
    )

    right_fingers_tool_dist = torch.zeros_like(gool_tool_dist)
    for i in range(num_fingers):
        right_fingers_tool_dist += torch.norm(
            right_fingertip_pos[:, i, :] - tool_pos, dim=-1
        )
    right_fingers_tool_dist = torch.where(
        right_fingers_tool_dist >= 3.0, 3.0, right_fingers_tool_dist
    )

    is_grasp_left = ((left_fingertips_object_dist <= 0.12 * num_fingers) + (left_palm_object_dist <= 0.12)).float()
    is_grasp_right = ((right_fingers_tool_dist <= 0.12 * num_fingers) + (right_palm_object_dist <= 0.12)).float()

    # stage 1: after hand approach object, lift_object
    lift_object_rew = torch.zeros_like(goal_object_dist)
    lift_object_rew = torch.where(
        is_grasp_left == True, 2 * (goal_height - table_height - goal_object_dist), lift_object_rew
    )
    lift_tool_rew = torch.zeros_like(gool_tool_dist)
    lift_tool_rew = torch.where(
        is_grasp_right == True, 2 * (goal_height - table_height - gool_tool_dist), lift_tool_rew
    )
    # stage 2: lift up reward
    left_hand_up_rew = torch.zeros_like(goal_object_dist)
    left_hand_up_rew = torch.where(is_grasp_left == True, 1 * (left_palm_pos[:, 2] - goal_height), left_hand_up_rew)
    right_hand_up_rew = torch.zeros_like(gool_tool_dist)
    right_hand_up_rew = torch.where(is_grasp_right == True, 1 * (right_palm_pos[:, 2] - goal_height), right_hand_up_rew)

    # stage 3: lift near goal bonus
    left_bonus = torch.zeros_like(goal_object_dist)
    left_bonus = torch.where(
        is_grasp_left == True,
        torch.where(
            goal_object_dist <= success_tolerance, 1.0 / (0.5 + goal_object_dist), left_bonus
        ),
        left_bonus,
    )
    right_bonus = torch.zeros_like(gool_tool_dist)
    right_bonus = torch.where(
        is_grasp_right == True,
        torch.where(
            gool_tool_dist <= success_tolerance, 1.0 / (0.5 + gool_tool_dist), right_bonus
        ),
        right_bonus,
    )
    # action penalty
    left_action, right_action = actions.split(actions.shape[1]//2,-1)
    left_action_penalty = (left_action.abs() - 0.8).clip(min=0).sum(-1) * action_penalty_scale
    right_action_penalty = (right_action.abs() - 0.8).clip(min=0).sum(-1) * action_penalty_scale


    left_approach_penalty = dist_reward_scale * left_fingertips_object_dist + 2 * dist_reward_scale * left_palm_object_dist
    right_approach_penalty = dist_reward_scale * right_fingers_tool_dist + 2 * dist_reward_scale * right_palm_object_dist
    left_after_grasp_reward = lift_object_rew + left_hand_up_rew + left_bonus
    right_after_grasp_reward = lift_tool_rew + right_hand_up_rew + right_bonus
    object_offset_penalty = torch.where(object_offset < 0.2, object_offset**2, object_offset*0.25)
    tool_offset_penalty = torch.where(tool_offset < 0.2, tool_offset**2, tool_offset*0.25)
    
    # total reward
    left_reward = - left_approach_penalty + left_after_grasp_reward - object_offset_penalty - left_action_penalty
    right_reward = - right_approach_penalty + right_after_grasp_reward - tool_offset_penalty - right_action_penalty
    reward = left_reward + right_reward

    # level 1
    info["left/fingertips_object_dist"] = left_fingertips_object_dist
    info["left/palm_object_dist"] = left_palm_object_dist
    info["left/lift_object_rew"] = lift_object_rew
    info["left/hand_up_rew"] = left_hand_up_rew
    info["left/bonus"] = left_bonus
    info["left/object_offset"] = object_offset
    info["left/action_penalty"] = left_action_penalty

    info["right/fingers_tool_dist"] = right_fingers_tool_dist
    info["right/palm_object_dist"] = right_palm_object_dist
    info["right/lift_tool_rew"] = lift_tool_rew
    info["right/hand_up_rew"] = right_hand_up_rew
    info["right/bonus"] = right_bonus
    info["right/tool_offset"] = tool_offset
    info["right/action_penalty"] = right_action_penalty
    # level 2
    info["left/left_approach_penalty"] = left_approach_penalty
    info["left/left_after_grasp_reward"] = left_after_grasp_reward
    info["left/object_offset_penalty"] = object_offset_penalty

    info["right/right_approach_penalty"] = right_approach_penalty
    info["right/right_after_grasp_reward"] = right_after_grasp_reward
    info["right/tool_offset_penalty"] = tool_offset_penalty
    # level 3
    info["left/reward"] = left_reward
    info["right/reward"] = right_reward
    info["reward"] = reward

    resets = reset_buf.clone()
    resets = torch.where(
        progress_buf >= max_episode_length, torch.ones_like(resets), resets
    )
    resets = torch.where(torch.logical_or(object_pos[:, 2] <= table_height, tool_pos[:, 2] <= table_height), torch.ones_like(resets), resets)
    left_dones = torch.logical_and(goal_object_dist <= success_tolerance, left_fingertips_object_dist + left_palm_object_dist < 0.12 * (num_fingers + 1))
    right_dones = torch.logical_and(gool_tool_dist <= success_tolerance, right_fingers_tool_dist + right_palm_object_dist < 0.12 * (num_fingers + 1))
    info["left/dones"] = left_dones
    info["right/dones"] = right_dones
    successes = torch.logical_and(left_dones, right_dones).float()
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    current_successes = torch.where(resets, successes, current_successes)
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        cons_successes,
        info,
    )
