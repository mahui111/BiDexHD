import os, random, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from os.path import join, isfile, isdir, dirname, abspath
import argparse
from pathlib import Path
import tempfile
import time
import numpy as np
from scipy.signal import butter, filtfilt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import trimesh
import torch
import imageio
import pickle
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation as R
# from pytransform3d import transformations as pt
# from sapien.asset import create_dome_envmap
# from sapien.utils import Viewer
# import pyrealsense2 as rs
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

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


class TACODataset:
    # constants
    IMAGE_SIZE = (1024, 750)
    INTRINSIC = np.float32([
        [9533.359863759411, 0.0, 2231.699969508665],
        [0.0, 9593.722282299485, 1699.3865932992662],
        [0.0,0.0,1.0]
    ]) / 4.0
    CAMERA_TO_WORLD = np.float32([
        [-9.861640182402281463e-01, -6.260122508879048531e-02, 1.534979340110796397e-01, -2.296398434650619436e-01],
        [-1.649820266894134468e-01, 2.803133276113436434e-01, -9.456243277501432676e-01, 1.459474531833096833e+00],
        [1.616972472681078854e-02, -9.578650870455813759e-01, -2.867629945118802537e-01, 1.091449991261020935e+00],
        [0, 0, 0, 1],
    ])
    EXTRINSIC = np.linalg.inv(CAMERA_TO_WORLD)

    def __init__(self, dataset_dir, mano_model_path, optimize_wrist):
        self.dataset_root = dataset_dir
        self.mano_model_path = mano_model_path
        self.optimize_wrist = optimize_wrist
        self.triplet_list = os.listdir(join(self.dataset_root, "Object_Poses"))
        if optimize_wrist:
            retarget_type = RetargetingType.position
            add_dummy_free_joint = True
        else:
            retarget_type = RetargetingType.dexpilot
            add_dummy_free_joint = False
        self.biretargetor = BiRetargetor(RobotName.leap, retarget_type, add_dummy_free_joint)
        random.seed(0)

    # main
    def make_task(self, triplet="(empty, bowl, bowl)", sequence_name='', is_visualize=False):
        '''
        return:
        object poses: (N, 4, 4) -> (N, 3+4)
        MANO hand keypoints (right): (N, 21, 3)
        '''
        if triplet not in self.triplet_list:
            triplet = random.choice(self.triplet_list)
        if sequence_name == '':
            sequence_name = random.choice(os.listdir(join(self.dataset_root, "Object_Poses", triplet)))
            
        object_pose_dir = join(self.dataset_root, "Object_Poses", triplet, sequence_name)
        hand_pose_dir = join(self.dataset_root, "Hand_Poses", triplet, sequence_name)
        for file_name in sorted(os.listdir(object_pose_dir)):
            if file_name.startswith("tool_"):
                tool_name = file_name.split(".")[0].split("_")[-1]
            elif file_name.startswith("target_"):
                target_name = file_name.split(".")[0].split("_")[-1]
        print(f"tool: {tool_name}, target: {target_name}")
        
        # joint pos (N,21,3) -> joint qpos (N,6+16)
        # all_left_trans[0]==all_left_joint_pos[0,0]
        left_hand_vertices, all_left_joint_pos, all_left_theta, all_left_trans = self.mano_params_to_hand_info(join(hand_pose_dir, "left_hand.pkl"), mano_beta=pickle.load(open(join(hand_pose_dir, "left_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy(), side="left", max_cnt=None, return_pose=True, return_faces=False,)
        right_hand_vertices, all_right_joint_pos, all_right_theta, all_right_trans = self.mano_params_to_hand_info(join(hand_pose_dir, "right_hand.pkl"), mano_beta=pickle.load(open(join(hand_pose_dir, "right_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy(), side="right", max_cnt=None, return_pose=True, return_faces=False,)
        
        # visualize vertices and trans
        if is_visualize:
            visualizer = Visualizer3D()
            visualizer.visualize_point_clouds(
                np.concatenate([left_hand_vertices[0],all_left_trans[0].reshape(-1,3)]), 
                colors=np.concatenate([len(left_hand_vertices[0])*[[0, 0, 1]],[[1, 0.,0.]]])
            )
            target_pose = np.eye(4)
            # target_pose[:3,:3] = R.from_rotvec(all_left_theta[0,:3]).as_matrix()
            target_pose[:3,3] = all_left_trans[0]
            visualizer.visualize_poses(target_pose[None,:])
            visualizer.draw(True)

            
            visualizer.visualize_point_clouds(
                np.concatenate([right_hand_vertices[0],all_right_trans[0].reshape(-1,3)]), 
                colors=np.concatenate([len(right_hand_vertices[0])*[[0, 0, 1]],[[1, 0.,0.]]])
            )
            target_pose = np.eye(4)
            # target_pose[:3,:3] = R.from_rotvec(all_right_theta[0,:3]).as_matrix()
            target_pose[:3,3] = all_right_trans[0]
            visualizer.visualize_poses(target_pose[None,:])
            visualizer.draw(True)

        # get qpos from dataset
        N = len(all_left_joint_pos)
        left_hand_dof = self.biretargetor.left_hand_dof
        all_left_qpos = np.empty((N, 3+9+left_hand_dof))
        right_hand_dof = self.biretargetor.right_hand_dof
        all_right_qpos = np.empty((N, 3+9+right_hand_dof))
        for i, (left_joint_pos, right_joint_pos) in enumerate(zip(all_left_joint_pos, all_right_joint_pos)):
            all_left_qpos[i,:3], all_right_qpos[i,:3] = all_left_trans[i], all_right_trans[i]
            all_left_qpos[i,3:3+9] = R.from_rotvec(all_left_theta[i,0:3]).as_matrix().reshape(-1)
            all_right_qpos[i,3:3+9] = R.from_rotvec(all_right_theta[i,0:3]).as_matrix().reshape(-1)
            left_qpos, right_qpos = self.biretargetor.retarget_to_robot_poses(left_joint_pos, right_joint_pos)
            all_left_qpos[i,-left_hand_dof:] = left_qpos
            all_right_qpos[i,-right_hand_dof:] = right_qpos
        
        # get object trajectory
        load_tool_poses = np.load(join(object_pose_dir, "tool_" + tool_name + ".npy"))
        load_target_poses = np.load(join(object_pose_dir, "target_" + target_name + ".npy"))
        # for target_pose, left_trans in zip(load_target_poses, all_left_trans):
        #     target_pos = target_pose[:3,3]
        #     print(f'object pos:{target_pos}, left hand pos:{left_trans}, distance:{np.linalg.norm(target_pos-left_trans)}')
        # for tool_pose, right_trans in zip(load_tool_poses, all_right_trans):
        #     tool_pos = tool_pose[:3,3]
        #     print(f'tool pos:{tool_pos}, right hand pos:{right_trans}, distance:{np.linalg.norm(tool_pos-right_trans)}')

        # To label key timesteps
        init_timestep, grasp_timestep, end_timestep = 1, 53, len(load_tool_poses)-1
        tool_dict = dict(id=tool_name, )#xyz=tool_xyz[init_timestep], rpy=tool_rpy[init_timestep]
        target_dict = dict(id=target_name, )#xyz=target_xyz[init_timestep], rpy=target_rpy[init_timestep]
        self.create_urdf(tool_dict, target_dict)  # overwrite a new urdf

        if is_visualize:
            # load object models and poses
            object_model_root = join(self.dataset_root, 'object_models/object_models_released')
            tool_model = trimesh.load(join(object_model_root, tool_name + "_cm.obj"))           # (vertices, faces), unit: cm
            tool_model.vertices *= 0.01  # unit: m      
            target_model = trimesh.load(join(object_model_root, target_name + "_cm.obj"))       # (vertices, faces), unit: cm
            target_model.vertices *= 0.01  # unit: m
            # self.visualize_open3d(tool_model, target_model, load_tool_poses, load_target_poses, right_hand_vertices, left_hand_vertices, save_path="./visualization.mp4", sampling_rate=1)
        
        # return all data
        total_data = dict(
            name=f'{triplet}-{sequence_name}',
            key_steps=dict(init=init_timestep, grasp=grasp_timestep, end=end_timestep),
            tool=dict(T=load_tool_poses.tolist()), #xyz=tool_xyz.tolist(), rpy=tool_rpy.tolist()
            object=dict(T=load_target_poses.tolist()),#xyz=target_xyz.tolist(), rpy=target_rpy.tolist()
            left=all_left_qpos.tolist(),
            right=all_right_qpos.tolist(),
        )

        with open("sampled_taco_task_data.json", "w") as f:
            json.dump(total_data, f, indent=4)
            print("Data saved to sampled_taco_task_data.json")
        return total_data

    def make_dataset(self, triplet="(empty, bowl, bowl)", save_dir="taco_dataset/sampled_data", vis_ref=False):
        total_dataset = []
        seqname_list = sorted(os.listdir(join(self.dataset_root, "Object_Poses", triplet)))
        for k, sequence_name in tqdm(enumerate(seqname_list), total=len(seqname_list)):
            object_pose_dir = join(self.dataset_root, "Object_Poses", triplet, sequence_name)
            hand_pose_dir = join(self.dataset_root, "Hand_Poses", triplet, sequence_name)
            for file_name in os.listdir(object_pose_dir):
                if file_name.startswith("tool_"):
                    tool_name = file_name.split(".")[0].split("_")[-1]
                elif file_name.startswith("target_"):
                    target_name = file_name.split(".")[0].split("_")[-1]
            print(f"tool: {tool_name}, target: {target_name}")
            
            # joint pos (N,21,3) -> joint qpos (N,6+16)
            # all_left_trans[0]==all_left_joint_pos[0,0]
            left_hand_vertices, all_left_joint_pos, all_left_theta, all_left_trans = self.mano_params_to_hand_info(join(hand_pose_dir, "left_hand.pkl"), mano_beta=pickle.load(open(join(hand_pose_dir, "left_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy(), side="left", max_cnt=None, return_pose=True, return_faces=False,)
            right_hand_vertices, all_right_joint_pos, all_right_theta, all_right_trans = self.mano_params_to_hand_info(join(hand_pose_dir, "right_hand.pkl"), mano_beta=pickle.load(open(join(hand_pose_dir, "right_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy(), side="right", max_cnt=None, return_pose=True, return_faces=False,)
            
            all_left_quat = R.from_rotvec(all_left_theta[:,0:3]).as_quat()
            all_right_quat = R.from_rotvec(all_right_theta[:,0:3]).as_quat()

            # get qpos from TACO dataset
            N = len(all_left_joint_pos)
            left_hand_dof = self.biretargetor.left_hand_dof
            all_left_finger_qpos = np.empty((N, left_hand_dof))
            right_hand_dof = self.biretargetor.right_hand_dof
            all_right_finger_qpos = np.empty((N, right_hand_dof))
            for i, (left_joint_pos, right_joint_pos) in enumerate(zip(all_left_joint_pos, all_right_joint_pos)):
                if self.optimize_wrist:
                    left_palm_pose, right_palm_pose, left_fingers_qpos, right_fingers_qpos = self.biretargetor.retarget_to_armrobot_poses(left_joint_pos, right_joint_pos)
                    # print(f"left_palm_pose_diff: {np.linalg.norm(left_palm_pose[:3] - all_left_trans[i])}")
                    # print(f"right_palm_pose_diff: {np.linalg.norm(right_palm_pose[:3] - all_right_trans[i])}")
                    # print(f"left_fingers_qpos_diff: {np.linalg.norm(left_palm_pose[3:] - all_left_quat[i])}")
                    # print(f"right_fingers_qpos_diff: {np.linalg.norm(right_palm_pose[3:] - all_right_quat[i])}")
                    # input()
                    # continue
                    all_left_finger_qpos[i] = left_fingers_qpos
                    all_right_finger_qpos[i] = right_fingers_qpos
                    # overwrite the palm pose with the co-optimized one
                    all_left_trans[i] = left_palm_pose[:3]
                    all_right_trans[i] = right_palm_pose[:3]
                    all_left_quat[i] = left_palm_pose[3:]
                    all_right_quat[i] = right_palm_pose[3:]
                else:
                    left_qpos, right_qpos = self.biretargetor.retarget_to_robot_poses(left_joint_pos, right_joint_pos)
                    all_left_finger_qpos[i] = left_qpos
                    all_right_finger_qpos[i] = right_qpos

            # get object trajectory
            load_tool_poses = np.load(join(object_pose_dir, "tool_" + tool_name + ".npy"))
            load_target_poses = np.load(join(object_pose_dir, "target_" + target_name + ".npy"))
        
            # TODO: smooth trajectory
            def low_pass_filter(data, cutoff=2.0, fs=20, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                y = filtfilt(b, a, data, axis=0)
                return y

            def visualize_ax(ax, trajectory, label, color, highlight=-1):
                ax.plot(*trajectory.T, label=label, color=color)
                ax.scatter(*trajectory[:5].T, color='green', s=50)
                ax.scatter(*trajectory[-5:].T, color='black', s=50)
                if highlight >=0 and highlight < len(trajectory):
                    ax.scatter(*trajectory[highlight:highlight+2], color='cyan', s=50)
                ax.set_title(label)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()


            def visualize_smoothed_trajectory(trajectory, smoothed_trajectory):
                # Plot the original and smoothed trajectories in 3D
                fig = plt.figure(figsize=(14, 7))

                ax1 = fig.add_subplot(121, projection='3d')
                visualize_ax(ax1, trajectory, 'Original', 'b')
                ax2 = fig.add_subplot(122, projection='3d')
                visualize_ax(ax2, smoothed_trajectory, 'Smoothed', 'r')
                plt.show()
            
            smoothed_object_pos = low_pass_filter(load_target_poses[:, :3, 3])
            smoothed_tool_pos = low_pass_filter(load_tool_poses[:, :3, 3])
            smoothed_left_pos = low_pass_filter(all_left_trans)
            smoothed_right_pos = low_pass_filter(all_right_trans)
            # visualize_smoothed_trajectory(load_target_poses[:, :3, 3], smoothed_object_pos)
            # visualize_smoothed_trajectory(load_tool_poses[:, :3, 3], smoothed_tool_pos)
            # visualize_smoothed_trajectory(all_left_trans, smoothed_left_pos)
            # visualize_smoothed_trajectory(all_right_trans, smoothed_right_pos)

            '''[important!]: smooth trajectory'''
            if not self.optimize_wrist:
                load_target_poses[:, :3, 3] = smoothed_object_pos
                load_tool_poses[:, :3, 3] = smoothed_tool_pos
                all_left_trans = smoothed_left_pos
                all_right_trans = smoothed_right_pos

            # get key timesteps
            init_timestep = 1  # int(len(load_tool_poses)*0.1)
            end_timestep = int(len(load_tool_poses)*0.8)   

            object_heights = smoothed_object_pos[:, 2]
            tool_heights = smoothed_tool_pos[:, 2]
            percentage = 75
            ref_object_height = np.percentile(object_heights[object_heights>object_heights[init_timestep]], percentage)
            ref_tool_height = np.percentile(object_heights[tool_heights>tool_heights[init_timestep]], percentage)
            # find the first False
            try:
                ref_object_timestep = np.where((object_heights<ref_object_height)==False)[0][0]
                ref_tool_timestep = np.where((tool_heights<ref_tool_height)==False)[0][0]
                ref_timestep = int(max(ref_object_timestep, ref_tool_timestep))
            except:
                ref_timestep = 30
            print(f"task: {k} | init_timestep: {init_timestep} | ref_timestep: {ref_timestep} | end_timestep: {end_timestep}")
            # visualize the height curve of the object and tool
            if vis_ref:
                fig = plt.figure(figsize=(14, 7))
                ax1 = fig.add_subplot(121)
                ax1.plot(np.arange(len(object_heights)), object_heights, label='Object')
                ax1.scatter(ref_timestep, object_heights[ref_timestep], color='red', s=50)
                ax2 = fig.add_subplot(122)
                ax2.plot(np.arange(len(tool_heights)), tool_heights, label='Tool')
                ax2.scatter(ref_timestep, tool_heights[ref_timestep], color='red', s=50)
                plt.show()

            # return all data
            total_data = dict(
                save_name=os.path.join(save_dir, f'{triplet}-{sequence_name}.json'),
                key_steps=dict(init=init_timestep, ref=ref_timestep, end=end_timestep),
                tool=dict(id=tool_name, T=load_tool_poses.tolist()),
                object=dict(id=target_name, T=load_target_poses.tolist()),
                left=dict(p=all_left_trans.tolist(), q=all_left_quat.tolist(), qpos=all_left_finger_qpos.tolist()),
                right=dict(p=all_right_trans.tolist(), q=all_right_quat.tolist(), qpos=all_right_finger_qpos.tolist()),
            )
            total_dataset.append(total_data)

        with open(os.path.join(save_dir,f"{triplet}.json"), "w") as f:
            json.dump(total_dataset, f, indent=4)
        return total_dataset

    def mano_params_to_hand_info(self, hand_pose_path, mano_beta=None, side="right", max_cnt=None, return_pose=False, return_faces=False, device="cuda:0"):
        """
        hand_pose_path: hand_00001.pkl to hand_{xxxxx}.pkl
        mano_beta: (10,)
        side: "left" / "right"

        return: a tuple:
            * hand_veratices:  numpy array, shape = (N_frame, 778, 3)
            * hand_joints: a numpy array, shape = (N_frame, 21, 3)
            * (optional) hand_faces: a numpy array, shape = (N_face, 3)
        """
        from manopth.manopth.manolayer import ManoLayer
        betas = torch.from_numpy(mano_beta).unsqueeze(0).to(torch.float32).to(device)  # (1, 10)
        mano_layer = ManoLayer(mano_root=self.mano_model_path, use_pca=False, ncomps=45, side=side, center_idx=0)
        mano_layer.to(device)

        theta_list, trans_list = [], []
        
        if hand_pose_path.endswith(".pkl"):
            hand_pose_data = pickle.load(open(hand_pose_path, "rb"))
            keys = list(hand_pose_data.keys())
            keys.sort()
            for key in keys:
                theta_list.append(hand_pose_data[key]["hand_pose"].detach().cpu().numpy())
                trans_list.append(hand_pose_data[key]["hand_trans"].detach().cpu().numpy())
        else:
            hand_pose_fns = os.listdir(hand_pose_path)
            hand_pose_fns.sort()
            if not max_cnt is None:
                hand_pose_fns = hand_pose_fns[:max_cnt]

            for hand_pose_fn in hand_pose_fns:
                hand_pose_data = pickle.load(open(join(hand_pose_path, hand_pose_fn), "rb"))
                theta_list.append(hand_pose_data["hand_pose"].detach().cpu().numpy())  # (48,)
                trans_list.append(hand_pose_data["hand_trans"].detach().cpu().numpy())  # (3,)
                
        batch_theta = torch.from_numpy(np.float32(theta_list)).to(device)  # (N_frame, 48)
        batch_trans = torch.from_numpy(np.float32(trans_list)).to(device)  # (N_frame, 3)
        
        # batch_theta, batch_trans = torch.zeros_like(batch_theta), torch.zeros_like(batch_trans)
        # batch_theta[:,1:2] = 1.57

        betas = betas.repeat(batch_theta.shape[0], 1)  # (N_frame, 10)
        '''
        MANO: (N, 48), (N, 10) -> (N, 778, 3), (N, 1538, 3), (N, 21, 3)
        '''    
        hand_verts_pred, hand_joints_pred, full_pose = mano_layer(batch_theta, betas)  # full_pose==batch_theta
        hand_verts_pred = hand_verts_pred / 1000.0
        hand_joints_pred = hand_joints_pred / 1000.0
        hand_verts_pred += batch_trans.unsqueeze(1)  # (N_frame, 778, 3)
        hand_joints_pred += batch_trans.unsqueeze(1)  # (N_frame, 21, 3)
        
        hand_vertices = hand_verts_pred.detach().cpu().numpy()
        hand_joints = hand_joints_pred.detach().cpu().numpy()
        if not return_pose:
            if return_faces:  # return vertices, joints, faces
                return hand_vertices, hand_joints, mano_layer.th_faces.detach().cpu().numpy().reshape(-1, 3)  
            # TODO: camera_mat = extrinsic_mat.inv()
            # hand_joints = np.ascontiguousarray(hand_joints @ camera_mat[:3, :3].T + camera_mat[:3, 3])
            return hand_vertices, hand_joints
        else:
            if return_faces:
                return hand_vertices, mano_layer.th_faces.detach().cpu().numpy().reshape(-1, 3), hand_joints, batch_theta.detach().cpu().numpy(), batch_trans.detach().cpu().numpy()
            return hand_vertices, hand_joints, batch_theta.detach().cpu().numpy(), batch_trans.detach().cpu().numpy()

    def create_urdf(self, tool_dict, target_dict, save_path="assets/meshobjects"):
        with open(join(save_path, "tool.urdf"), 'w') as urdf_file:
            urdf_file.write(self._generate_urdf(tool_dict))
        
        with open(join(save_path, "target.urdf"), 'w') as urdf_file:
            urdf_file.write(self._generate_urdf(target_dict))

    def visualize_open3d(self, tool_model, target_model, tool_poses, target_poses, right_hand_meshes, left_hand_meshes, save_path=None, sampling_rate=1, device="cuda:0"):
        N = tool_poses.shape[0]
        assert target_poses.shape[0] == N

        tool_pts = tool_model.vertices
        tool_faces = tool_model.faces
        target_pts = target_model.vertices
        target_faces = target_model.faces

        print("###### start visualization ... ######")

        assert save_path is not None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = imageio.get_writer(save_path, fps=10)
        for frame_idx in tqdm(range(0, N, sampling_rate)):
            # get object poses in this frame
            tool_pose = tool_poses[frame_idx]
            target_pose = target_poses[frame_idx]

            # construct object meshes in this frame
            tool_vertices = (tool_pts.copy() @ tool_pose[:3, :3].T) + tool_pose[:3, 3]
            tool_mesh = trimesh.Trimesh(vertices=tool_vertices, faces=tool_faces)

            target_vertices = (target_pts.copy() @ target_pose[:3, :3].T) + target_pose[:3, 3]
            target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces)

            right_hand_mesh = right_hand_meshes[frame_idx]
            left_hand_mesh = left_hand_meshes[frame_idx]

            # Convert trimesh meshes to Open3D meshes
            tool_mesh_o3d = self._trimesh_to_open3d(tool_mesh)
            target_mesh_o3d = self._trimesh_to_open3d(target_mesh)
            right_hand_mesh_o3d = self._trimesh_to_open3d(right_hand_mesh)
            left_hand_mesh_o3d = self._trimesh_to_open3d(left_hand_mesh)
            meshes = [right_hand_mesh_o3d, left_hand_mesh_o3d, tool_mesh_o3d, target_mesh_o3d]

            # Visualize the meshes
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=self.IMAGE_SIZE[0], height=self.IMAGE_SIZE[1])
            for mesh in meshes:
                vis.add_geometry(mesh)

            # Set the camera parameters
            fx, fy, cx, cy = self.INTRINSIC[0, 0], self.INTRINSIC[1, 1], self.INTRINSIC[0, 2], self.INTRINSIC[1, 2]
            intrinsics = o3d.camera.PinholeCameraIntrinsic(self.IMAGE_SIZE[0],self.IMAGE_SIZE[1], fx, fy, cx, cy)
            camera_parameters = o3d.camera.PinholeCameraParameters()
            camera_parameters.intrinsic = intrinsics
            # ctr = vis.get_view_control()
            # success = ctr.convert_from_pinhole_camera_parameters(camera_parameters)

            # Capture the screen
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            img = (np.asarray(image) * 255).astype(np.uint8)
            vis.destroy_window()
            writer.append_data(img)

        writer.close()

    def _trimesh_to_open3d(self, trimesh_mesh):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
        o3d_mesh.compute_vertex_normals()
        return o3d_mesh

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

    def visualize_robot_and_mano(self, triplet, side):
        sequence_name = os.listdir(join(self.dataset_root, "Egocentric_RGB_Videos", triplet))[-1]
        print(f"Triplet: {triplet} | Sequence: {sequence_name}")

        hand_pose_dir = join(self.dataset_root, "Hand_Poses", triplet, sequence_name)
        left_hand_vertices, left_hand_faces, all_left_joint_pos, all_left_theta, all_left_trans = self.mano_params_to_hand_info(
            join(hand_pose_dir, "left_hand.pkl"), 
            mano_beta=pickle.load(open(join(hand_pose_dir, "left_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy(), 
            side="left", 
            max_cnt=None, return_pose=True, return_faces=True,
        )
        right_hand_vertices, right_hand_faces, all_right_joint_pos, all_right_theta, all_right_trans = self.mano_params_to_hand_info(
            join(hand_pose_dir, "right_hand.pkl"), 
            mano_beta=pickle.load(open(join(hand_pose_dir, "right_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy(), 
            side="right", 
            max_cnt=None, return_pose=True, return_faces=True,
        )
        
        N = len(all_left_joint_pos)
        left_hand_dof = self.biretargetor.left_hand_dof
        all_left_qpos = np.empty((N, 3+4+left_hand_dof))
        right_hand_dof = self.biretargetor.right_hand_dof
        all_right_qpos = np.empty((N, 3+4+right_hand_dof))
        # rotate -90 along y axis
        rot_left_comp = R.from_euler('xyz', [0,0,0])
        rot_right_comp = R.from_euler('xyz', [0, 0, 0])
        for i, (left_joint_pos, right_joint_pos) in enumerate(zip(all_left_joint_pos, all_right_joint_pos)):
            all_left_qpos[i,:3], all_right_qpos[i,:3] = all_left_trans[i], all_right_trans[i]
            all_left_qpos[i,3:3+4] = (R.from_rotvec(all_left_theta[i,0:3]) * rot_left_comp).as_quat()
            all_right_qpos[i,3:3+4] = (R.from_rotvec(all_right_theta[i,0:3]) * rot_right_comp).as_quat()
            left_qpos, right_qpos = self.biretargetor.retarget_to_robot_poses(left_joint_pos, right_joint_pos)
            all_left_qpos[i,-left_hand_dof:] = left_qpos
            all_right_qpos[i,-right_hand_dof:] = right_qpos
        



        import sapien.core as sapien
        from sapien.utils import Viewer
        from typing import List
        # Scene
        scene = sapien.Scene()

        # Lighting
        scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
        scene.add_directional_light(np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True)
        scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
        scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

        # Add ground
        visual_material = sapien.render.RenderMaterial()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        scene.add_ground(-1, render_material=visual_material)

        # Viewer
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.set_camera_xyz(-0.6, 0., 0.6)
        # viewer.set_camera_rpy(0, -0.8, 3.14)
        viewer.control_window.toggle_origin_frame(False)
        sapien.render.set_log_level("error")

        # for update mano
        self.nodes: List[R.Node] = []
        self.internal_scene: R.Scene = scene.render_system._internal_scene
        self.context: R.Context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0)

        def clear_node():
            for _ in range(len(self.nodes)):
                node = self.nodes.pop()
                self.internal_scene.remove_node(node)
        def compute_smooth_shading_normal_np(vertices, indices):
            """
            Compute the vertex normal from vertices and triangles with numpy
            Args:
                vertices: (n, 3) to represent vertices position
                indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
            Returns:
                (n, 3) vertex normal

            References:
                https://www.iquilezles.org/www/articles/normals/normals.htm
            """
            v1 = vertices[indices[:, 0]]
            v2 = vertices[indices[:, 1]]
            v3 = vertices[indices[:, 2]]
            face_normal = np.cross(v2 - v1, v3 - v1)  # (n, 3) normal without normalization to 1

            vertex_normal = np.zeros_like(vertices)
            vertex_normal[indices[:, 0]] += face_normal
            vertex_normal[indices[:, 1]] += face_normal
            vertex_normal[indices[:, 2]] += face_normal
            vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
            return vertex_normal
        def _update_hand(vertex, face):
            clear_node()
            # normal = compute_smooth_shading_normal_np(vertex, self.mano_face)
            mesh = self.context.create_mesh_from_array(vertex, face,)
            model = self.context.create_model([mesh], [self.mat_hand])
            node = self.internal_scene.add_node()
            node.set_position(np.array([0, 0, 0]))
            obj = self.internal_scene.add_object(model, node)
            obj.shading_mode = 0
            obj.cast_shadow = True
            obj.transparency = 0
            self.nodes.append(node)



        # Load the robot URDF & qpos & mano vertex
        urdf_path = f'assets/mjcf/leap_hand_description/leap_hand_{side}.urdf'
        robot = scene.create_urdf_loader().load(urdf_path)
        if side == 'left':
            qpos_data = all_left_qpos
            vertices, faces = left_hand_vertices, left_hand_faces
        else:
            qpos_data = all_right_qpos
            vertices, faces = right_hand_vertices, right_hand_faces

        N = qpos_data.shape[0]
        i = 0
        while True:
            # set robot qpos
            qpos = qpos_data[i]
            robot.set_root_pose(sapien.Pose(qpos[:3], [qpos[6]] + list(qpos[3:6])))  # sapien(w,x,y,z)
            robot.set_qpos(qpos[7:])
            # set mano 
            mesh = trimesh.Trimesh(vertices=vertices[i].copy(), faces=faces.copy())  # necessary
            _update_hand(mesh.vertices + np.array([0,0.2,0]), mesh.faces)

            # Update the scene and render
            scene.update_render()
            viewer.render()

            i = (i + 1) % N
            time.sleep(0.1)

        # Close viewer
        viewer.close()
        

class BiRetargetor:
    def __init__(self, robot_name:RobotName, retarget_type:RetargetingType, add_dummy_free_joint:bool=False):
        self.side_convernion = dict(
            left=np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0],]),
            right=np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0],]),
        )
        
        self.retarget_type = retarget_type

        left_config_path = get_default_config_path(robot_name, retarget_type, HandType.left)
        right_config_path = get_default_config_path(robot_name, retarget_type, HandType.right) 

        self.left_retargetor = self.get_retargetor(left_config_path, add_dummy_free_joint=add_dummy_free_joint)
        self.right_retargetor = self.get_retargetor(right_config_path, add_dummy_free_joint=add_dummy_free_joint)

        left_hand_joint_names = ['1', '0', '2', '3', '12', '13', '14', '15', '5', '4', '6', '7', '9', '8', '10', '11']  # self.left_hand.get_active_joints()
        right_hand_joint_names = ['1', '0', '2', '3', '12', '13', '14', '15', '5', '4', '6', '7', '9', '8', '10', '11']  # self.right_hand.get_active_joints()
        self.left_retarget_idxs = np.array([self.left_retargetor.joint_names.index(joint_name) for joint_name in left_hand_joint_names]).astype(int)
        self.right_retarget_idxs = np.array([self.right_retargetor.joint_names.index(joint_name) for joint_name in right_hand_joint_names]).astype(int)  
        self.left_hand_dof = len(self.left_retarget_idxs)
        self.right_hand_dof = len(self.right_retarget_idxs)

    def get_retargetor(self, config_path, add_dummy_free_joint, return_robot=False):
        '''
        return SeqRetargeting
        '''
        config = RetargetingConfig.load_from_file(config_path, override=dict(add_dummy_free_joint=add_dummy_free_joint))  # [Important!] Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
        retargeting = config.build()
        if not return_robot:
            return retargeting
        # Build robot
        urdf_path = Path(config.urdf_path)
        if "glb" not in urdf_path.stem:
            urdf_path = str(urdf_path).replace(".urdf", "_glb.urdf")
        robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=add_dummy_free_joint, build_scene_graph=False)
        urdf_name = urdf_path.split("/")[-1]
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)
        robot = self.loader.load(temp_path)
        return robot, retargeting
    
    # [Important!] Update poses for robot hands
    def retarget_to_robot_poses(self, left_joint_pos, right_joint_pos):
        # normalize to the wrist
        left_joint_pos = left_joint_pos - left_joint_pos[0:1, :]
        right_joint_pos = right_joint_pos - right_joint_pos[0:1, :]
        left_joint_pos = left_joint_pos @ self.estimate_frame_from_hand_points(left_joint_pos, 'left') @ self.side_convernion['left']
        right_joint_pos = right_joint_pos @ self.estimate_frame_from_hand_points(right_joint_pos, 'right') @ self.side_convernion['right']

        left_indices = self.left_retargetor.optimizer.target_link_human_indices
        right_indices = self.right_retargetor.optimizer.target_link_human_indices
        if self.retarget_type == RetargetingType.position:
            left_ref_value = left_joint_pos[left_indices, :]
            right_ref_value = right_joint_pos[right_indices, :]
        else:
            left_ref_value = left_joint_pos[left_indices[1, :], :] - left_joint_pos[left_indices[0, :], :]
            right_ref_value = right_joint_pos[right_indices[1, :], :] - right_joint_pos[right_indices[0, :], :]
        left_qpos = self.left_retargetor.retarget(left_ref_value)
        right_qpos = self.right_retargetor.retarget(right_ref_value)
        assert len(left_qpos) == len(self.left_retarget_idxs)
        return left_qpos[self.left_retarget_idxs], right_qpos[self.right_retarget_idxs]

             
    def retarget_to_armrobot_poses(self, left_joint_pos, right_joint_pos):     
        left_indices = self.left_retargetor.optimizer.target_link_human_indices
        right_indices = self.right_retargetor.optimizer.target_link_human_indices
        if self.retarget_type == RetargetingType.position:
            left_ref_value = left_joint_pos[left_indices, :]
            right_ref_value = right_joint_pos[right_indices, :]
        else:
            raise NotImplementedError
        for _ in range(10):
            left_qpos = self.left_retargetor.retarget(left_ref_value)
            right_qpos = self.right_retargetor.retarget(right_ref_value)
        left_palm_pose = left_qpos[:3].tolist() + R.from_euler('zyx',left_qpos[3:6][::-1]).as_quat().tolist()   
        right_palm_pose = right_qpos[:3].tolist() + R.from_euler('zyx',right_qpos[3:6][::-1]).as_quat().tolist()
        left_fingers_qpos = left_qpos[self.left_retarget_idxs]
        right_fingers_qpos = right_qpos[self.right_retarget_idxs]
        return left_palm_pose, right_palm_pose, left_fingers_qpos, right_fingers_qpos 
             
    def get_joint_keypoints(self, hand_pose_frame, use_camera_frame=False):
        '''
        hand_pose_frame: (N, 21, 3)
        '''
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
        vertex, joint = self.mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]
        if not use_camera_frame:
            camera_mat = self.camera_pose.to_transformation_matrix()
            vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            vertex = np.ascontiguousarray(vertex)
            joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint = np.ascontiguousarray(joint)

        return vertex, joint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray, side) -> np.ndarray:  # useless
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        keypoint_3d_array = keypoint_3d_array -  keypoint_3d_array[0:1, :]  # Normalize the wrist to the origin
        points = keypoint_3d_array[[0, 5, 9], :]  # wrist, forefinger proximal, middle finger proximal

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)
        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x_vector = points[0] - points[2]  # Compute vector from palm to the first joint of middle finger
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)

        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/zbh/Desktop/zbh/robot/TACO-Instructions/dataset/overall")
    parser.add_argument("--mano_model_path", type=str, default="/home/zbh/Desktop/zbh/robot/BVDex/rl_policy/taco_dataset/manopth/mano/models")
    parser.add_argument("--triplet", type=str, default='(smear, eraser, plate)')
    parser.add_argument("--viz_sapien", action="store_true")
    parser.add_argument("--optimize_wrist", type=int, default=1)
    parser.add_argument("--mode", type=str, default="make_dataset")  # make_task / make_dataset
    args = parser.parse_args()
    
    taco_dataset = TACODataset(args.dataset_dir, args.mano_model_path, args.optimize_wrist) 
    if args.viz_sapien:  # useless
        taco_dataset.visualize_robot_and_mano(args.triplet,'right')
    if args.mode == "make_dataset":
        sampled_taco_dataset = taco_dataset.make_dataset(args.triplet)
    elif args.mode == "make_task":
        sampled_taco_task_data = taco_dataset.make_task(args.triplet, is_visualize=False)
    
