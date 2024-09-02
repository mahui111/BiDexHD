import os, json, sys
import random
import torch
import numpy as np
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
import trimesh
from urdfpy import URDF

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

# sys.path.append('../')
# from taco_dataset import Visualizer3D

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
def orientation_error(desired, current):
    '''
    desired: (num_envs, 4)
    current: (num_envs, 4)
    '''
    current = standardize_quaternion(current)
    desired = standardize_quaternion(desired)
    q_r = quat_mul(desired, quat_conjugate(current))
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def quat_diff_theta(desired, current):
    '''
    desired: (num_envs, 4)
    current: (num_envs, 4)

    return (num_envs,)  [-pi, pi]
    '''
    quat_diff = orientation_error(desired, current)
    theta_diff = 2 * torch.asin(torch.clamp(torch.norm(quat_diff, dim=-1), -1, 1))  
    return theta_diff

@torch.jit.script
def quat_rew(desired, current):
    '''
    desired: (num_envs, 4)
    current: (num_envs, 4)

    return (num_envs,)  [-1,1]
    '''
    return quat_diff_theta(desired, current).cos()

@torch.jit.script
def rotmat_dist(desired, current):
    '''
    desired: (num_envs, 3, 3)
    current: (num_envs, 3, 3)
    return: (num_envs,)  [1,0]
    '''
    return torch.acos(
        torch.clamp(
            (torch.diagonal(desired.transpose(-1, -2) @ current, dim1=-2, dim2=-1).sum(-1) - 1) / 2,
            -1,
            1,
        )
    ) / torch.pi

@torch.jit.script
def rotmat_rew(desired, current):
    '''
    desired: (num_envs, 3, 3)
    current: (num_envs, 3, 3)
    return: (num_envs,)  [-1,1]
    '''
    return torch.clamp(
        (torch.diagonal(desired.transpose(-1, -2) @ current, dim1=-2, dim2=-1).sum(-1) - 1) / 2,
        -1,
        1,
    )

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
def rotation_matrix_z(angles):
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    
    R_z = torch.stack([
        torch.stack([cos_theta, -sin_theta, torch.zeros_like(angles)], dim=1),
        torch.stack([sin_theta, cos_theta, torch.zeros_like(angles)], dim=1),
        torch.stack([torch.zeros_like(angles), torch.zeros_like(angles), torch.ones_like(angles)], dim=1)
    ], dim=2)
        
    return R_z

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
def transformation_multiply(
    quat1: torch.Tensor, pos1: torch.Tensor, quat2: torch.Tensor, pos2: torch.Tensor
):
    """Multiply two transformations.

    Args:
        quat1: Quaternion of the first transformation.
        pos1: Position of the first transformation.
        quat2: Quaternion of the second transformation.
        pos2: Position of the second transformation.

    Returns:
        The quaternion and position of the resulting transformation.
    """
    quat1, quat2 = torch.broadcast_tensors(quat1, quat2)
    pos1, pos2 = torch.broadcast_tensors(pos1, pos2)
    return quat_mul(quat1, quat2), quat_apply(quat1, pos2) + pos1

@torch.jit.script
def transformation_inverse(quat: torch.Tensor, pos: torch.Tensor):
    """Invert a transformation.

    Args:
        quat: Quaternion of the transformation.
        pos: Position of the transformation.

    Returns:
        The quaternion and position of the inverted transformation.
    """
    quat = quat_conjugate(quat)
    return quat, -quat_apply(quat, pos)

@torch.jit.script
def transformation_apply(pos: torch.Tensor, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a transformation to a vector.

    Args:
        pos: Position of the transformation.
        quat: Quaternion of the transformation.
        vec: Vector to transform.

    Returns:
        The transformed vector.
    """
    pos, vec = torch.broadcast_tensors(pos, vec)
    quaternion_shape = pos.shape[:-1] + (4,)
    quat = torch.broadcast_to(quat, quaternion_shape)
    return quat_apply(quat, vec) + pos
    
# sample object point cloud & transform within the world coordinate
def farthest_point_sample(xyz, npoint, device, init=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if init is not None:
        farthest = torch.tensor(init).long().reshape(B).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.cuda()

def index_points(points, idx, device):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.size()[0]
    view_shape = list(idx.size())
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.size())
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



@torch.jit.script
def compute_relative_pose(
    a_position: torch.Tensor,a_orientation: torch.Tensor,b_position: torch.Tensor,b_orientation: torch.Tensor,
):
    """Compute a pose in b's frame.

    Args:
        a_position (torch.Tensor): Positions of a, shape (..., 3).
        a_orientation (torch.Tensor): Orientations of a, shape (..., 4).
        b_position (torch.Tensor): Positions of b, shape (..., 3).
        b_orientation (torch.Tensor): Orientations of b, shape (..., 4).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Orientation & Position of a in b's frame.
    """
    assert a_position.dim() == b_position.dim()
    assert a_orientation.dim() == b_orientation.dim()

    w2b_rotation, w2b_translation = transformation_inverse(b_orientation, b_position)

    a_position, w2b_translation = torch.broadcast_tensors(a_position, w2b_translation)
    a_orientation, w2b_rotation = torch.broadcast_tensors(a_orientation, w2b_rotation)

    orientation, position = transformation_multiply(w2b_rotation, w2b_translation, a_orientation, a_position)
    return position, orientation

@torch.jit.script
def compute_relative_position(
    a_position: torch.Tensor,
    b_orientation: torch.Tensor,
    b_position: torch.Tensor,
) -> torch.Tensor:
    """Compute a position in b's frame.

    Args:
        a_position (torch.Tensor): Positions of a, shape (..., 3).
        b_orientation (torch.Tensor): Orientations of b, shape (..., 4).
        b_position (torch.Tensor): Positions of b, shape (..., 3).

    Returns:
        torch.Tensor: Position of a in b's frame.
    """
    assert a_position.dim() == b_position.dim() == b_orientation.dim()

    w2b_rotation, w2b_translation = transformation_inverse(b_orientation, b_position)

    a_position, w2b_translation = torch.broadcast_tensors(a_position, w2b_translation)
    quaternion_shape = a_position.shape[:-1] + (4,)
    w2b_rotation = torch.broadcast_to(w2b_rotation, quaternion_shape)

    position = quat_apply(w2b_rotation, a_position) + w2b_translation
    return position

@torch.jit.script
def compute_bvdex_stage12_rewards(
    reset_buf,
    progress_buf,
    stage1_left_successes, stage1_right_successes, stage1_successes, stage1_cul_left_successes, stage1_cul_right_successes,
    stage2_left_successes, stage2_right_successes, stage2_successes,
    max_episode_length: float,
    object_pose, tool_pose,
    left_palm_pose, right_palm_pose,
    left_fingertip_pose, right_fingertip_pose,
    dist_reward_scale: float,
    action_penalty_scale: float,
    success_tolerance: float,
    av_factor: float,
    table_heights, left_robot_link1_pos, right_robot_link1_pos,
    frequency: float,
    timestep, reach_ref_timestep, left_reach_ref_timestep, right_reach_ref_timestep,
    ref_object_pose, ref_init_object_pos_dist,  #ref_ref_object_palm_pose_diff, ref_ref_object_left_fingers_pos_diff,
    ref_tool_pose, ref_init_tool_pos_dist,      #ref_ref_tool_palm_pose_diff, ref_ref_tool_right_fingers_pos_diff,
    object_grasp_pos, tool_grasp_pos,
    is_expect_end, # is_stage1_hand_object_rew: int, is_stage1_lin_rew:int, is_stage2_pos_rew_exp: int,
):
    '''
    stage 1: reach a static ref object pose (linear reward)
    stage 2: stage 1 finished, follow a dynamic ref object pose (exponential reward)   
    '''
    info = {}

    left_palm_object_dist = torch.norm(object_grasp_pos - left_palm_pose[:, :3], dim=-1)
    left_palm_object_dist = torch.where(left_palm_object_dist >= 0.5, 0.5 * torch.ones_like(left_palm_object_dist), left_palm_object_dist)
    right_palm_tool_dist = torch.norm(tool_grasp_pos - right_palm_pose[:, :3], dim=-1)  
    right_palm_tool_dist = torch.where(right_palm_tool_dist >= 0.5, 0.5 * torch.ones_like(right_palm_tool_dist), right_palm_tool_dist)

    num_fingers = left_fingertip_pose.shape[1]
    left_fingertips_object_dist = torch.zeros_like(left_palm_object_dist)
    for i in range(num_fingers):
        left_fingertips_object_dist += torch.norm(left_fingertip_pose[:, i, :3] - object_grasp_pos, dim=-1)
    left_fingertips_object_dist = torch.where(
        left_fingertips_object_dist >= 3.0, 3.0 * torch.ones_like(left_fingertips_object_dist), left_fingertips_object_dist
    )  # Important!

    right_fingertips_tool_dist = torch.zeros_like(right_palm_tool_dist)
    for i in range(num_fingers):
        right_fingertips_tool_dist += torch.norm(right_fingertip_pose[:, i, :3] - tool_grasp_pos, dim=-1)
    right_fingertips_tool_dist = torch.where(
        right_fingertips_tool_dist >= 3.0, 3.0 * torch.ones_like(right_fingertips_tool_dist), right_fingertips_tool_dist
    )  # Important!
    info["left_palm_object_dist"] = left_palm_object_dist
    info["right_palm_tool_dist"] = right_palm_tool_dist
    info["left_fingertips_object_dist"] = left_fingertips_object_dist
    info["right_fingertips_tool_dist"] = right_fingertips_tool_dist

    is_grasp_left = ((left_fingertips_object_dist <= 0.12 * num_fingers) + (left_palm_object_dist <= 0.12)).float()
    is_grasp_right = ((right_fingertips_tool_dist <= 0.12 * num_fingers) + (right_palm_tool_dist <= 0.12)).float()
    info["left_is_grasp"] = is_grasp_left
    info["right_is_grasp"] = is_grasp_right

    # after hand approach object, lift_object
    ref_object_pos_dist = torch.norm(ref_object_pose[:, :3] - object_pose[:, :3], dim=-1)
    ref_object_rot_rew = quat_rew(ref_object_pose[:, 3:7], object_pose[:, 3:7]) # [-1,1]
    ref_tool_pos_dist = torch.norm(ref_tool_pose[:, :3] - tool_pose[:, :3], dim=-1)
    ref_tool_rot_rew = quat_rew(ref_tool_pose[:, 3:7], tool_pose[:, 3:7])  # [-1,1]

    '''lift object reward for stage 1'''
    left_lift_object_pos_rew1 = (1 - ref_object_pos_dist / ref_init_object_pos_dist).clip(min=0)  # [-1, 1]
    left_lift_object_rot_rew1 = ref_object_rot_rew
    right_lift_tool_pos_rew1 = (1 - ref_tool_pos_dist / ref_init_tool_pos_dist).clip(min=0)  # [-1, 1]
    right_lift_tool_rot_rew1 = ref_tool_rot_rew
    '''lift object reward for stage 2: trajectory following'''
    # trajectory following
    left_successes = torch.logical_and(ref_object_pos_dist <= success_tolerance, left_fingertips_object_dist + left_palm_object_dist < 0.12 * (num_fingers + 1)).float()
    right_successes = torch.logical_and(ref_tool_pos_dist <= success_tolerance, right_fingertips_tool_dist + right_palm_tool_dist < 0.12 * (num_fingers + 1)).float()
    stage1_cul_left_successes = torch.where(left_successes > 0, left_successes + stage1_cul_left_successes, torch.zeros_like(left_successes))
    stage1_cul_right_successes = torch.where(right_successes > 0, right_successes + stage1_cul_right_successes, torch.zeros_like(right_successes))
    stage1_left_success_flag = stage1_cul_left_successes >= frequency
    stage1_right_success_flag = stage1_cul_right_successes >= frequency
    left_reach_ref_timestep = torch.where(torch.logical_and(stage1_left_success_flag, left_reach_ref_timestep == -1), timestep, left_reach_ref_timestep)
    right_reach_ref_timestep = torch.where(torch.logical_and(stage1_right_success_flag, right_reach_ref_timestep == -1), timestep, right_reach_ref_timestep)
    reach_ref_timestep = torch.where(torch.logical_and(torch.logical_and(stage1_left_success_flag, stage1_right_success_flag), reach_ref_timestep == -1), timestep, reach_ref_timestep)
    info["left_successes"] = left_successes
    info["right_successes"] = right_successes
    
    left_lift_object_pos_rew2 = torch.exp(-15 * ref_object_pos_dist) 
    left_lift_object_rot_rew2 = ref_object_rot_rew
    right_lift_tool_pos_rew2 = torch.exp(-15 * ref_tool_pos_dist)
    right_lift_tool_rot_rew2 = ref_tool_rot_rew
    '''lift object reward'''
    left_lift_object_pos_rew = torch.where(
        is_grasp_left > 0,
        torch.where(reach_ref_timestep == -1, left_lift_object_pos_rew1, left_lift_object_pos_rew2),
        torch.zeros_like(ref_object_pos_dist),
    )
    left_lift_object_rot_rew = torch.where(
        is_grasp_left > 0,
        torch.where(reach_ref_timestep == -1, left_lift_object_rot_rew1, left_lift_object_rot_rew2),
        torch.zeros_like(ref_object_pos_dist),
    )
    right_lift_tool_pos_rew = torch.where(
        is_grasp_right > 0,
        torch.where(reach_ref_timestep == -1, right_lift_tool_pos_rew1, right_lift_tool_pos_rew2),
        torch.zeros_like(ref_tool_pos_dist),
    )
    right_lift_tool_rot_rew = torch.where(
        is_grasp_right > 0,
        torch.where(reach_ref_timestep == -1, right_lift_tool_rot_rew1, right_lift_tool_rot_rew2),
        torch.zeros_like(ref_tool_pos_dist),
    )
    info["left_lift_object_pos_rew"] = left_lift_object_pos_rew
    info["left_lift_object_rot_rew"] = left_lift_object_rot_rew
    info["right_lift_tool_pos_rew"] = right_lift_tool_pos_rew
    info["right_lift_tool_rot_rew"] = right_lift_tool_rot_rew

    # stage 3: lift near goal bonus
    left_stage1_bonus = torch.zeros_like(ref_object_pos_dist)
    left_stage1_bonus = torch.where(
        is_grasp_left,
        torch.where(
            ref_object_pos_dist <= success_tolerance, 1.0 / (1 + ref_object_pos_dist), left_stage1_bonus
        ),
        left_stage1_bonus,
    )
    right_stage1_bonus = torch.zeros_like(ref_object_pos_dist)
    right_stage1_bonus = torch.where(
        is_grasp_right,
        torch.where(
            ref_tool_pos_dist <= success_tolerance, 1.0 / (1 + ref_tool_pos_dist), right_stage1_bonus
        ),
        right_stage1_bonus,
    )
    info["left_stage1_bonus"] = left_stage1_bonus
    info["right_stage1_bonus"] = right_stage1_bonus

    # hand-object joint pose difference reward only in stage 1, notice no grasp condition 
    # record below for reward design
    left_object_pos_wrt_palm, left_object_ori_wrt_palm = compute_relative_pose(
        left_palm_pose[:, :3], left_palm_pose[:, 3:7], object_pose[:, :3], object_pose[:, 3:7], 
    )
    right_tool_pos_wrt_palm, right_tool_ori_wrt_palm = compute_relative_pose(
        right_palm_pose[:, :3], right_palm_pose[:, 3:7], tool_pose[:, :3], tool_pose[:, 3:7]
    )
    info["left_object_palm_pos_dist"] = torch.norm(left_object_pos_wrt_palm, dim=-1)
    info["right_tool_palm_pos_dist"] = torch.norm(right_tool_pos_wrt_palm, dim=-1)
    left_object_pos_wrt_fingers = left_fingertip_pose[...,:3] - object_pose[:, None, :3]
    right_tool_pos_wrt_fingers = right_fingertip_pose[...,:3] - tool_pose[:, None, :3]
    info["left_object_fingertip_pos_dist"] = torch.norm(left_object_pos_wrt_fingers, dim=-1).mean(-1)
    info["right_tool_fingertip_pos_dist"] = torch.norm(right_tool_pos_wrt_fingers, dim=-1).mean(-1)
    '''
    if is_stage1_hand_object_rew:
        # hand-object relative reward design, below are all unreasonable!!!
        left_ref_ref_object_palm_pos_dist = pos_error(left_object_pos_wrt_palm, ref_ref_object_palm_pose_diff[:, :3])
        left_ref_ref_object_palm_rot_dist = quat_diff_theta(left_object_ori_wrt_palm, ref_ref_object_palm_pose_diff[:, 3:]).abs()
        right_ref_ref_tool_palm_pos_dist = pos_error(right_tool_pos_wrt_palm, ref_ref_tool_palm_pose_diff[:, :3])
        right_ref_ref_tool_palm_rot_dist = quat_diff_theta(right_tool_ori_wrt_palm, ref_ref_tool_palm_pose_diff[:, 3:]).abs()
        info["left_ref_ref_object_palm_pos_dist"] = left_ref_ref_object_palm_pos_dist
        info["left_ref_ref_object_palm_rot_dist"] = left_ref_ref_object_palm_rot_dist
        info["right_ref_ref_tool_palm_pos_dist"] = right_ref_ref_tool_palm_pos_dist
        info["right_ref_ref_tool_palm_rot_dist"] = right_ref_ref_tool_palm_rot_dist
        left_ref_ref_object_fingers_pos_dist = pos_error(left_object_pos_wrt_fingers, ref_ref_object_left_fingers_pos_diff).mean(-1)
        right_ref_ref_tool_fingers_pos_dist = pos_error(right_tool_pos_wrt_fingers, ref_ref_tool_right_fingers_pos_diff).mean(-1)
        info["left_ref_ref_object_fingers_pos_dist"] = left_ref_ref_object_fingers_pos_dist
        info["right_ref_ref_tool_fingers_pos_dist"] = right_ref_ref_tool_fingers_pos_dist
        # if not is_stage1_lin_rew:  # TODO: quadratic or saturated reward
        #     trans_scale, rot_eps = 3.5, 0.1
        #     # left_pos_idx = (rot_eps / trans_scale) / torch.max(left_object_hand_pos_dist, torch.tensor(rot_eps / trans_scale).to(actions.device))
        #     left_object_hand_pos_rew = 1.0 / (trans_scale * torch.abs(left_ref_ref_object_palm_pos_dist) + rot_eps)# * left_pos_idx
        #     # left_rot_idx = rot_eps / torch.max(left_object_hand_rot_dist, torch.tensor(rot_eps).to(actions.device))
        #     left_object_hand_rot_rew = 1.0 / (left_ref_ref_object_palm_rot_dist + rot_eps)# * left_rot_idx
        #     left_object_hand_pos_rew, left_object_hand_rot_rew = 0.1 * left_object_hand_pos_rew.clip(max=1.5), 0.1 * left_object_hand_rot_rew.clip(max=1.5)
        #     # right_pos_idx = (rot_eps / trans_scale) / torch.max(right_tool_hand_pos_dist, torch.tensor(rot_eps / trans_scale).to(actions.device))
        #     right_tool_hand_pos_rew = 1.0 / (trans_scale * torch.abs(right_ref_ref_tool_palm_pos_dist) + rot_eps)#  * right_pos_idx
        #     # right_rot_idx = rot_eps / torch.max(right_tool_hand_rot_dist, torch.tensor(rot_eps).to(actions.device))
        #     right_tool_hand_rot_rew = 1.0 / (right_ref_ref_tool_palm_rot_dist + rot_eps)#  * right_rot_idx
        #     right_tool_hand_pos_rew, right_tool_hand_rot_rew = 0.1 * right_tool_hand_pos_rew.clip(max=1.5), 0.1 * right_tool_hand_rot_rew.clip(max=1.5)
        # else:  # linear reward
        #     left_object_hand_pos_rew = - left_ref_ref_object_palm_pos_dist
        #     left_object_hand_rot_rew = - 0.3 * left_ref_ref_object_palm_rot_dist
        #     right_tool_hand_pos_rew = - right_ref_ref_tool_palm_pos_dist
        #     right_tool_hand_rot_rew = - 0.3 * right_ref_ref_tool_palm_rot_dist
        
        left_object_palm_pos_rew = torch.exp(-200 * left_ref_ref_object_palm_pos_dist)
        left_object_fingers_pos_rew = torch.exp(-200 * left_ref_ref_object_fingers_pos_dist)
        right_tool_palm_pos_rew = torch.exp(-200 * right_ref_ref_tool_palm_pos_dist)
        right_tool_fingers_pos_rew = torch.exp(-200 * right_ref_ref_tool_fingers_pos_dist)
        # TODO: torch.minimum to force
        left_object_hand_pose_rew = torch.where(reach_ref_timestep == -1, 0.5 * (left_object_palm_pos_rew + left_object_fingers_pos_rew), torch.zeros_like(ref_object_pos_dist))
        right_tool_hand_pose_rew = torch.where(reach_ref_timestep == -1, 0.5 * (right_tool_palm_pos_rew + right_tool_fingers_pos_rew), torch.zeros_like(ref_tool_pos_dist))
        info["left_object_hand_pose_rew"] = left_object_hand_pose_rew
        info["right_tool_hand_pose_rew"] = right_tool_hand_pose_rew

    else:
        left_object_hand_pose_rew = torch.zeros_like(ref_object_pos_dist)
        right_tool_hand_pose_rew = torch.zeros_like(ref_tool_pos_dist)
    info["left_object_hand_pose_rew"] = left_object_hand_pose_rew
    info["right_tool_hand_pose_rew"] = right_tool_hand_pose_rew
    '''

    # every-step success
    info["step-success"] = torch.logical_and(ref_object_pos_dist <= success_tolerance, ref_tool_pos_dist <= success_tolerance).float()
    # stage 1 success
    stage1_left_successes = torch.logical_or(stage1_left_successes, stage1_left_success_flag)
    stage1_right_successes = torch.logical_or(stage1_right_successes, stage1_right_success_flag)
    stage1_successes = torch.logical_or(torch.logical_and(stage1_left_successes, stage1_right_successes), stage1_successes)
    # satge 2 success
    stage2_left_successes = torch.where(stage1_left_successes, ((timestep - reach_ref_timestep) * stage2_left_successes + (ref_object_pos_dist <= success_tolerance)) / (timestep - reach_ref_timestep + 1), stage2_left_successes)
    stage2_right_successes = torch.where(stage1_right_successes, ((timestep - reach_ref_timestep) * stage2_right_successes + (ref_tool_pos_dist <= success_tolerance)) / (timestep - reach_ref_timestep + 1), stage2_right_successes)
    stage2_successes = torch.where(stage1_successes, ((timestep - reach_ref_timestep) * stage2_successes + torch.logical_and(ref_object_pos_dist <= success_tolerance, ref_tool_pos_dist <= success_tolerance)) / (timestep - reach_ref_timestep + 1), stage2_successes)
    
    # bonus for second stage success
    # stage12_successes = torch.logical_and(stage1_successes, stage2_successes >= 0.5).float()
    # left_stage2_bonus = torch.where(is_expect_end, stage2_left_successes, torch.zeros_like(stage2_successes))
    # right_stage2_bonus = torch.where(is_expect_end, stage2_right_successes, torch.zeros_like(stage2_successes))
    # info["left_stage2_bonus"] = left_stage2_bonus
    # info["right_stage2_bonus"] = right_stage2_bonus

    # total reward
    left_approach_penalty = dist_reward_scale * left_fingertips_object_dist + 2 * dist_reward_scale * left_palm_object_dist
    right_approach_penalty = dist_reward_scale * right_fingertips_tool_dist + 2 * dist_reward_scale * right_palm_tool_dist
    left_lift_to_refpose_reward = left_lift_object_pos_rew + 0.2 * left_lift_object_rot_rew
    right_lift_to_refpose_reward = right_lift_tool_pos_rew + 0.2 * right_lift_tool_rot_rew 
    info["left_approach_penalty"] = left_approach_penalty
    info["left_lift_to_refpose_reward"] = left_lift_to_refpose_reward
    info["right_approach_penalty"] = right_approach_penalty
    info["right_lift_to_refpose_reward"] = right_lift_to_refpose_reward

    left_reward = - left_approach_penalty + left_lift_to_refpose_reward + left_stage1_bonus #+ left_stage2_bonus + left_object_hand_pose_rew 
    right_reward = - right_approach_penalty + right_lift_to_refpose_reward + right_stage1_bonus #+ right_stage2_bonus + right_tool_hand_pose_rew
    reward = left_reward + right_reward
    info["left_reward"] = left_reward
    info["right_reward"] = right_reward
    info["reward"] = reward

    # if random.random() < 0.03:
    #     print(left_object_hand_pos_dist,left_object_hand_rot_dist,right_tool_hand_pos_dist,right_tool_hand_rot_dist)
    #     breakpoint()

    # reset
    resets = reset_buf.clone()
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)   # 1. reach max episode length
    resets = torch.where(torch.logical_or(object_pose[:, 2] <= table_heights, tool_pose[:, 2] <= table_heights), torch.ones_like(resets), resets)  # 2. fall under table
    resets = torch.where(torch.logical_or(torch.pairwise_distance(left_robot_link1_pos, object_pose[:, :3]) >= 0.85, torch.pairwise_distance(right_robot_link1_pos, tool_pose[:, :3]) >= 0.85), torch.ones_like(resets), resets)  # 3. object out of scope
    resets = torch.where(is_expect_end, torch.ones_like(resets), resets)  # 4. dataset end

    return (
        reward,
        resets,
        progress_buf,
        stage1_left_successes, stage1_right_successes, stage1_successes, stage1_cul_left_successes, stage1_cul_right_successes,
        stage2_left_successes, stage2_right_successes, stage2_successes,
        timestep, reach_ref_timestep, left_reach_ref_timestep, right_reach_ref_timestep,
        info,
    )


def read_pointcloud_from_urdf(urdf_file, num_sample=4096):
    robot = URDF.load(urdf_file)
    all_points = []
    for link in robot.links:
        for visual in link.visuals:
            if visual.geometry.mesh is not None:
                mesh = trimesh.load_mesh(os.path.join(urdf_file, '..', visual.geometry.mesh.filename))
                if visual.geometry.mesh.scale is not None:  # scale=0.01
                    mesh.apply_scale(visual.geometry.mesh.scale)
                points = mesh.sample(num_sample) 
                all_points.append(points)
    all_points = np.vstack(all_points)
    return torch.from_numpy(all_points).float().cuda()



class BiLeapHandGraspM3Dagger(VecTask):
    '''
    Dagger for multi object objects diversities and ids
    '''
    def get_obs_idx_dict(self,obs_type=''):
        if obs_type == '':
            obs_type = self.obs_type
        cnt = 0
        left_robostate_idx, right_robostate_idx = [], []
        left_pointcloud_idx, right_pointcloud_idx = [], []
        left_objlabel_idx, right_objlabel_idx = [], []
        left_futureobjps_idx, right_futureobjps_idx = [], []

        if 'dofps' in obs_type:  # dof pos, 44 
            num_robot_dofs = 44
            left_robostate_idx.extend(list(range(cnt, cnt + num_robot_dofs//2)))
            right_robostate_idx.extend(list(range(cnt + num_robot_dofs//2, cnt + num_robot_dofs)))
            cnt += num_robot_dofs

        if 'dofvel' in obs_type:  # dof vel, 44
            num_robot_dofs = 44
            left_robostate_idx.extend(list(range(cnt, cnt + num_robot_dofs//2)))
            right_robostate_idx.extend(list(range(cnt + num_robot_dofs//2, cnt + num_robot_dofs)))
            cnt += num_robot_dofs

        if 'ftps' in obs_type:  # fingertip pos, 3 * 4 * 2
            num_ft_states = 4 * 3
            left_robostate_idx.extend(list(range(cnt, cnt + num_ft_states)))
            right_robostate_idx.extend(list(range(cnt + num_ft_states, cnt + 2 * num_ft_states)))
            cnt += 2 * num_ft_states

        if 'ftstate' in obs_type:  # fingertip state, 13 * 4 * 2
            num_ft_states = 4 * 13
            left_robostate_idx.extend(list(range(cnt, cnt + num_ft_states)))
            right_robostate_idx.extend(list(range(cnt + num_ft_states, cnt + 2 * num_ft_states)))
            cnt += 2 * num_ft_states

        if 'lastact' in obs_type:  # last action, 44
            num_actions = 44
            left_robostate_idx.extend(list(range(cnt, cnt + num_actions//2)))
            right_robostate_idx.extend(list(range(cnt + num_actions//2, cnt + num_actions)))
            cnt += num_actions

        if 'objpose' in obs_type:  # object pose, 7 * 2
            obj_dim = 7
            left_robostate_idx.extend(list(range(cnt, cnt + obj_dim)))
            right_robostate_idx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim

        if 'objstate' in obs_type:  # object state, pose, linvel, angvel. 13 * 2
            obj_dim = 13
            left_robostate_idx.extend(list(range(cnt, cnt + obj_dim)))
            right_robostate_idx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim
        
        if 'palmps' in obs_type:  # palm pos, 3 * 2
            obj_dim = 3
            left_robostate_idx.extend(list(range(cnt, cnt + obj_dim)))
            right_robostate_idx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim

        if 'palmpose' in obs_type:  # palm pose, 7 * 2
            obj_dim = 7
            left_robostate_idx.extend(list(range(cnt, cnt + obj_dim)))
            right_robostate_idx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim

        if 'palmstate' in obs_type: # palm state, 13 * 2
            obj_dim = 13
            left_robostate_idx.extend(list(range(cnt, cnt + obj_dim)))
            right_robostate_idx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim

        if 'relps' in obs_type:  # relative pos to object center, 15 * 2
            relpos_dim = 3 * (4 + 1)
            left_robostate_idx.extend(list(range(cnt, cnt + relpos_dim)))
            right_robostate_idx.extend(list(range(cnt + relpos_dim, cnt + 2 * relpos_dim)))
            cnt += 2 * relpos_dim

        if 'meshpc' in obs_type:  # point cloud from object mesh
            self.num_pc_downsample = self.cfg['env']['vision']['pointclouds']['numDownsample']
            self.num_each_pt = self.cfg['env']['vision']['pointclouds']['numEachPoint']
            self.num_pc_flatten = self.num_pc_downsample * self.num_each_pt
            left_pointcloud_idx.extend(list(range(cnt, cnt + self.num_pc_flatten)))
            right_pointcloud_idx.extend(list(range(cnt + self.num_pc_flatten, cnt + 2 * self.num_pc_flatten)))
            cnt += 2 * self.num_pc_flatten

        if 'objlabel' in obs_type:  # object label, 1
            label_dim = 1
            left_objlabel_idx.extend(list(range(cnt, cnt + label_dim)))
            right_objlabel_idx.extend(list(range(cnt + label_dim, cnt + 2 * label_dim)))
            cnt += 2 * label_dim

        if 'futureps' in self.obs_type:
            Kfuturestep_dim = 3 * self.Kfuturestep
            left_futureobjps_idx.extend(list(range(cnt, cnt + Kfuturestep_dim)))
            right_futureobjps_idx.extend(list(range(cnt + Kfuturestep_dim, cnt + 2 * Kfuturestep_dim)))
            cnt += 2 * Kfuturestep_dim

        assert cnt == len(left_robostate_idx) + len(right_robostate_idx) + len(left_pointcloud_idx) + len(right_pointcloud_idx) + len(left_objlabel_idx) + len(right_objlabel_idx) + len(left_futureobjps_idx) + len(right_futureobjps_idx)
        left_indices = dict(
            robostate_indices=left_robostate_idx,
            pointcloud_indices=left_pointcloud_idx,
            objlabel_indices=left_objlabel_idx,
            futureobjps_indices=left_futureobjps_idx,
        )
        right_indices = dict(
            robostate_indices=right_robostate_idx,
            pointcloud_indices=right_pointcloud_idx,
            objlabel_indices=right_objlabel_idx,
            futureobjps_indices=right_futureobjps_idx,
        )
        return left_indices, right_indices, cnt

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
        **kwargs,
    ):
        self.cfg = cfg
        self.mode = self.cfg["mode"]
        self.frequency, self.horizon = self.cfg["task"]['frequency'], self.cfg["task"]['horizon']
        self.is_stage1_hand_object_rew = self.cfg["task"]["isStage1HOReward"]
        self.is_stage1_lin_rew = self.cfg["task"]["isStage1LinReward"]
        self.is_stage2_pos_rew_exp = self.cfg["task"]["isStage2PosRewExp"]
        self.Kfuturestep = 5

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
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)
        
        # pointcloud
        self.n_resample = self.cfg['env']['vision']['pointclouds']['numDownsample']
        self.num_max_sample_points = self.cfg['env']['vision']['pointclouds']['nMaxSamplePoints']
        self.num_sample_points = self.cfg['env']['vision']['pointclouds']['nSamplePoints']
        self.apply_pointcloud_noise = self.cfg['env']['vision']['noise']['apply']
        self.pointcloud_noise_scale = self.cfg['env']['vision']['noise']['scale']
        self.pointcloud_noise_threshold = self.cfg['env']['vision']['noise']['threshold']
        
        self.obs_type = self.cfg["env"]["observationType"]

        assert self.arm_controller in ["ik", "qpos"]

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        self.cfg["env"]["numObservations"] = self.get_obs_idx_dict()[-1]
        print(f'number of observation: {self.cfg["env"]["numObservations"]}')
        self.cfg["env"]["numStates"] = 0
        self.cfg["env"]["numActions"] = 44

        if self.arm_controller == "ik":  # use rotation 6D representation
            self.cfg["env"]["numObservations"] += 3
            self.cfg["env"]["numStates"] += 3 if self.asymmetric_obs else 0
            self.cfg["env"]["numActions"] += 3

        # need to set the names according to the robot
        self.palm = "palm"#_lower
        self.fingertips = [
            "thumb_tip_head",
            "index_tip_head",
            "middle_tip_head",
            "ring_tip_head",
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
            **kwargs,
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
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(*self.cfg['env']['cameraPosition']), gymapi.Vec3(*self.cfg['env']['cameraTarget']))

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
        self.stage1_left_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stage1_right_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stage1_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stage1_cul_left_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stage1_cul_right_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stage2_left_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stage2_right_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stage2_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # customize
        self.timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reach_ref_timestep = - torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.left_reach_ref_timestep = - torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.right_reach_ref_timestep = - torch.ones(self.num_envs, dtype=torch.long, device=self.device)

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
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)),)

        # if randamizing, apply once immediately on startup before the first sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _generate_urdf(self, object_dict):
        assert "id" in object_dict
        link_id = object_dict["id"]
        xyz = ' '.join(map(str, object_dict["xyz"])) if "xyz" in object_dict else '0 0 0'
        rpy = ' '.join(map(str, object_dict["rpy"])) if "rpy" in object_dict else '0 0 0'
        # modify the scale of object here
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
        self.left_robot_dof_lower_limits, self.left_robot_dof_upper_limits = self._prepare_robot_asset(asset_root, self.cfg["env"]["asset"]["leftAssetFile"])
        right_asset, right_dof_props, self.right_palm_handle, self.right_fingertip_handles, self.right_eef_index,\
        self.right_arm_dof_indices, self.right_fingers_dof_indices, self.right_robot_dof_indices, \
        self.right_robot_dof_lower_limits, self.right_robot_dof_upper_limits = self._prepare_robot_asset(asset_root, self.cfg["env"]["asset"]["rightAssetFile"])    

        self.robot_dof_lower_limits = to_torch(self.left_robot_dof_lower_limits + self.right_robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.left_robot_dof_upper_limits + self.right_robot_dof_upper_limits, device=self.device)
        
        # get hand asset info
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(left_asset) + self.gym.get_asset_rigid_body_count(right_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(left_asset) + self.gym.get_asset_rigid_shape_count(right_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(left_asset) + self.gym.get_asset_dof_count(right_asset)
        self.robot_dof_default_pos = torch.zeros(self.num_robot_dofs, dtype=torch.float, device=self.device)  
        self.robot_dof_default_vel = torch.zeros(self.num_robot_dofs, dtype=torch.float, device=self.device)  
        # update right handles & dof_indices
        self.right_palm_handle += self.num_robot_bodies//2
        self.right_fingertip_handles = [i + self.num_robot_bodies//2 for i in self.right_fingertip_handles]
        self.right_arm_dof_indices = [i + self.num_robot_dofs//2 for i in self.right_arm_dof_indices]
        self.right_fingers_dof_indices = [i + self.num_robot_dofs//2 for i in self.right_fingers_dof_indices]
        self.right_robot_dof_indices = [i + self.num_robot_dofs//2 for i in self.right_robot_dof_indices]
        self.both_arm_dof_indices = to_torch(self.left_arm_dof_indices + self.right_arm_dof_indices, dtype=torch.long, device=self.device)
        self.both_fingers_dof_indices = to_torch(self.left_fingers_dof_indices + self.right_fingers_dof_indices, dtype=torch.long, device=self.device)
        self.both_robot_dof_indices = to_torch(self.left_robot_dof_indices + self.right_robot_dof_indices, dtype=torch.long, device=self.device)

        self.envs, self.cameras = [], []
        self.left_robot_indices, self.right_robot_indices = [], []
        self.object_indices, self.tool_indices = [], []
        self.left_start_states, self.right_start_states = [], []
        self.object_init_states, self.tool_init_states = [], []
        self.table_heights = []
        if self.arm_controller == "ik":
            self.eef_idx = []

        self._prepare_dataset()  
        
        self.all_ref_object_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self.all_ref_tool_poses = torch.zeros((self.num_envs, 7), device=self.device)

        for i in range(num_envs):
            i_task = i % self.num_task  # multi-objects
            self.all_ref_object_poses[i] = self.dataset_object_poses[i_task,self.dataset_ref_timesteps[i_task]]
            self.all_ref_tool_poses[i] = self.dataset_tool_poses[i_task,self.dataset_ref_timesteps[i_task]]

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            object_asset, tool_asset = self.object_assets[i_task], self.tool_assets[i_task]

            if self.aggregate_mode > 0:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies[i_task], self.max_agg_shapes[i_task], True)

            # create robot actor
            left_robot_start_pose, right_robot_start_pose = self.left_robot_start_poses[i_task], self.right_robot_start_poses[i_task]
            left_robot_actor = self.gym.create_actor(env_ptr, left_asset, left_robot_start_pose, "left", i, -1, 0)
            right_robot_actor = self.gym.create_actor(env_ptr, right_asset, right_robot_start_pose, "right", i, -1, 0)
            self.left_start_states.append([
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
            ])
            self.right_start_states.append([
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
            ])
            self.gym.set_actor_dof_properties(env_ptr, left_robot_actor, left_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, right_robot_actor, right_dof_props)
            
            self.left_robot_indices.append(self.gym.get_actor_index(env_ptr, left_robot_actor, gymapi.DOMAIN_SIM))
            self.right_robot_indices.append(self.gym.get_actor_index(env_ptr, right_robot_actor, gymapi.DOMAIN_SIM))

            # add object
            object_start_pose = self.object_start_poses[i_task]
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1, 0)
            self.object_init_states.append([
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
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add tool
            tool_start_pose = self.tool_start_poses[i_task]
            tool_handle = self.gym.create_actor(env_ptr, tool_asset, tool_start_pose, "tool", i, -1, 0)
            self.tool_init_states.append([
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
            ])
            tool_idx = self.gym.get_actor_index(env_ptr, tool_handle, gymapi.DOMAIN_SIM)
            self.tool_indices.append(tool_idx)


            # add table
            table_asset, table_start_pose = self.table_assets[i_task], self.table_start_poses[i_task]
            self.table_heights.append(table_start_pose.p.z * 2)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, -1, 0)

            # enable DOF force sensors, if needed
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, right_robot_actor)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

            if self.arm_controller == "ik":
                eef_idx = self.gym.find_actor_rigid_body_index(env_ptr, right_robot_actor, self.palm, gymapi.DOMAIN_SIM)
                self.eef_idx.append(eef_idx)

        self.left_start_states = to_torch(self.left_start_states, device=self.device)
        self.right_start_states = to_torch(self.right_start_states, device=self.device)
        self.object_init_states = to_torch(self.object_init_states, device=self.device)
        self.tool_init_states = to_torch(self.tool_init_states, device=self.device)
        self.left_fingertip_handles = to_torch(self.left_fingertip_handles, dtype=torch.long, device=self.device)
        self.right_fingertip_handles = to_torch(self.right_fingertip_handles, dtype=torch.long, device=self.device)
        self.left_palm_handle = to_torch(self.left_palm_handle, dtype=torch.long, device=self.device)
        self.right_palm_handle = to_torch(self.right_palm_handle, dtype=torch.long, device=self.device)
        self.left_robot_indices = to_torch(self.left_robot_indices, dtype=torch.long, device=self.device)
        self.right_robot_indices = to_torch(self.right_robot_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.tool_indices = to_torch(self.tool_indices, dtype=torch.long, device=self.device)
        self.table_heights = to_torch(self.table_heights, device=self.device)
        self.all_object_labels = to_torch(self.object_labels, dtype=torch.float, device=self.device)[self.all_task_idx] / 255.
        self.all_tool_labels = to_torch(self.tool_labels, dtype=torch.float, device=self.device)[self.all_task_idx] / 255.
        self.all_object_grasp_pos = self.dataset_object_grasp_pos[self.all_task_idx]
        self.all_tool_grasp_pos = self.dataset_tool_grasp_pos[self.all_task_idx]

        self.ref_init_object_pos_dist = torch.norm(self.all_ref_object_poses[:, :3] - self.object_init_states[:, :3], dim=-1)       # (n,)
        self.ref_init_tool_pos_dist = torch.norm(self.all_ref_tool_poses[:, :3] - self.tool_init_states[:, :3], dim=-1)             # (n,)

        # calculate hand-object relative
        # self.ref_ref_object_palm_pos_diff, self.ref_ref_object_palm_rot_diff = compute_relative_pose(                               # (n, 3), (n, 4)
        #     self.dataset_left_palm_ref_poses[self.all_task_idx, :3], self.dataset_left_palm_ref_poses[self.all_task_idx, 3:7],      # (n, 3), (n, 4)
        #     self.all_ref_object_poses[:, :3], self.all_ref_object_poses[:, 3:7], 
        # )
        # self.ref_ref_object_palm_pose_diff = torch.cat([self.ref_ref_object_palm_pos_diff, self.ref_ref_object_palm_rot_diff], dim=-1).expand(self.num_envs, -1)  # (n, 7)
        # self.ref_ref_tool_palm_pos_diff, self.ref_ref_tool_palm_rot_diff = compute_relative_pose(                                   # (n, 3), (n, 4)
        #     self.dataset_right_palm_ref_poses[self.all_task_idx, :3], self.dataset_right_palm_ref_poses[self.all_task_idx, 3:7],
        #     self.all_ref_tool_poses[:, :3], self.all_ref_tool_poses[:, 3:7],
        # )
        # self.ref_ref_tool_palm_pose_diff = torch.cat([self.ref_ref_tool_palm_pos_diff, self.ref_ref_tool_palm_rot_diff], dim=-1).expand(self.num_envs, -1)  # (n, 7)
        

        if self.arm_controller == "ik":
            self.eef_idx = to_torch(self.eef_idx, dtype=torch.long, device=self.device)

    def _prepare_robot_asset(self, asset_root, asset_file, vhacd_enabled=False):
        # load arm hand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if vhacd_enabled:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000

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
        
        return robot_asset, robot_dof_props, palm_handle, fingertip_handles, arm_eef_index, \
                arm_dof_indices, hand_dof_indices, robot_dof_indices, \
                robot_dof_lower_limits, robot_dof_upper_limits

    def _prepare_object_asset(self, asset_root, asset_file, vhacd_enabled, obj_asset_storage, max_shape=-1):
        if obj_asset_storage.get((asset_file, max_shape)) is not None:
            return obj_asset_storage[asset_file, max_shape]
        # load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if vhacd_enabled:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000
            # asset_options.vhacd_params.concavity = 0.0025
            # asset_options.vhacd_params.alpha = 0.04
            # asset_options.vhacd_params.beta = 1.0
            # asset_options.vhacd_params.convex_hull_downsampling = 1 
            if max_shape > 0:
                asset_options.vhacd_params.max_convex_hulls = max_shape 

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0
        object_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        obj_asset_storage[asset_file, max_shape] = object_asset
        return object_asset

    def _prepare_dataset(self, vhacd_enabled=True):                
        dataset_taco_data = {}  # {triplet: taco_data}
        self.object_assets, self.tool_assets, \
        self.object_start_poses, self.tool_start_poses, \
        self.left_robot_start_poses, self.right_robot_start_poses, \
        self.table_assets, self.table_start_poses, \
        self.max_agg_bodies, self.max_agg_shapes, \
        self.dataset_object_poses, self.dataset_tool_poses, \
        self.dataset_ref_timesteps, self.dataset_end_timesteps, \
        self.object_mesh_pointclouds, self.tool_mesh_pointclouds, \
        self.object_labels, self.tool_labels, \
        self.dataset_object_grasp_pos, self.dataset_tool_grasp_pos\
        = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


        self.all_triplet = [ecfile.split('/')[-3] for ecfile in self.train_cfg['expertCkptFiles']]
        category2idx = {
            'brush': 0,
            'cut': 1,
            'dust': 2,
            'empty': 3,
            'hit': 4,
            'measure': 5,
            'pour in some': 6,
            'put in': 7,
            'put out': 8,
            'scrape off': 9,
            'screw': 10,
            'skim off': 11,
            'smear': 12,
            'stir-fry': 13,
            'stir': 14,
        }
        assert len(self.all_triplet) == len(set(self.all_triplet)), 'triplet names should be unique'
        self.num_task = 0
        self.dataset_taco_datas, self.expert_ids, self.verb_category, self.types = [], [], [], []
        with open(os.path.join('taco_dataset/task_data/blacklist.txt')) as f:
            regen_hull_task_list = [line.strip('\n') for line in f]
        is_regen_hull_list = []

        cul_len = 0
        self.train_task_ids, self.test_task_ids = [], []
        train_task_id_lists = self.train_cfg['train_task_id_lists']
        assert len(train_task_id_lists) == len(self.all_triplet)
        for itriplet, triplet in enumerate(self.all_triplet):
            with open(f'taco_dataset/task_data/{triplet}.json', 'r') as f:
                dataset_taco_data = json.load(f)
                train_task_id_list = train_task_id_lists[itriplet]
                if not train_task_id_list:
                    if len(dataset_taco_data) < 5:
                        b, e = 0, len(dataset_taco_data)
                    elif len(dataset_taco_data) < 9:
                        b, e = 1, len(dataset_taco_data)
                    else:
                        proportion = 0.8
                        b, e = 1, int(len(dataset_taco_data) * proportion)
                    train_task_id_list = np.arange(b, e) + cul_len
                else:
                    train_task_id_list = np.array(train_task_id_list) + cul_len
                len_dataset_taco_data = len(dataset_taco_data)
                test_task_ids = np.array([idx for idx in cul_len + np.arange(len_dataset_taco_data) if idx not in train_task_id_list])
                self.train_task_ids.extend(train_task_id_list)
                self.test_task_ids.extend(test_task_ids)
                cur_train_task_id_list = train_task_id_list - cul_len
                cur_test_task_ids = test_task_ids - cul_len
                num_train_set = len(train_task_id_list)
                num_test_set = len(test_task_ids)
                # judge testing type: 0 for training, 1 for testing seen, 2 for testing unseen
                id_pairs = np.array([(dataset_taco_data[k]['left']['object']['id'], dataset_taco_data[k]['right']['tool']['id']) for k in range(len_dataset_taco_data)])
                types = np.int_([_ in test_task_ids for _ in range(cul_len, cul_len + len(dataset_taco_data))])
                unique_trained_object_ids = np.unique(id_pairs[cur_train_task_id_list, 0])
                unique_trained_tool_ids = np.unique(id_pairs[cur_test_task_ids,1])
                for k in range(len_dataset_taco_data):
                    if k in cur_test_task_ids:
                        objid, toolid = id_pairs[k]
                        if objid not in unique_trained_object_ids or toolid not in unique_trained_tool_ids:
                            types[k] = 2
                self.types.extend(types)
                cul_len += len_dataset_taco_data
                if not self.cfg['task']['is_all_task']:
                    print(f'training set: {num_train_set}', f'testing set: {num_test_set}')
                    dataset_taco_data = [dataset_taco_data[idx] for idx in cur_train_task_id_list]
                else:
                    print(f'training set: 0, testing set: {num_train_set + num_test_set}')
                len_dataset_taco_data = len(dataset_taco_data)
                self.dataset_taco_datas.extend(dataset_taco_data)
                is_regen_hull_list.extend([triplet in regen_hull_task_list] * len_dataset_taco_data)
                self.expert_ids.extend([itriplet] * len_dataset_taco_data)
                self.verb_category.extend([category2idx[triplet.strip('()').split(', ')[0]]] * len_dataset_taco_data)
                self.num_task += len_dataset_taco_data

        assert len(self.dataset_taco_datas) == self.num_task
        self.all_task_idx = torch.tensor([i % self.num_task for i in range(self.num_envs)], dtype=torch.long, device=self.device)
        assert len(self.expert_ids) == self.num_task
        self.expert_ids = torch.tensor(self.expert_ids, dtype=torch.long, device=self.device)[self.all_task_idx]
        assert len(self.verb_category) == self.num_task
        self.verb_category = torch.tensor(self.verb_category, dtype=torch.long, device=self.device)[self.all_task_idx]
        self.types = torch.tensor(self.types, dtype=torch.long, device=self.device)[self.all_task_idx]
        obj_asset_storage = dict()
        object_max_shape, tool_max_shape = -1, -1
        for task_id in range(self.num_task):
            object_start_pose, tool_start_pose, \
            left_robot_start_pose, right_robot_start_pose, \
            table_asset, table_start_pose, \
            dataset_object_pose, dataset_tool_pose, \
            ref_timestep, end_timestep, \
            object_grasp_pos, tool_grasp_pos\
            = self._initialize_task(self.dataset_taco_datas[task_id])
            self.dataset_object_grasp_pos.append(object_grasp_pos)
            self.dataset_tool_grasp_pos.append(tool_grasp_pos)
            self.dataset_object_poses.append(dataset_object_pose)   
            self.dataset_tool_poses.append(dataset_tool_pose)
            self.dataset_ref_timesteps.append(ref_timestep)
            self.dataset_end_timesteps.append(end_timestep)    
            # self.dataset_left_palm_ref_poses.append(dataset_left_palm_ref_pose)  
            # self.dataset_right_palm_ref_poses.append(dataset_right_palm_ref_pose)

            self.object_start_poses.append(object_start_pose)
            self.tool_start_poses.append(tool_start_pose)
            self.left_robot_start_poses.append(left_robot_start_pose)
            self.right_robot_start_poses.append(right_robot_start_pose)
            self.table_assets.append(table_asset)
            self.table_start_poses.append(table_start_pose)
            # create object and tool urdf
            objects_mesh_path = os.path.join(self.cfg["env"]["asset"]["assetRoot"], 'TACOobjects')
            object_id = self.dataset_taco_datas[task_id]['left']['object']['id']
            self.object_labels.append(int(object_id))
            task_object_urdf_file = os.path.join(objects_mesh_path, f'{object_id}.urdf')   
            if not os.path.exists(task_object_urdf_file):             
                with open(task_object_urdf_file, 'w') as urdf_file:
                    urdf_file.write(self._generate_urdf(dict(id=object_id)))
            
            self.object_mesh_pointclouds.append(read_pointcloud_from_urdf(task_object_urdf_file, self.num_max_sample_points))

            tool_id = self.dataset_taco_datas[task_id]['right']['tool']['id']
            self.tool_labels.append(int(tool_id))
            task_tool_urdf_file = os.path.join(objects_mesh_path, f'{tool_id}.urdf')
            if not os.path.exists(task_tool_urdf_file):
                with open(task_tool_urdf_file, 'w') as urdf_file:
                    urdf_file.write(self._generate_urdf(dict(id=tool_id)))
            self.tool_mesh_pointclouds.append(read_pointcloud_from_urdf(task_tool_urdf_file, self.num_max_sample_points))
            # get object and tool asset
            while True:
                object_asset = self._prepare_object_asset(*os.path.split(task_object_urdf_file), vhacd_enabled, obj_asset_storage, object_max_shape)
                tool_asset = self._prepare_object_asset(*os.path.split(task_tool_urdf_file), vhacd_enabled, obj_asset_storage, tool_max_shape)
                # aggregate size
                num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset) + self.gym.get_asset_rigid_body_count(tool_asset)
                num_object_shape = self.gym.get_asset_rigid_shape_count(object_asset)
                num_tool_shape = self.gym.get_asset_rigid_shape_count(tool_asset)
                num_object_shapes = num_object_shape + num_tool_shape
                max_agg_bodies = self.num_robot_bodies + num_object_bodies + 2
                max_agg_shapes = self.num_robot_shapes + num_object_shapes + 2
                if not is_regen_hull_list[task_id] or max_agg_shapes <= 128:
                    self.object_assets.append(object_asset)
                    self.tool_assets.append(tool_asset)
                    self.max_agg_bodies.append(max_agg_bodies)
                    self.max_agg_shapes.append(max_agg_shapes)
                    break
                spare_max_shape = 126 - self.num_robot_shapes  # 78
                object_max_shape = int(num_object_shape / (num_object_shape + num_tool_shape) * spare_max_shape)
                tool_max_shape = spare_max_shape - object_max_shape
                print(f'task_id:{task_id} | num_object_shape:{num_object_shape} | num_tool_shape:{num_tool_shape} | object_max_shape:{object_max_shape} | tool_max_shape:{tool_max_shape}')
        self.dataset_object_poses = torch.stack(self.dataset_object_poses, dim=0)  # (K, T, 7)
        self.dataset_tool_poses = torch.stack(self.dataset_tool_poses, dim=0)  # (K, T, 7)
        self.dataset_ref_timesteps = torch.tensor(self.dataset_ref_timesteps, dtype=torch.long, device=self.device)  # (K,)
        self.dataset_end_timesteps = torch.tensor(self.dataset_end_timesteps, dtype=torch.long, device=self.device)  # (K,)
        # self.dataset_left_palm_ref_poses = torch.stack(self.dataset_left_palm_ref_poses, dim=0)  # (K, 7)
        # self.dataset_right_palm_ref_poses = torch.stack(self.dataset_right_palm_ref_poses, dim=0)  # (K, 7)
        self.dataset_object_grasp_pos = to_torch(self.dataset_object_grasp_pos, dtype=torch.float, device=self.device)  # (K, 3)
        self.dataset_tool_grasp_pos = to_torch(self.dataset_tool_grasp_pos, dtype=torch.float, device=self.device)  # (K, 3)
        self.object_mesh_pointclouds = torch.stack(self.object_mesh_pointclouds, dim=0)  # (K, N, 3)
        self.tool_mesh_pointclouds = torch.stack(self.tool_mesh_pointclouds, dim=0)  # (K, N, 3)

    def _initialize_task(self, taco_task_data):
        epi_len = self.cfg['env']['episodeLength']
        # timestep
        init_timestep = taco_task_data['key_steps']['init']
        ref_timestep = taco_task_data['key_steps']['ref']
        end_timestep = taco_task_data['key_steps']['end']
        # object grasp pos
        dataset_object_grasp_pos = np.array(taco_task_data['left']['object']['gpos'])
        dataset_tool_grasp_pos = np.array(taco_task_data['right']['tool']['gpos'])
        # object poses
        dataset_object_pos = np.array(taco_task_data['left']['object']['pos'])
        dataset_object_quat = np.array(taco_task_data['left']['object']['quat'])
        dataset_object_init_pos = dataset_object_pos[init_timestep]
        # tool poses
        dataset_tool_pos = np.array(taco_task_data['right']['tool']['pos'])
        dataset_tool_quat = np.array(taco_task_data['right']['tool']['quat'])
        dataset_tool_init_pos = dataset_tool_pos[init_timestep]
        # table asset and pose
        # table_height = min(dataset_object_init_pos[2],dataset_tool_init_pos[2]) - 0.03  # objects above table
        table_height = 0.7
        table_dim = (1.5, 1.5, table_height)
        table_asset, table_start_pose = self._prepare_table_asset(table_dim)
        # constants
        object_center_coord = (dataset_object_init_pos + dataset_tool_init_pos) / 2
        rbx, rby, rbz = 0.34, 0.4, table_height + 0.52
        left_robot_start_pose = gymapi.Transform()
        self.left_robot_link1_pos = torch.tensor([[-rbx, 0.24 - rby, rbz]], dtype=torch.float, device=self.device)
        left_robot_start_pose.p = gymapi.Vec3(-rbx, -rby, rbz)
        left_robot_start_pose.r = gymapi.Quat(0.5,  0.5,  0.5, -0.5)
        right_robot_start_pose = gymapi.Transform()
        self.right_robot_link1_pos = torch.tensor([[rbx, 0.24 - rby, rbz]], dtype=torch.float, device=self.device)
        right_robot_start_pose.p = gymapi.Vec3(rbx, -rby, rbz)
        right_robot_start_pose.r = gymapi.Quat(0.5,  0.5,  0.5, -0.5)
        # add offset to dataset
        offset = np.array([-object_center_coord[0], -object_center_coord[1], table_height + 0.03 - min(dataset_object_init_pos[2],dataset_tool_init_pos[2])])
        dataset_object_pose = to_torch(torch.from_numpy(np.concatenate([
            dataset_object_pos + offset,
            dataset_object_quat,
        ], axis=-1)), device=self.device, dtype=torch.float)
        # (epi_len, 7)
        if epi_len > len(dataset_object_pose):
            dataset_object_pose = torch.cat([dataset_object_pose, dataset_object_pose[-1].repeat(epi_len - len(dataset_object_pose), 1)])
        else:
            dataset_object_pose = dataset_object_pose[:epi_len]
        
        dataset_tool_pose = to_torch(torch.from_numpy(np.concatenate([
            dataset_tool_pos + offset,
            dataset_tool_quat,
        ], axis=-1)), device=self.device, dtype=torch.float)
        # (epi_len, 7)
        if epi_len > len(dataset_tool_pose):
            dataset_tool_pose = torch.cat([dataset_tool_pose, dataset_tool_pose[-1].repeat(epi_len - len(dataset_tool_pose), 1)])
        else:
            dataset_tool_pose = dataset_tool_pose[:epi_len]
        
        # initial object poses
        self.objoffset = self.cfg['env'].get('objectOffset', 0.1)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(-self.objoffset, 0, dataset_object_init_pos[2] + offset[2])
        object_start_pose.r = gymapi.Quat(0,0,0,1)
        tool_start_pose = gymapi.Transform()
        tool_start_pose.p = gymapi.Vec3(self.objoffset, 0, dataset_tool_init_pos[2] + offset[2])
        tool_start_pose.r = gymapi.Quat(0,0,0,1)
        '''
        # left palm poses
        dataset_left_palm_pos = np.array(self.sampled_taco_task_data['left']['palm']['pos'])
        dataset_left_palm_quat = np.array(self.sampled_taco_task_data['left']['palm']['quat'])
        # (7)
        dataset_left_palm_pose = to_torch(torch.from_numpy(np.concatenate([
            dataset_left_palm_pos[[ref_timestep]],
            dataset_left_palm_quat[[ref_timestep]],
        ], axis=-1)), device=self.device, dtype=torch.float)
        # right palm poses
        dataset_right_palm_pos = np.array(self.sampled_taco_task_data['right']['palm']['pos'])
        dataset_right_palm_quat = np.array(self.sampled_taco_task_data['right']['palm']['quat'])
        # (7)
        dataset_right_palm_pose = to_torch(torch.from_numpy(np.concatenate([
            dataset_right_palm_pos[[ref_timestep]],
            dataset_right_palm_quat[[ref_timestep]],
        ], axis=-1)), device=self.device, dtype=torch.float)
        '''
        

        return object_start_pose, tool_start_pose, \
            left_robot_start_pose, right_robot_start_pose, \
            table_asset, table_start_pose, \
            dataset_object_pose, dataset_tool_pose, \
            ref_timestep, end_timestep, \
            dataset_object_grasp_pos, dataset_tool_grasp_pos,\
            # dataset_left_palm_pose, dataset_right_palm_pose

    def _prepare_table_asset(self, table_dims):
        # create table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *table_dims, asset_options)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.0, 0.0, table_dims[-1] / 2)

        return table_asset, table_start_pose

    def compute_reward(self, mode):
        if mode == 's12':
            t = torch.where(self.reach_ref_timestep == -1, torch.zeros_like(self.timestep), torch.ceil((self.timestep - self.reach_ref_timestep) / self.frequency).long()) + self.dataset_ref_timesteps[self.all_task_idx]
            tl = torch.where(self.left_reach_ref_timestep == -1, torch.zeros_like(self.timestep), torch.ceil((self.timestep - self.left_reach_ref_timestep) / self.frequency).long()) + self.dataset_ref_timesteps[self.all_task_idx]
            tr = torch.where(self.right_reach_ref_timestep == -1, torch.zeros_like(self.timestep), torch.ceil((self.timestep - self.right_reach_ref_timestep) / self.frequency).long()) + self.dataset_ref_timesteps[self.all_task_idx]
            ref_object_pose = self.dataset_object_poses[self.all_task_idx,tl.clip(max=self.dataset_end_timesteps[self.all_task_idx])] 
            ref_tool_pose = self.dataset_tool_poses[self.all_task_idx,tr.clip(max=self.dataset_end_timesteps[self.all_task_idx])]
            is_expect_end = (self.dataset_end_timesteps[self.all_task_idx] == t.clip(max=self.dataset_end_timesteps[self.all_task_idx]))
            (
                self.rew_buf[:],
                self.reset_buf[:],
                self.progress_buf[:],
                self.stage1_left_successes[:], self.stage1_right_successes[:], self.stage1_successes[:], self.stage1_cul_left_successes[:], self.stage1_cul_right_successes[:],
                self.stage2_left_successes[:], self.stage2_right_successes[:], self.stage2_successes[:],
                self.timestep[:], self.reach_ref_timestep[:], self.left_reach_ref_timestep[:], self.right_reach_ref_timestep[:],
                reward_info
            ) = compute_bvdex_stage12_rewards(
                self.reset_buf,
                self.progress_buf,
                self.stage1_left_successes, self.stage1_right_successes, self.stage1_successes, self.stage1_cul_left_successes, self.stage1_cul_right_successes,
                self.stage2_left_successes, self.stage2_right_successes, self.stage2_successes,
                self.max_episode_length,
                self.object_pose, self.tool_pose,
                self.left_palm_pose, self.right_palm_pose,
                self.left_fingertip_pose, self.right_fingertip_pose,
                self.dist_reward_scale,
                self.action_penalty_scale,
                self.success_tolerance,
                self.av_factor,
                self.table_heights, self.left_robot_link1_pos, self.right_robot_link1_pos,
                self.frequency,
                self.timestep, self.reach_ref_timestep, self.left_reach_ref_timestep, self.right_reach_ref_timestep,
                ref_object_pose, self.ref_init_object_pos_dist, #self.ref_ref_object_palm_pose_diff, self.ref_ref_object_left_fingers_pos_diff,
                ref_tool_pose, self.ref_init_tool_pos_dist,     #self.ref_ref_tool_palm_pose_diff, self.ref_ref_tool_right_fingers_pos_diff,
                self.actual_object_grasp_pos, self.actual_tool_grasp_pos,
                is_expect_end, # self.is_stage1_hand_object_rew, self.is_stage1_lin_rew, self.is_stage2_pos_rew_exp,
            )

        self.extras.update(reward_info)
        self.extras["stage1_left_successes"] = self.stage1_left_successes
        self.extras["stage1_right_successes"] = self.stage1_right_successes
        self.extras["stage1_successes"] = self.stage1_successes
        self.extras["stage2_left_successes"] = self.stage2_left_successes
        self.extras["stage2_right_successes"] = self.stage2_right_successes
        self.extras["stage2_successes"] = self.stage2_successes

        return reward_info

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
        self.left_palm_pose = self.left_palm_state[..., :7]
        self.left_palm_pos = self.left_palm_state[..., :3]
        self.left_palm_rot = self.left_palm_state[..., 3:7]
        self.right_palm_state = self.rigid_body_states[:, self.right_palm_handle][..., :13]
        self.right_palm_pose = self.right_palm_state[..., :7]
        self.right_palm_pos = self.right_palm_state[..., :3]
        self.right_palm_rot = self.right_palm_state[..., 3:7]

        self.left_fingertip_state = self.rigid_body_states[:, self.left_fingertip_handles][..., :13]
        self.left_fingertip_pose = self.left_fingertip_state[..., :7]
        self.left_fingertip_pos = self.left_fingertip_state[..., :3]
        self.left_fingertip_rot = self.left_fingertip_state[..., 3:7]

        self.right_fingertip_state = self.rigid_body_states[:, self.right_fingertip_handles][..., :13]
        self.right_fingertip_pose = self.right_fingertip_state[..., :7]
        self.right_fingertip_pos = self.right_fingertip_state[..., :3]
        self.right_fingertip_rot = self.right_fingertip_state[..., 3:7]

        self.compute_full_observations()
        self.timestep[:] += 1


    def compute_full_observations(self):
        cnt = 0

        if 'dofps' in self.obs_type:  # dof pos, 44 
            self.obs_buf[:, cnt : cnt + self.num_robot_dofs] = unscale(
                self.robot_dof_pos,
                self.robot_dof_lower_limits,
                self.robot_dof_upper_limits,
            )
            cnt += self.num_robot_dofs

        if 'dofvel' in self.obs_type:  # dof vel, 44
            self.obs_buf[:, cnt : cnt + self.num_robot_dofs] = self.vel_obs_scale * self.robot_dof_vel
            cnt += self.num_robot_dofs

        if 'ftps' in self.obs_type:  # fingertip pos, 3 * 4 * 2
            num_ft_states = len(self.fingertips) * 3
            idxs = (np.arange(len(self.fingertips))[:, None] * 13 + np.array([0, 1, 2])).flatten()
            self.obs_buf[:, cnt : cnt + num_ft_states] = self.left_fingertip_state.reshape(self.num_envs, -1)[...,idxs]
            self.obs_buf[:, cnt + num_ft_states : cnt + 2 * num_ft_states] = self.right_fingertip_state.reshape(self.num_envs, -1)[...,idxs]
            cnt += 2 * num_ft_states

        if 'ftstate' in self.obs_type:  # fingertip state, 13 * 4 * 2
            num_ft_states = len(self.fingertips) * 13
            self.obs_buf[:, cnt : cnt + num_ft_states] = self.left_fingertip_state.reshape(self.num_envs, num_ft_states)
            self.obs_buf[:, cnt + num_ft_states : cnt + 2 * num_ft_states] = self.right_fingertip_state.reshape(self.num_envs, num_ft_states)
            cnt += 2 * num_ft_states

        if 'lastact' in self.obs_type:  # last action, 44
            self.obs_buf[:, cnt : cnt + self.num_actions] = self.actions
            cnt += self.num_actions

        if 'objpose' in self.obs_type:  # object pose, 7 * 2
            obj_dim = 7
            self.obs_buf[:, cnt : cnt + 7] = self.object_pose
            self.obs_buf[:, cnt + 7 : cnt + 14] = self.tool_pose
            cnt += 2 * obj_dim

        if 'objstate' in self.obs_type:  # object state, pose, linvel, angvel. 13 * 2
            obj_dim = 13
            self.obs_buf[:, cnt : cnt + 7] = self.object_pose
            self.obs_buf[:, cnt + 7 : cnt + 10] = self.object_linvel
            self.obs_buf[:, cnt + 10 : cnt + 13] = self.object_angvel
            self.obs_buf[:, cnt + 13 : cnt + 20] = self.tool_pose
            self.obs_buf[:, cnt + 20 : cnt + 23] = self.tool_linvel
            self.obs_buf[:, cnt + 23 : cnt + 26] = self.tool_angvel
            cnt += 2 * obj_dim
        
        if 'palmps' in self.obs_type:  # palm pos, 3 * 2
            obj_dim = 3
            self.obs_buf[:, cnt : cnt + 3] = self.left_palm_pos
            self.obs_buf[:, cnt + 3 : cnt + 6] = self.right_palm_pos
            cnt += 2 * obj_dim

        if 'palmpose' in self.obs_type:  # palm pose, 7 * 2
            obj_dim = 7
            self.obs_buf[:, cnt : cnt + 7] = self.left_palm_pose
            self.obs_buf[:, cnt + 7 : cnt + 14] = self.right_palm_pose
            cnt += 2 * obj_dim

        if 'palmstate' in self.obs_type: # palm state, 13 * 2
            obj_dim = 13
            self.obs_buf[:, cnt : cnt + 13] = self.left_palm_state
            self.obs_buf[:, cnt + 13 : cnt + 26] = self.right_palm_state
            cnt += 2 * obj_dim

        if 'relps' in self.obs_type:  # relative pos to object center, 15 * 2
            self.actual_object_grasp_pos = transformation_apply(self.object_pos, self.object_rot, self.all_object_grasp_pos)
            self.actual_tool_grasp_pos = transformation_apply(self.tool_pos, self.tool_rot, self.all_tool_grasp_pos)
            self.obs_buf[:, cnt : cnt + 3] = self.actual_object_grasp_pos - self.left_palm_pos
            self.obs_buf[:, cnt + 3 : cnt + 15] = (self.actual_object_grasp_pos.unsqueeze(1) - self.left_fingertip_pos).reshape(-1,12)
            self.obs_buf[:, cnt + 15: cnt + 18] = self.actual_tool_grasp_pos - self.right_palm_pos
            self.obs_buf[:, cnt + 18 : cnt + 30] = (self.actual_tool_grasp_pos.unsqueeze(1) - self.right_fingertip_pos).reshape(-1,12)
            cnt += 30

        if 'meshpc' in self.obs_type:  # mesh point cloud, 1024 * 2
            # visualizer = Visualizer3D()
            # visualizer.visualize_point_clouds(self.object_meshpc[0].detach().cpu().numpy())
            # visualizer.visualize_point_clouds(self.tool_meshpc[0].detach().cpu().numpy())
            # visualizer.draw(True)
            if self.control_steps % self.n_resample == 0:  # resample
                self.sampled_object_point_idxs = farthest_point_sample(self.object_mesh_pointclouds, self.num_sample_points, self.device)
                self.sampled_tool_point_idxs = farthest_point_sample(self.tool_mesh_pointclouds, self.num_sample_points, self.device)
            self.object_pointclouds = index_points(self.object_mesh_pointclouds, self.sampled_object_point_idxs, self.device)
            self.tool_pointclouds = index_points(self.tool_mesh_pointclouds, self.sampled_tool_point_idxs, self.device)
            if self.apply_pointcloud_noise:
                self.object_pointclouds += torch.randn_like(self.object_pointclouds).to(self.device) * self.pointcloud_noise_scale * (torch.rand(self.object_pointclouds.shape[:-1]+(1,)) < self.pointcloud_noise_threshold).float().to(self.device)
                self.tool_pointclouds += torch.randn_like(self.tool_pointclouds).to(self.device) * self.pointcloud_noise_scale * (torch.rand(self.tool_pointclouds.shape[:-1]+(1,)) < self.pointcloud_noise_threshold).float().to(self.device)
            for i_task in range(self.num_task):
                self.obs_buf[i_task::self.num_task, cnt : cnt + self.num_pc_flatten] = transformation_apply(self.object_pos[i_task::self.num_task,None,:], self.object_rot[i_task::self.num_task,None,:], self.object_pointclouds[i_task]).view(-1, self.num_pc_flatten)
                self.obs_buf[i_task::self.num_task, cnt + self.num_pc_flatten : cnt + 2 * self.num_pc_flatten] = transformation_apply(self.tool_pos[i_task::self.num_task,None,:], self.tool_rot[i_task::self.num_task,None,:], self.tool_pointclouds[i_task]).view(-1, self.num_pc_flatten)
            cnt += 2 * self.num_pc_flatten
        
        if 'objlabel' in self.obs_type:  
            self.obs_buf[:, cnt] = self.all_object_labels
            self.obs_buf[:, cnt + 1] = self.all_tool_labels
            cnt += 2

        if 'futureps' in self.obs_type:
            future_steps = torch.clip(self.timestep.unsqueeze(-1) + torch.arange(self.Kfuturestep).to(self.device), max=self.dataset_tool_poses.shape[1]).unsqueeze(-1).expand(-1,-1,3)
            self.obs_buf[:, cnt : cnt + self.Kfuturestep * 3] = self.dataset_object_poses[self.all_task_idx].gather(1, future_steps).view(self.num_envs, -1)
            self.obs_buf[:, cnt + self.Kfuturestep * 3 : cnt + 2 * self.Kfuturestep * 3] = self.dataset_tool_poses[self.all_task_idx].gather(1, future_steps).view(self.num_envs, -1)
            cnt += 2 * self.Kfuturestep * 3

        # assert dim
        assert cnt == self.obs_buf.shape[1]

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
        cur_right_pose = self.rigid_body_states.view(self.num_envs, -1, 13)[:,self.right_eef_index + self.num_robot_bodies//2, 0:7]
        # print('target_left_pos:', target_left_pos[0], 'target_right_pos:', target_right_pos[0])
        # print('target_left_rot:', target_left_rot[0], 'target_right_rot:', target_left_rot[0])
        # print('cur_left_pos:', cur_left_pose[0,:3], 'cur_right_pos:', cur_right_pose[0,:3])
        # print('cur_left_rot:', cur_left_pose[0,3:7], 'cur_right_rot:', cur_right_pose[0,3:7])
        # print(self.rigid_body_states.view(self.num_envs, -1, 13)[0,:,:3])

        left_pos_err = target_left_pos - cur_left_pose[:,:3]
        left_rot_err = orientation_error(target_left_rot,cur_left_pose[:,3:7])
        left_delta_qpos = self._control_ik(torch.cat([left_pos_err, left_rot_err], -1).unsqueeze(-1), self.left_j_eef)
        right_pos_err = target_right_pos - cur_right_pose[:,:3]
        right_rot_err = orientation_error(target_right_rot,cur_right_pose[:,3:7])
        right_delta_qpos = self._control_ik(torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1), self.right_j_eef)

        # print(left_pos_err, right_pos_err, left_rot_err, right_rot_err)
        return self.robot_dof_pos[:, self.both_arm_dof_indices] + torch.cat([left_delta_qpos, right_delta_qpos], -1)

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
        pos = self.robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta

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
        self.timestep[env_ids] = 0
        self.stage1_left_successes[env_ids] = 0
        self.stage1_right_successes[env_ids] = 0
        self.stage1_successes[env_ids] = 0
        self.stage1_cul_left_successes[env_ids] = 0
        self.stage1_cul_right_successes[env_ids] = 0
        self.stage2_left_successes[env_ids] = 0
        self.stage2_right_successes[env_ids] = 0
        self.stage2_successes[env_ids] = 0
        self.reach_ref_timestep[env_ids] = -1
        self.left_reach_ref_timestep[env_ids] = -1
        self.right_reach_ref_timestep[env_ids] = -1

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        '''for hand'''
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
        '''for arm'''
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
                        torch.concat(
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
        torch.cuda.empty_cache()
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(mode='s12')

        if self.viewer and self.debug_vis:
            # draw axes to debug
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            object_state = self.root_state_tensor[self.object_indices, :]
            tool_state = self.root_state_tensor[self.tool_indices, :]
            for i in range(self.num_envs):
                self._add_debug_lines(self.envs[i], object_state[i, :3], object_state[i, 3:7])
                self._add_debug_lines(self.envs[i], tool_state[i, :3], tool_state[i, 3:7])
                self._add_debug_lines(self.envs[i], self.left_palm_pos[i], self.left_palm_rot[i])
                self._add_debug_lines(self.envs[i], self.right_palm_pos[i], self.right_palm_rot[i])
                for j in range(len(self.fingertips)):
                    self._add_debug_lines(self.envs[i],self.left_fingertip_pos[i][j],self.left_fingertip_rot[i][j])
                    self._add_debug_lines(self.envs[i],self.right_fingertip_pos[i][j],self.right_fingertip_rot[i][j])

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

    def _control_ik(self, dpose, j_eef):
        damping = 0.1
        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping**2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        return u
