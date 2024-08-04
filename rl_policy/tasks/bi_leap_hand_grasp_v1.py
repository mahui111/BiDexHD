import os, json
import random
import pickle
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
from pprint import pprint
from collections import defaultdict
import matplotlib.pyplot as plt

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


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
def transformation_apply(quat: torch.Tensor, pos: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a transformation to a vector.

    Args:
        quat: Quaternion of the transformation.
        pos: Position of the transformation.
        vec: Vector to transform.

    Returns:
        The transformed vector.
    """
    pos, vec = torch.broadcast_tensors(pos, vec)
    quaternion_shape = pos.shape[:-1] + (4,)
    quat = torch.broadcast_to(quat, quaternion_shape)
    return quat_apply(quat, vec) + pos

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
def compute_grasp_rewards(
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
    actions,
):
    info = {}
    goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
    goal_tool_dist = torch.abs(goal_height - tool_pos[:, 2])
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

    right_fingers_tool_dist = torch.zeros_like(goal_tool_dist)
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
        is_grasp_left == True, 3 * (goal_height - table_height) - 2 * goal_object_dist, lift_object_rew
    )
    lift_tool_rew = torch.zeros_like(goal_tool_dist)
    lift_tool_rew = torch.where(
        is_grasp_right == True, 3 * (goal_height - table_height) - 2 * goal_tool_dist, lift_tool_rew
    )
    # stage 2: lift up reward
    left_hand_up_rew = torch.zeros_like(goal_object_dist)
    left_hand_up_rew = torch.where(is_grasp_left == True, 1 * (left_palm_pos[:, 2] - goal_height), left_hand_up_rew)
    right_hand_up_rew = torch.zeros_like(goal_tool_dist)
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
    right_bonus = torch.zeros_like(goal_tool_dist)
    right_bonus = torch.where(
        is_grasp_right == True,
        torch.where(
            goal_tool_dist <= success_tolerance, 1.0 / (0.5 + goal_tool_dist), right_bonus
        ),
        right_bonus,
    )

    left_approach_penalty = dist_reward_scale * left_fingertips_object_dist + 2 * dist_reward_scale * left_palm_object_dist
    right_approach_penalty = dist_reward_scale * right_fingers_tool_dist + 2 * dist_reward_scale * right_palm_object_dist
    left_after_grasp_reward = lift_object_rew + left_hand_up_rew + left_bonus
    right_after_grasp_reward = lift_tool_rew + right_hand_up_rew + right_bonus
    object_offset_penalty = 0.3 * object_offset
    tool_offset_penalty = 0.3 * tool_offset

    # total reward
    left_reward = - left_approach_penalty + left_after_grasp_reward - object_offset_penalty# - left_action_penalty
    right_reward = - right_approach_penalty + right_after_grasp_reward - tool_offset_penalty# - right_action_penalty
    reward = left_reward + right_reward

    # level 1
    info["left/is_grasp"] = is_grasp_left
    info["left/fingertips_object_dist"] = left_fingertips_object_dist
    info["left/palm_object_dist"] = left_palm_object_dist
    info["left/lift_object_rew"] = lift_object_rew
    info["left/hand_up_rew"] = left_hand_up_rew
    info["left/bonus"] = left_bonus
    info["left/object_offset"] = object_offset

    info["right/is_grasp"] = is_grasp_right
    info["right/fingers_tool_dist"] = right_fingers_tool_dist
    info["right/palm_tool_dist"] = right_palm_object_dist
    info["right/lift_tool_rew"] = lift_tool_rew
    info["right/hand_up_rew"] = right_hand_up_rew
    info["right/bonus"] = right_bonus
    info["right/tool_offset"] = tool_offset
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
    successes = torch.where(
        torch.logical_and(goal_object_dist <= success_tolerance, goal_tool_dist <= success_tolerance),
        torch.where(
            torch.logical_and(
                left_fingertips_object_dist + left_palm_object_dist < 0.12 * (num_fingers + 1), 
                right_fingers_tool_dist + right_palm_object_dist < 0.12 * (num_fingers + 1)
            ),
            torch.ones_like(successes),
            successes,
        ),
        torch.zeros_like(successes),
    )
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


@torch.jit.script
def compute_bvdex_stage1_rewards(
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    consecutive_successes,
    max_episode_length: float,
    object_pose, tool_pose,
    left_palm_pose, right_palm_pose,
    left_fingertip_pos, right_fingertip_pos,
    dist_reward_scale: float,
    action_penalty_scale: float,
    success_tolerance: float,
    av_factor: float,
    table_height: float,
    actions,
    ref_object_pose, ref_init_object_pos_dist, ref_init_object_hand_rot_diff,
    ref_tool_pose, ref_init_tool_pos_dist, ref_init_tool_hand_rot_diff,
    object_hand_joint_rot_diff: int,
):
    object_pos = object_pose[:, :3]
    tool_pos = tool_pose[:, :3]
    info = {}

    left_palm_object_dist = torch.norm(object_pos - left_palm_pose[:, :3], dim=-1)
    left_palm_object_dist = torch.where(left_palm_object_dist >= 0.5, 0.5, left_palm_object_dist)
    right_palm_object_dist = torch.norm(tool_pos - right_palm_pose[:, :3], dim=-1)
    right_palm_object_dist = torch.where(right_palm_object_dist >= 0.5, 0.5, right_palm_object_dist)

    num_fingers = left_fingertip_pos.shape[1]
    left_fingertips_object_dist = torch.zeros_like(left_palm_object_dist)
    for i in range(num_fingers):
        left_fingertips_object_dist += torch.norm(
            left_fingertip_pos[:, i, :] - object_pos, dim=-1
        )
    left_fingertips_object_dist = torch.where(
        left_fingertips_object_dist >= 3.0, 3.0, left_fingertips_object_dist
    )

    right_fingers_tool_dist = torch.zeros_like(right_palm_object_dist)
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
    ref_object_pos_dist = torch.norm(ref_object_pose[:, :3] - object_pos, dim=-1)
    ref_object_rot_rew = quat_rew(ref_object_pose[:, 3:7].repeat(len(object_pose),1), object_pose[:, 3:7]) # [-1,1]
    left_lift_object_pos_rew, left_lift_object_rot_rew = torch.zeros_like(ref_object_pos_dist), torch.zeros_like(ref_object_rot_rew)
    left_lift_object_pos_rew = torch.where(is_grasp_left == True, 1 - ref_object_pos_dist / ref_init_object_pos_dist, left_lift_object_pos_rew)
    left_lift_object_rot_rew = torch.where(is_grasp_left == True, ref_object_rot_rew, left_lift_object_rot_rew)

    ref_tool_pos_dist = torch.norm(ref_tool_pose[:, :3] - tool_pos, dim=-1)
    ref_tool_rot_rew = quat_rew(ref_tool_pose[:, 3:7].repeat(len(tool_pose),1), tool_pose[:, 3:7])  # [-1,1]
    right_lift_tool_pos_rew, right_lift_tool_rot_rew = torch.zeros_like(ref_tool_pos_dist), torch.zeros_like(ref_tool_rot_rew)
    right_lift_tool_pos_rew = torch.where(is_grasp_right == True, 1 - ref_tool_pos_dist / ref_init_tool_pos_dist, right_lift_tool_pos_rew)
    right_lift_tool_rot_rew = torch.where(is_grasp_right == True, ref_tool_rot_rew, right_lift_tool_rot_rew)

    # stage 2: hand-object joint rotation, no grasp condition
    if object_hand_joint_rot_diff:
        object_hand_rot_diff = quat_diff_theta(object_pose[:, 3:7], left_palm_pose[:, 3:7])
        left_object_hand_rot_rew = - torch.abs((ref_init_object_hand_rot_diff - object_hand_rot_diff))  # [0, -2pi]
        left_object_hand_rot_rew = 0.5 + left_object_hand_rot_rew * (0.5 / torch.pi)  # [0.5, -0.5]
        tool_hand_rot_diff = quat_diff_theta(tool_pose[:, 3:7], right_palm_pose[:, 3:7])
        right_tool_hand_rot_rew = - torch.abs((ref_init_tool_hand_rot_diff - tool_hand_rot_diff))  # [0, -2pi]
        right_tool_hand_rot_rew = 0.5 + right_tool_hand_rot_rew * (0.5 / torch.pi)  # [0.5, -0.5]
    else:
        left_object_hand_rot_rew = torch.zeros_like(ref_object_pos_dist)
        right_tool_hand_rot_rew = torch.zeros_like(ref_tool_pos_dist)

    # stage 3: lift near goal bonus
    left_bonus = torch.zeros_like(ref_object_pos_dist)
    left_bonus = torch.where(
        is_grasp_left == True,
        torch.where(
            ref_object_pos_dist <= success_tolerance, 1.0 / (1 + ref_object_pos_dist), left_bonus
        ),
        left_bonus,
    )
    right_bonus = torch.zeros_like(ref_object_pos_dist)
    right_bonus = torch.where(
        is_grasp_right == True,
        torch.where(
            ref_tool_pos_dist <= success_tolerance, 1.0 / (1 + ref_tool_pos_dist), right_bonus
        ),
        right_bonus,
    )

    # additional (x,y) offset
    object_offset = torch.norm(object_pos[:, 0:2] - ref_object_pose[:, 0:2], dim=-1)
    tool_offset = torch.norm(tool_pos[:, 0:2] - ref_tool_pose[:, 0:2], dim=-1)
    object_offset_penalty = 0.3 * object_offset
    tool_offset_penalty = 0.3 * tool_offset

    # total reward
    left_approach_penalty = dist_reward_scale * left_fingertips_object_dist + 2 * dist_reward_scale * left_palm_object_dist
    right_approach_penalty = dist_reward_scale * right_fingers_tool_dist + 2 * dist_reward_scale * right_palm_object_dist
    left_lift_to_refpose_reward = left_lift_object_pos_rew + left_lift_object_rot_rew * 0.2
    right_lift_to_refpose_reward = right_lift_tool_pos_rew + right_lift_tool_rot_rew  * 0.2

    left_reward = - left_approach_penalty + left_lift_to_refpose_reward + left_object_hand_rot_rew * 0.3 + left_bonus   # - object_offset_penalty
    right_reward = - right_approach_penalty + right_lift_to_refpose_reward + right_tool_hand_rot_rew * 0.3 + right_bonus  # - tool_offset_penalty
    reward = left_reward + right_reward



    # level 1
    info["left/is_grasp"] = is_grasp_left
    info["left/fingertips_object_dist"] = left_fingertips_object_dist
    info["left/palm_object_dist"] = left_palm_object_dist
    info["left/lift_object_pos_rew"] = left_lift_object_pos_rew
    info["left/lift_object_rot_rew"] = left_lift_object_rot_rew
    info["left/object_hand_rot_rew"] = left_object_hand_rot_rew
    info["left/bonus"] = left_bonus
    info["left/object_offset"] = object_offset

    info["right/is_grasp"] = is_grasp_right
    info["right/fingers_tool_dist"] = right_fingers_tool_dist
    info["right/palm_tool_dist"] = right_palm_object_dist
    info["right/lift_tool_pos_rew"] = right_lift_tool_pos_rew
    info["right/lift_tool_rot_rew"] = right_lift_tool_rot_rew
    info["right/tool_hand_rot_rew"] = right_tool_hand_rot_rew
    info["right/bonus"] = right_bonus
    info["right/tool_offset"] = tool_offset

    # level 2
    info["left/left_approach_penalty"] = left_approach_penalty
    info["left/lift_to_refpos_reward"] = left_lift_to_refpose_reward

    info["right/right_approach_penalty"] = right_approach_penalty
    info["right/lift_to_refpos_reward"] = right_lift_to_refpose_reward
    
    # level 3
    info["left/reward"] = left_reward
    info["right/reward"] = right_reward
    info["reward"] = reward

    resets = reset_buf.clone()
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    resets = torch.where(torch.logical_or(object_pos[:, 2] <= table_height, tool_pos[:, 2] <= table_height), torch.ones_like(resets), resets)
    successes = torch.where(
        torch.logical_and(ref_object_pos_dist <= success_tolerance, ref_tool_pos_dist <= success_tolerance),
        torch.where(
            torch.logical_and(
                left_fingertips_object_dist + left_palm_object_dist < 0.12 * (num_fingers + 1), 
                right_fingers_tool_dist + right_palm_object_dist < 0.12 * (num_fingers + 1)
            ),
            torch.ones_like(successes),
            successes,
        ),
        torch.zeros_like(successes),
    )
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    current_successes = torch.where(resets==True, successes, current_successes)
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


@torch.jit.script
def compute_bvdex_stage12_rewards(
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    consecutive_successes,
    max_episode_length: float,
    object_pose, tool_pose,
    left_palm_pose, right_palm_pose,
    left_fingertip_pos, right_fingertip_pos,
    dist_reward_scale: float,
    action_penalty_scale: float,
    success_tolerance: float,
    av_factor: float,
    table_height: float,
    actions,
    timestep, left_reach_ref_timestep, right_reach_ref_timestep,
    ref_object_pose, ref_init_object_pos_dist, ref_init_object_hand_pos_diff, ref_init_object_hand_rot_diff, ref_init_left_fingers_palm_pose,
    ref_tool_pose, ref_init_tool_pos_dist, ref_init_tool_hand_pos_diff, ref_init_tool_hand_rot_diff, ref_init_right_fingers_palm_pose,
    is_stage1_min_rew: int, is_stage1_lin_rew:int, is_stage2_pos_rew_exp: int,
):
    '''
    stage 1: reach a static ref object pose (linear reward)
    stage 2: stage 1 finished, follow a dynamic ref object pose (exponential reward)   
    '''
    info = {}

    left_palm_object_dist = torch.norm(object_pose[:, :3] - left_palm_pose[:, :3], dim=-1)  # ref_init_left_fingers_palm_pose[:,-1,:3]
    left_palm_object_dist = torch.where(left_palm_object_dist >= 0.5, 0.5, left_palm_object_dist)
    right_palm_object_dist = torch.norm(tool_pose[:, :3] - right_palm_pose[:, :3], dim=-1)  #ref_init_right_fingers_palm_pose[:,-1,:3]
    right_palm_object_dist = torch.where(right_palm_object_dist >= 0.5, 0.5, right_palm_object_dist)

    num_fingers = left_fingertip_pos.shape[1]
    left_fingertips_object_dist = torch.zeros_like(left_palm_object_dist)
    for i in range(num_fingers):
        left_fingertips_object_dist += torch.norm(
            left_fingertip_pos[:, i, :] - object_pose[:, :3], dim=-1  # ref_init_left_fingers_palm_pose[:,i,:3]
        )
    left_fingertips_object_dist = torch.where(
        left_fingertips_object_dist >= 3.0, 3.0, left_fingertips_object_dist
    )

    right_fingers_tool_dist = torch.zeros_like(right_palm_object_dist)
    for i in range(num_fingers):
        right_fingers_tool_dist += torch.norm(
            right_fingertip_pos[:, i, :] - tool_pose[:, :3], dim=-1  # ref_init_right_fingers_palm_pose[:,i,:3]
        )
    right_fingers_tool_dist = torch.where(
        right_fingers_tool_dist >= 3.0, 3.0, right_fingers_tool_dist
    )

    is_grasp_left = ((left_fingertips_object_dist <= 0.12 * num_fingers) + (left_palm_object_dist <= 0.12)).float()
    is_grasp_right = ((right_fingers_tool_dist <= 0.12 * num_fingers) + (right_palm_object_dist <= 0.12)).float()

    # useless above, start: hand object relative pose to encourage getting close to object and keep certain pose
    left_object_pos_wrt_hand, left_object_ori_wrt_hand = compute_relative_pose(
        object_pose[:, :3], object_pose[:, 3:7], left_palm_pose[:, :3], left_palm_pose[:, 3:7]
    )
    left_object_hand_pos_dist = F.pairwise_distance(left_object_pos_wrt_hand, ref_init_object_hand_pos_diff)
    left_object_hand_rot_dist = quat_diff_rad(left_object_ori_wrt_hand, ref_init_object_hand_rot_diff).abs()
    right_tool_pos_wrt_hand, right_tool_ori_wrt_hand = compute_relative_pose(
        tool_pose[:, :3], tool_pose[:, 3:7], right_palm_pose[:, :3], right_palm_pose[:, 3:7]
    )
    right_tool_hand_pos_dist = F.pairwise_distance(right_tool_pos_wrt_hand, ref_init_tool_hand_pos_diff)
    right_tool_hand_rot_dist = quat_diff_rad(right_tool_ori_wrt_hand, ref_init_tool_hand_rot_diff).abs()

    left_ready_grasp = (left_object_hand_pos_dist <= 0.15) * (left_object_hand_rot_dist <= 0.3)
    right_ready_grasp = (right_tool_hand_pos_dist <= 0.15) * (right_tool_hand_rot_dist <= 0.3)

    # stage 1: after hand approach object, lift_object
    ref_object_pos_dist = torch.norm(ref_object_pose[:, :3] - object_pose[:, :3], dim=-1)
    ref_object_rot_rew = quat_rew(ref_object_pose[:, 3:7], object_pose[:, 3:7]) # [-1,1]
    ref_tool_pos_dist = torch.norm(ref_tool_pose[:, :3] - tool_pose[:, :3], dim=-1)
    ref_tool_rot_rew = quat_rew(ref_tool_pose[:, 3:7], tool_pose[:, 3:7])  # [-1,1]
    '''lift object reward for stage 1'''
    left_lift_object_pos_rew1 = (1 - ref_object_pos_dist / ref_init_object_pos_dist)  # [0, 1]
    left_lift_object_rot_rew1 = ref_object_rot_rew
    right_lift_tool_pos_rew1 = (1 - ref_tool_pos_dist / ref_init_tool_pos_dist)  # [0, 1]
    right_lift_tool_rot_rew1 = ref_tool_rot_rew
    '''lift object reward for stage 2'''
    # TODO: add stage 2 reward
    left_lift_object_pos_rew2 = torch.exp(-15 * ref_object_pos_dist) if is_stage2_pos_rew_exp else left_lift_object_pos_rew1
    left_lift_object_rot_rew2 = ref_object_rot_rew
    right_lift_tool_pos_rew2 = torch.exp(-15 * ref_tool_pos_dist) if is_stage2_pos_rew_exp else right_lift_tool_pos_rew1
    right_lift_tool_rot_rew2 = ref_tool_rot_rew
    '''lift object reward'''
    left_lift_object_pos_rew = torch.where(
        left_ready_grasp > 0,
        torch.where(left_reach_ref_timestep == -1, left_lift_object_pos_rew1, left_lift_object_pos_rew2),
        torch.zeros_like(ref_object_pos_dist),
    )
    left_lift_object_rot_rew = torch.where(
        left_ready_grasp > 0,
        torch.where(left_reach_ref_timestep == -1, left_lift_object_rot_rew1, left_lift_object_rot_rew2),
        torch.zeros_like(ref_object_pos_dist),
    )
    right_lift_tool_pos_rew = torch.where(
        right_ready_grasp > 0,
        torch.where(right_reach_ref_timestep == -1, right_lift_tool_pos_rew1, right_lift_tool_pos_rew2),
        torch.zeros_like(ref_tool_pos_dist),
    )
    right_lift_tool_rot_rew = torch.where(
        right_ready_grasp > 0,
        torch.where(right_reach_ref_timestep == -1, right_lift_tool_rot_rew1, right_lift_tool_rot_rew2),
        torch.zeros_like(ref_tool_pos_dist),
    )
    
    # stage 2: hand-object joint pose difference reward only in stage 1, notice no grasp condition 
    '''
    object_hand_rot_diff1 = quat_diff_theta(object_pose[:, 3:7], left_palm_pose[:, 3:7])
    left_object_hand_rot_rew1 = - torch.abs((ref_init_object_hand_rot_diff - object_hand_rot_diff1))  # [0, -2pi]
    left_object_hand_rot_rew1 = 0.5 + left_object_hand_rot_rew1 * (0.5 / torch.pi)  # [0.5, -0.5]
    left_object_hand_rot_rew = torch.where(left_reach_ref_timestep == -1, left_object_hand_rot_rew1, torch.zeros_like(ref_object_pos_dist))

    tool_hand_rot_diff1 = quat_diff_theta(tool_pose[:, 3:7], right_palm_pose[:, 3:7])
    right_tool_hand_rot_rew1 = - torch.abs((ref_init_tool_hand_rot_diff - tool_hand_rot_diff1))  # [0, -2pi]
    right_tool_hand_rot_rew1 = 0.5 + right_tool_hand_rot_rew1 * (0.5 / torch.pi)  # [0.5, -0.5]
    right_tool_hand_rot_rew = torch.where(right_reach_ref_timestep == -1, right_tool_hand_rot_rew1, torch.zeros_like(ref_tool_pos_dist))

    object_hand_pos_diff1 = object_pos - left_palm_pose[:, :3]
    # left_object_hand_pos_rew1 = - torch.abs(torch.norm(ref_init_object_hand_pos_diff,dim=-1) - torch.norm(object_hand_pos_diff1,dim=-1)) + 0.3 * F.cosine_similarity(ref_init_object_hand_pos_diff, object_hand_pos_diff1, dim=-1) 
    left_object_hand_pos_rew1 = - torch.norm(ref_init_object_hand_pos_diff-object_hand_pos_diff1,dim=-1)
    left_object_hand_pos_rew = torch.where(right_reach_ref_timestep == -1, left_object_hand_pos_rew1, torch.zeros_like(ref_object_pos_dist))

    tool_hand_pos_diff1 = tool_pos - right_palm_pose[:, :3]
    # right_tool_hand_pos_rew1 = - torch.abs(torch.norm(ref_init_tool_hand_pos_diff,dim=-1) - torch.norm(tool_hand_pos_diff1,dim=-1)) + 0.3 * F.cosine_similarity(ref_init_tool_hand_pos_diff, tool_hand_pos_diff1, dim=-1)
    right_tool_hand_pos_rew1 = - torch.norm(ref_init_tool_hand_pos_diff-tool_hand_pos_diff1,dim=-1)
    right_tool_hand_pos_rew = torch.where(right_reach_ref_timestep == -1, right_tool_hand_pos_rew1, torch.zeros_like(ref_tool_pos_dist))

    left_object_hand_posediff_rew = 0.3 * (left_object_hand_rot_rew + left_object_hand_pos_rew)
    right_tool_hand_posediff_rew = 0.3 * (right_tool_hand_rot_rew + right_tool_hand_pos_rew)
    '''
    if not is_stage1_lin_rew: 
        trans_scale, rot_eps = 3.5, 0.4
        # left_pos_idx = (rot_eps / trans_scale) / torch.max(left_object_hand_pos_dist, torch.tensor(rot_eps / trans_scale).to(actions.device))
        left_object_hand_pos_rew = 1.0 / (trans_scale * torch.abs(left_object_hand_pos_dist) + rot_eps)# * left_pos_idx
        # left_rot_idx = rot_eps / torch.max(left_object_hand_rot_dist, torch.tensor(rot_eps).to(actions.device))
        left_object_hand_rot_rew = 1.0 / (left_object_hand_rot_dist + rot_eps)# * left_rot_idx
        if is_stage1_min_rew: 
            left_object_hand_posediff_rew = 0.2 * torch.minimum(left_object_hand_pos_rew, left_object_hand_rot_rew)
        else:
            left_object_hand_posediff_rew = 0.1 * (left_object_hand_pos_rew + left_object_hand_rot_rew)

        # right_pos_idx = (rot_eps / trans_scale) / torch.max(right_tool_hand_pos_dist, torch.tensor(rot_eps / trans_scale).to(actions.device))
        right_tool_hand_pos_rew = 1.0 / (trans_scale * torch.abs(right_tool_hand_pos_dist) + rot_eps)#  * right_pos_idx
        # right_rot_idx = rot_eps / torch.max(right_tool_hand_rot_dist, torch.tensor(rot_eps).to(actions.device))
        right_tool_hand_rot_rew = 1.0 / (right_tool_hand_rot_dist + rot_eps)#  * right_rot_idx
        if is_stage1_min_rew: 
            right_tool_hand_posediff_rew = 0.2 * torch.minimum(right_tool_hand_pos_rew, right_tool_hand_rot_rew) 
        else:
            right_tool_hand_posediff_rew = 0.1 * (right_tool_hand_pos_rew + right_tool_hand_rot_rew)
    else:  # linear reward
        left_object_hand_pos_rew = - left_object_hand_pos_dist
        left_object_hand_rot_rew = - 0.33 * left_object_hand_rot_dist
        right_tool_hand_pos_rew = - right_tool_hand_pos_dist
        right_tool_hand_rot_rew = - 0.33 * right_tool_hand_rot_dist
        if is_stage1_min_rew: 
            left_object_hand_posediff_rew = torch.minimum(left_object_hand_pos_rew, left_object_hand_rot_rew)
            right_tool_hand_posediff_rew = torch.minimum(right_tool_hand_pos_rew, right_tool_hand_rot_rew)
        else:
            left_object_hand_posediff_rew = (left_object_hand_pos_rew + left_object_hand_rot_rew) / 2
            right_tool_hand_posediff_rew = (right_tool_hand_pos_rew + right_tool_hand_rot_rew) / 2
    info["left/object_hand_pos_rew"] = left_object_hand_pos_rew
    info["left/object_hand_rot_rew"] = left_object_hand_rot_rew
    info["right/tool_hand_pos_rew"] = right_tool_hand_pos_rew
    info["right/tool_hand_rot_rew"] = right_tool_hand_rot_rew

    # stage 3: lift near goal bonus
    left_bonus = torch.zeros_like(ref_object_pos_dist)
    left_bonus = torch.where(
        left_ready_grasp > 0,
        torch.where(
            ref_object_pos_dist <= success_tolerance, 1.0 / (1 + ref_object_pos_dist), left_bonus
        ),
        left_bonus,
    )
    right_bonus = torch.zeros_like(ref_object_pos_dist)
    right_bonus = torch.where(
        right_ready_grasp > 0,
        torch.where(
            ref_tool_pos_dist <= success_tolerance, 1.0 / (1 + ref_tool_pos_dist), right_bonus
        ),
        right_bonus,
    )
    # stage 4: trajectory following
    left_successes = torch.logical_and(ref_object_pos_dist <= success_tolerance, left_ready_grasp).float()
    left_reach_ref_timestep = torch.where(torch.logical_and(left_successes == 1, left_reach_ref_timestep == -1), timestep, left_reach_ref_timestep)
    
    right_successes = torch.logical_and(ref_tool_pos_dist <= success_tolerance, right_ready_grasp).float()
    right_reach_ref_timestep = torch.where(torch.logical_and(right_successes == 1, right_reach_ref_timestep == -1), timestep, right_reach_ref_timestep)

    # additional (x,y) offset
    object_offset = torch.norm(object_pose[:, 0:2] - ref_object_pose[:, 0:2], dim=-1)
    tool_offset = torch.norm(tool_pose[:, 0:2] - ref_tool_pose[:, 0:2], dim=-1)
    object_offset_penalty = 0.3 * object_offset
    tool_offset_penalty = 0.3 * tool_offset

    # total reward
    left_approach_penalty = dist_reward_scale * left_fingertips_object_dist + 2 * dist_reward_scale * left_palm_object_dist
    right_approach_penalty = dist_reward_scale * right_fingers_tool_dist + 2 * dist_reward_scale * right_palm_object_dist
    left_lift_to_refpose_reward = 2 * left_lift_object_pos_rew + 0.2 * left_lift_object_rot_rew
    right_lift_to_refpose_reward = 2 * right_lift_tool_pos_rew + 0.2 * right_lift_tool_rot_rew 


    left_reward = left_object_hand_posediff_rew + left_lift_to_refpose_reward + left_bonus   # - left_approach_penalty +  - object_offset_penalty
    right_reward = right_tool_hand_posediff_rew + right_lift_to_refpose_reward + right_bonus  # - right_approach_penalty +  - tool_offset_penalty
    reward = left_reward + right_reward

    # breakpoint()

    # level 1
    info["left/successes"] = left_successes
    info["left/object_hand_pos_dist"] = left_object_hand_pos_dist
    info["left/object_hand_rot_dist"] = left_object_hand_rot_dist
    info["left/is_ready_grasp"] = left_ready_grasp
    info["left/is_grasp"] = is_grasp_left
    info["left/fingertips_object_dist"] = left_fingertips_object_dist
    info["left/palm_object_dist"] = left_palm_object_dist
    info["left/lift_object_pos_rew"] = left_lift_object_pos_rew
    info["left/lift_object_rot_rew"] = left_lift_object_rot_rew
    info["left/bonus"] = left_bonus
    info["left/object_offset"] = object_offset
    info["left/ref_object_pos_dist"] = ref_object_pos_dist

    info["right/successes"] = right_successes
    info["right/tool_hand_pos_dist"] = right_tool_hand_pos_dist
    info["right/tool_hand_rot_dist"] = right_tool_hand_rot_dist
    info["right/is_ready_grasp"] = right_ready_grasp
    info["right/is_grasp"] = is_grasp_right
    info["right/fingers_tool_dist"] = right_fingers_tool_dist
    info["right/palm_tool_dist"] = right_palm_object_dist
    info["right/lift_tool_pos_rew"] = right_lift_tool_pos_rew
    info["right/lift_tool_rot_rew"] = right_lift_tool_rot_rew
    info["right/bonus"] = right_bonus
    info["right/tool_offset"] = tool_offset
    info["right/ref_tool_pos_dist"] = ref_tool_pos_dist

    # level 2
    info["left/left_approach_penalty"] = left_approach_penalty
    info["left/lift_to_refpose_reward"] = left_lift_to_refpose_reward
    info["left/object_hand_posediff_rew"] = left_object_hand_posediff_rew

    info["right/right_approach_penalty"] = right_approach_penalty
    info["right/lift_to_refpose_reward"] = right_lift_to_refpose_reward
    info["right/tool_hand_posediff_rew"] = right_tool_hand_posediff_rew
    
    # level 3
    info["left/reward"] = left_reward
    info["right/reward"] = right_reward
    info["reward"] = reward

    resets = reset_buf.clone()
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    resets = torch.where(torch.logical_or(object_pose[:, 2] <= table_height, tool_pose[:, 2] <= table_height), torch.ones_like(resets), resets)
    successes = torch.where(
        torch.logical_and(ref_object_pos_dist <= success_tolerance, ref_tool_pos_dist <= success_tolerance),
        torch.where(
            torch.logical_and(
                left_fingertips_object_dist + left_palm_object_dist < 0.12 * (num_fingers + 1), 
                right_fingers_tool_dist + right_palm_object_dist < 0.12 * (num_fingers + 1)
            ),
            torch.ones_like(successes),
            successes,
        ),
        torch.zeros_like(successes),
    )
    # print(f'timestep:{self.timestep[0]} | left_approach_dist:{(left_fingertips_object_dist + left_palm_object_dist)[0]:.3f} | right_approach_dist:{(right_fingers_tool_dist + right_palm_object_dist)[0]:.3f} | ref_object_pos_dist:{ref_object_pos_dist[0]:.3f} | ref_tool_pos_dist:{ref_tool_pos_dist[0]:.3f}')
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    current_successes = torch.where(resets == 1, successes, current_successes)
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
        timestep, left_reach_ref_timestep, right_reach_ref_timestep,
        info,
    )



class BiLeapHandGraspV1(VecTask):
    def get_obs_idx(self,):
        cnt = 0
        lidx, ridx = [], []

        if 'dofps' in self.obs_type:  # dof pos, 44 
            num_robot_dofs = 44
            lidx.extend(list(range(cnt, cnt + num_robot_dofs//2)))
            ridx.extend(list(range(cnt + num_robot_dofs//2, cnt + num_robot_dofs)))
            cnt += num_robot_dofs

        if 'dofvel' in self.obs_type:  # dof vel, 44
            num_robot_dofs = 44
            lidx.extend(list(range(cnt, cnt + num_robot_dofs//2)))
            ridx.extend(list(range(cnt + num_robot_dofs//2, cnt + num_robot_dofs)))
            cnt += num_robot_dofs

        if 'ftps' in self.obs_type:  # fingertip pos, 3 * 4 * 2
            num_ft_states = 4 * 3
            lidx.extend(list(range(cnt, cnt + num_ft_states)))
            ridx.extend(list(range(cnt + num_ft_states, cnt + 2 * num_ft_states)))
            cnt += 2 * num_ft_states

        if 'ftstate' in self.obs_type:  # fingertip state, 13 * 4 * 2
            num_ft_states = 4 * 13
            lidx.extend(list(range(cnt, cnt + num_ft_states)))
            ridx.extend(list(range(cnt + num_ft_states, cnt + 2 * num_ft_states)))
            cnt += 2 * num_ft_states

        if 'lastact' in self.obs_type:  # last action, 44
            num_actions = 44
            lidx.extend(list(range(cnt, cnt + num_actions//2)))
            ridx.extend(list(range(cnt + num_actions//2, cnt + num_actions)))
            cnt += num_actions

        if 'objpose' in self.obs_type:  # object pose, 7 * 2
            obj_dim = 7
            lidx.extend(list(range(cnt, cnt + obj_dim)))
            ridx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim

        if 'objstate' in self.obs_type:  # object state, pose, linvel, angvel. 13 * 2
            obj_dim = 13
            lidx.extend(list(range(cnt, cnt + obj_dim)))
            ridx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim
        
        if 'palmps' in self.obs_type:  # palm pos, 3 * 2
            obj_dim = 3
            lidx.extend(list(range(cnt, cnt + obj_dim)))
            ridx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim

        if 'palmpose' in self.obs_type:  # palm pose, 7 * 2
            obj_dim = 7
            lidx.extend(list(range(cnt, cnt + obj_dim)))
            ridx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim

        if 'palmstate' in self.obs_type: # palm state, 13 * 2
            obj_dim = 13
            lidx.extend(list(range(cnt, cnt + obj_dim)))
            ridx.extend(list(range(cnt + obj_dim, cnt + 2 * obj_dim)))
            cnt += 2 * obj_dim

        if 'relps' in self.obs_type:  # relative pos to object center, 15 * 2
            relpos_dim = 3 * (4 + 1)
            lidx.extend(list(range(cnt, cnt + relpos_dim)))
            ridx.extend(list(range(cnt + relpos_dim, cnt + 2 * relpos_dim)))
            cnt += 2 * relpos_dim

        return lidx, ridx

    def get_obs_num(self,):
        lidx, ridx = self.get_obs_idx()
        return len(lidx) + len(ridx)

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
        self.mode = self.cfg["mode"]
        self.frequency, self.horizon = self.cfg["task"]['frequency'], self.cfg["task"]['horizon']
        self.is_stage1_min_rew = self.cfg["task"]["isStage1MinReward"]
        self.is_stage1_lin_rew = self.cfg["task"]["isStage1LinReward"]
        self.is_stage2_pos_rew_exp = self.cfg["task"]["isStage2PosRewExp"]

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
        
        # self.palm_offset = self.cfg["env"]["palm_offset"]
        # self.fingertip_offset = self.cfg["env"]["finger_offset"]
        # self.thumb_offset = self.cfg["env"]["thumb_offset"]
        
        self.obs_type = self.cfg["env"]["observationType"]

        assert self.arm_controller in ["ik", "qpos"]

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        self.cfg["env"]["numObservations"] = self.get_obs_num()
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
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # customize
        self.timestep = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.left_reach_ref_timestep = - torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        self.right_reach_ref_timestep = - torch.ones(self.num_envs, dtype=torch.int32, device=self.device)

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
        self._prepare_dataset()  
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)),)

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
        print(f'Creating URDFs for object {target_id} and tool {tool_id}')
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
        # self.robot_dof_default_pos = to_torch(
        #     [ 2.6396e-01,  2.5209e-01,  2.1412e+00,  7.3334e-01, -1.3495e+00,
        #  -1.2485e+00,  1.4559e-02, -2.9828e-01,  9.1046e-01,  8.6533e-01,
        #  -1.1081e+00,  1.5081e-02,  1.3430e+00, -5.7619e-03,  4.8028e-01,
        #   1.0533e-01,  1.1771e-02,  2.3770e-03,  5.2847e-01,  2.7517e-01,
        #   9.2905e-03,  1.1253e-03,  8.7288e-01,  2.2177e+00, -1.1985e+00,
        #  -1.6249e+00,  6.2858e-01, -8.3031e-01,  3.1325e-01,  6.1444e-02,
        #   2.2033e-02, -5.7871e-03,  5.2441e-01,  3.1776e-03,  1.3648e+00,
        #   4.5659e-03,  4.3381e-01,  2.8884e-02,  3.6321e-02,  9.6377e-04,
        #   3.9599e-01, -1.1753e-01,  2.0954e-01,  1.3920e-01], 
        # device=self.device)
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
        
        # object
        self._prepare_task(task_id=self.task_id)
        object_asset, tool_asset = self._prepare_object_tool_pair(asset_root)
        
        # object_asset = self._prepare_object_asset(asset_root, self.cfg["env"]["asset"]["objectAssetFile"])
        # tool_asset = self._prepare_object_asset(asset_root, self.cfg["env"]["asset"]["toolAssetFile"])

        # get object asset info
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset) + self.gym.get_asset_rigid_body_count(tool_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset) + self.gym.get_asset_rigid_shape_count(tool_asset)
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset) + self.gym.get_asset_dof_count(tool_asset)

        # table
        table_asset, self.table_start_pose, side_panel_asset, side_panel_start_pose = self._prepare_table_asset()
        self.table_height = self.table_start_pose.p.z*2
        self.goal_height = self.table_height + 0.3

        # initialize pose
        object_center = (self.dataset_object_init_pos + self.dataset_tool_init_pos) / 2
        left_robot_start_pose = gymapi.Transform()
        left_robot_start_pose.p = gymapi.Vec3(object_center[0] - 0.34, object_center[1] - 0.5, self.table_height + 0.52)
        left_robot_start_pose.r = gymapi.Quat(0.5,  0.5,  0.5, -0.5)#0,1/np.sqrt(2),0,-1/np.sqrt(2)
        right_robot_start_pose = gymapi.Transform()
        right_robot_start_pose.p = gymapi.Vec3(object_center[0] + 0.34, object_center[1] - 0.5, self.table_height + 0.52)
        right_robot_start_pose.r = gymapi.Quat(0.5,  0.5,  0.5, -0.5)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(*self.dataset_object_init_pos)
        object_start_pose.r = gymapi.Quat(*self.dataset_object_init_quat)
        tool_start_pose = gymapi.Transform()
        tool_start_pose.p = gymapi.Vec3(*self.dataset_tool_init_pos)
        tool_start_pose.r = gymapi.Quat(*self.dataset_tool_init_quat)


        self.envs, self.cameras = [], []
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

            # add camera
            if self.cfg['env']['enableCameraSensors']:
                camera_props = gymapi.CameraProperties()
                camera_props.width = self.cfg['env']['imageWidth']
                camera_props.height = self.cfg['env']['imageHeight']
                camera_ptr = self.gym.create_camera_sensor(env_ptr, camera_props)
                self.gym.set_camera_location(camera_ptr, env_ptr, gymapi.Vec3(*self.cfg['env']['cameraPosition']), gymapi.Vec3(*self.cfg['env']['cameraTarget']))
                self.cameras.append(camera_ptr)

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

        # calculate static 
        self.ref_init_object_pos_dist = torch.norm(self.ref_object_pose[:, :3] - self.object_init_states[:, :3], dim=-1)    # (1,)
        self.ref_init_object_rot_diff = quat_diff_theta(self.ref_object_pose[:, 3:7], self.object_init_states[:1, 3:7])     # (1,)
        self.ref_init_object_hand_pos_diff, self.ref_init_object_hand_rot_diff = compute_relative_pose(                     # (1, 3), (1, 4)
            self.ref_object_pose[:, :3], self.ref_object_pose[:, 3:7], self.ref_left_pose[:, :3], self.ref_left_pose[:, 3:7]
        )
        self.ref_init_object_hand_pos_diff, self.ref_init_object_hand_rot_diff = self.ref_init_object_hand_pos_diff.expand(num_envs, -1), self.ref_init_object_hand_rot_diff.expand(num_envs, -1)  # (num_envs, 3), (num_envs, 4)
        self.ref_init_tool_pos_dist = torch.norm(self.ref_tool_pose[:, :3] - self.tool_init_states[:, :3], dim=-1)          # (1,)
        self.ref_init_tool_rot_diff = quat_diff_theta(self.ref_tool_pose[:, 3:7], self.tool_init_states[:1, 3:7])           # (1,)        
        self.ref_init_tool_hand_pos_diff, self.ref_init_tool_hand_rot_diff = compute_relative_pose(                         # (1, 3), (1, 4)
            self.ref_tool_pose[:, :3], self.ref_tool_pose[:, 3:7], self.ref_right_pose[:, :3], self.ref_right_pose[:, 3:7]
        )
        self.ref_init_tool_hand_pos_diff, self.ref_init_tool_hand_rot_diff = self.ref_init_tool_hand_pos_diff.expand(num_envs, -1), self.ref_init_tool_hand_rot_diff.expand(num_envs, -1)  # (num_envs, 3), (num_envs, 4)

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

    def _prepare_object_asset(self, asset_root, asset_file, vhacd_enabled):
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
            # asset_options.vhacd_params.max_num_vertices_per_ch = 64 


        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0
        object_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return object_asset

    def _prepare_dataset(self):
        with open(self.cfg['dataset']['meta_data_path'], 'r') as f:
            self.dataset_taco_data = json.load(f)
        self.len_dataset = len(self.dataset_taco_data)
        self.task_id = self.cfg['task']['task_id']

    def _prepare_task(self, task_id=0, trans=None):
        assert len(self.dataset_taco_data) > 0 and isinstance(self.dataset_taco_data, list), "Please load the dataset first!"
        if task_id < 0 or task_id >= len(self.dataset_taco_data):
            print(f'Invalid task id {task_id}, using random task instead')
            task_id = random.randint(0, len(self.dataset_taco_data)-1)
        self.sampled_taco_task_data = self.dataset_taco_data[task_id]
        # timestep
        self.init_timestep = self.sampled_taco_task_data['key_steps']['init']
        self.ref_timestep = self.sampled_taco_task_data['key_steps']['ref']
        self.end_timestep = min(self.cfg['env']['episodeLength']-1, self.sampled_taco_task_data['key_steps']['end'])
        # objects
        if trans is None:
            trans_z_180 = np.array([
                [-1,  0,  0, 0],
                [ 0, -1,  0, 0],
                [ 0,  0,  1, 0],
                [ 0,  0,  0, 1]
            ])
            trans_z_neg90 = np.array([
                [ 0,  1,  0, 0],
                [-1,  0,  0, 0],
                [ 0,  0,  1, 0],
                [ 0,  0,  0, 1]
            ])
            trans = np.eye(4)
            # if self.mode == "visualize":
            #     trans[:3,3] = np.array([1,1,0])*0.1
        dataset_object_pose = trans @ np.array(self.sampled_taco_task_data['object']['T'])
        dataset_object_pos = dataset_object_pose[:,:3,3]
        dataset_object_quat = R.from_matrix(dataset_object_pose[:,:3,:3]).as_quat()
        self.dataset_object_init_pos = dataset_object_pos[self.init_timestep]
        self.dataset_object_init_quat = dataset_object_quat[self.init_timestep]
        self.dataset_object_pose = to_torch(torch.from_numpy(np.concatenate([
            dataset_object_pos,
            dataset_object_quat,
        ], axis=-1)), device=self.device, dtype=torch.float)
        
        dataset_tool_pose = trans @ np.array(self.sampled_taco_task_data['tool']['T'])
        dataset_tool_pos = dataset_tool_pose[:,:3,3]
        dataset_tool_quat = R.from_matrix(dataset_tool_pose[:,:3,:3]).as_quat()
        self.dataset_tool_init_pos = dataset_tool_pos[self.init_timestep]
        self.dataset_tool_init_quat = dataset_tool_quat[self.init_timestep]
        self.dataset_tool_pose = to_torch(torch.from_numpy(np.concatenate([
            dataset_tool_pos,
            dataset_tool_quat,
        ], axis=-1)), device=self.device, dtype=torch.float)
        

        # hand poses and finger joints
        dataset_left_dof = self.sampled_taco_task_data['left']
        dataset_left_finger_dof = dataset_left_dof['qpos']
        dataset_right_dof = self.sampled_taco_task_data['right']
        dataset_right_finger_dof = dataset_right_dof['qpos']

        self.dataset_left_pose = np.repeat(np.eye(4)[np.newaxis, ...], len(dataset_left_finger_dof), axis=0)
        self.dataset_left_pose[:,:3,:3] = R.from_quat(dataset_left_dof['q']).as_matrix()
        self.dataset_left_pose[:,:3,3] = dataset_left_dof['p']
        self.dataset_left_pose = trans @ self.dataset_left_pose
        dataset_left_pos = self.dataset_left_pose[:,:3,3]
        dataset_left_quat = R.from_matrix(self.dataset_left_pose[:,:3,:3]).as_quat()

        self.dataset_right_pose = np.repeat(np.eye(4)[np.newaxis, ...], len(dataset_right_finger_dof), axis=0)
        self.dataset_right_pose[:,:3,:3] = R.from_quat(dataset_right_dof['q']).as_matrix()
        self.dataset_right_pose[:,:3,3] = dataset_right_dof['p']
        self.dataset_right_pose = trans @ self.dataset_right_pose
        dataset_right_pos = self.dataset_right_pose[:,:3,3]
        dataset_right_quat = R.from_matrix(self.dataset_right_pose[:,:3,:3]).as_quat()


        self.both_fingers_dof = torch.from_numpy(np.concatenate([
            dataset_left_finger_dof,
            dataset_right_finger_dof,
        ], axis=-1)).to(self.device).float()
        self.target_left_pose = torch.from_numpy(np.concatenate([  # (num_timesteps, 7)
            dataset_left_pos,
            dataset_left_quat,
        ], axis=-1)).to(self.device).float()
        self.target_right_pose = torch.from_numpy(np.concatenate([  # (num_timesteps, 7)
            dataset_right_pos,
            dataset_right_quat,
        ], axis=-1)).to(self.device).float()

        # reference starting pose 
        self.ref_left_pose = self.target_left_pose[self.ref_timestep].unsqueeze(0)          # (1, 7)
        self.ref_right_pose = self.target_right_pose[self.ref_timestep].unsqueeze(0)        # (1, 7)
        self.ref_object_pose = self.dataset_object_pose[self.ref_timestep].unsqueeze(0)     # (1, 7)
        self.ref_tool_pose = self.dataset_tool_pose[self.ref_timestep].unsqueeze(0)         # (1, 7)

        self.ref_init_left_fingers_palm_pose = self.ref_init_right_fingers_palm_pose = None
        # if 'ref_fingers_pose' in self.sampled_taco_task_data['left']:
        #     self.ref_init_left_fingers_pose = to_torch(self.sampled_taco_task_data['left']['ref_fingers_pose'], device=self.device)     # (4, 7)
        #     self.ref_init_left_fingers_palm_pose = torch.cat([                                                                        # (1, 5, 7)
        #         self.ref_init_left_fingers_pose,
        #         self.ref_left_pose
        #     ]).unsqueeze(0)
        # elif self.mode == "visualize":
        #     self.ref_init_left_fingers_palm_pose = torch.cat([torch.zeros_like(self.ref_left_pose).repeat(4, 1),self.ref_left_pose]).unsqueeze(0)
        # else:
        #     raise ValueError("Please provide reference fingers pose for left hand!")
        # if 'ref_fingers_pose' in self.sampled_taco_task_data['right']:
        #     self.ref_init_right_fingers_pose = to_torch(self.sampled_taco_task_data['right']['ref_fingers_pose'], device=self.device)   # (4, 7)
        #     self.ref_init_right_fingers_palm_pose = torch.cat([                                                                       # (1, 5, 7)
        #         self.ref_init_right_fingers_pose,
        #         self.ref_right_pose
        #     ]).unsqueeze(0)
        # elif self.mode == "visualize":
        #     self.ref_init_right_fingers_palm_pose = torch.cat([torch.zeros_like(self.ref_right_pose).repeat(4, 1),self.ref_right_pose]).unsqueeze(0)
        # else:
        #     raise ValueError("Please provide reference fingers pose for right hand!")

    def _prepare_object_tool_pair(self, asset_root, vhacd_enabled=True):  
        assert isinstance(self.sampled_taco_task_data, dict), "Please load the dataset first!"
        object_mesh_path = os.path.join(asset_root, 'TACOobjects')
        self._create_urdf(self.sampled_taco_task_data['tool']['id'], self.sampled_taco_task_data['object']['id'], object_mesh_path)
        object_asset = self._prepare_object_asset(object_mesh_path, 'object.urdf', vhacd_enabled)
        tool_asset = self._prepare_object_asset(object_mesh_path, 'tool.urdf', vhacd_enabled)
        return object_asset, tool_asset

    def _prepare_table_asset(self):
        # create table asset
        keep_dis = 0.03  #0.3 if self.mode == "visualize" else 0.01
        table_dims = gymapi.Vec3(1.5, 1.5, min(self.dataset_object_init_pos[2],self.dataset_tool_init_pos[2]) - keep_dis)  # objects above table
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options
        )

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z / 2)

        side_panel_dims = gymapi.Vec3(0.06, 1.5, table_dims.z)
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

    def compute_reward(self, mode):
        if mode == 'grasp':
            (
                self.rew_buf[:],
                self.reset_buf[:],
                self.progress_buf[:],
                self.successes[:],
                self.current_successes[:],
                self.consecutive_successes[:],
                reward_info
            ) = compute_grasp_rewards(
                self.reset_buf,
                self.progress_buf,
                self.successes,
                self.current_successes,
                self.consecutive_successes,
                self.max_episode_length,
                self.object_pos, self.tool_pos,
                self.goal_height,
                self.left_palm_pos, self.right_palm_pos,
                self.left_fingertip_pos, self.right_fingertip_pos,
                self.dist_reward_scale,
                self.object_init_states, self.tool_init_states,
                self.action_penalty_scale,
                self.success_tolerance,
                self.av_factor,
                self.table_height,
                self.actions,
            )
        elif mode == 's1':
            (
                self.rew_buf[:],
                self.reset_buf[:],
                self.progress_buf[:],
                self.successes[:],
                self.current_successes[:],
                self.consecutive_successes[:],
                reward_info
            ) = compute_bvdex_stage1_rewards(
                self.reset_buf,
                self.progress_buf,
                self.successes,
                self.current_successes,
                self.consecutive_successes,
                self.max_episode_length,
                self.object_pose, self.tool_pose,
                self.left_palm_pose, self.right_palm_pose,
                self.left_fingertip_pos, self.right_fingertip_pos,
                self.dist_reward_scale,
                self.action_penalty_scale,
                self.success_tolerance,
                self.av_factor,
                self.table_height,
                self.actions,
                self.ref_object_pose, self.ref_init_object_pos_dist, self.ref_init_object_hand_rot_diff,
                self.ref_tool_pose, self.ref_init_tool_pos_dist, self.ref_init_tool_hand_rot_diff,
                self.is_stage1_min_rew,
            )
        elif mode == 's12':
            # ref_object_pose = self.ref_object_pose.repeat(self.num_envs, 1)
            # ref_tool_pose = self.ref_tool_pose.repeat(self.num_envs, 1)
            t_left = torch.where(self.left_reach_ref_timestep == -1, torch.zeros_like(self.timestep), torch.ceil((self.timestep - self.left_reach_ref_timestep)/self.frequency).int()) + self.ref_timestep
            t_right = torch.where(self.right_reach_ref_timestep == -1, torch.zeros_like(self.timestep), torch.ceil((self.timestep - self.right_reach_ref_timestep)/self.frequency).int()) + self.ref_timestep
            ref_object_pose = self.dataset_object_pose[t_left.clip(max=self.end_timestep)]  
            ref_tool_pose = self.dataset_tool_pose[t_right.clip(max=self.end_timestep)]  

            (
                self.rew_buf[:],
                self.reset_buf[:],
                self.progress_buf[:],
                self.successes[:],
                self.current_successes[:],
                self.consecutive_successes[:],
                self.timestep[:], self.left_reach_ref_timestep[:], self.right_reach_ref_timestep[:],
                reward_info
            ) = compute_bvdex_stage12_rewards(
                self.reset_buf,
                self.progress_buf,
                self.successes,
                self.current_successes,
                self.consecutive_successes,
                self.max_episode_length,
                self.object_pose, self.tool_pose,
                self.left_palm_pose, self.right_palm_pose,
                self.left_fingertip_pos, self.right_fingertip_pos,
                self.dist_reward_scale,
                self.action_penalty_scale,
                self.success_tolerance,
                self.av_factor,
                self.table_height,
                self.actions,
                self.timestep, self.left_reach_ref_timestep, self.right_reach_ref_timestep,
                ref_object_pose, self.ref_init_object_pos_dist, self.ref_init_object_hand_pos_diff, self.ref_init_object_hand_rot_diff,self.ref_init_left_fingers_palm_pose,
                ref_tool_pose, self.ref_init_tool_pos_dist, self.ref_init_tool_hand_pos_diff, self.ref_init_tool_hand_rot_diff,self.ref_init_right_fingers_palm_pose,
                self.is_stage1_min_rew, self.is_stage1_lin_rew, self.is_stage2_pos_rew_exp,
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

        self.timestep[:] += 1

        return self.compute_full_observations()

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
            self.obs_buf[:, cnt : cnt + 3] = self.object_pos - self.left_palm_pos
            self.obs_buf[:, cnt + 3 : cnt + 15] = (self.object_pos.unsqueeze(1) - self.left_fingertip_pos).reshape(-1,12)
            self.obs_buf[:, cnt + 15: cnt + 18] = self.tool_pos - self.right_palm_pos
            self.obs_buf[:, cnt + 18 : cnt + 30] = (self.tool_pos.unsqueeze(1) - self.right_fingertip_pos).reshape(-1,12)
            cnt += 30

        # assert dim
        assert cnt == self.obs_buf.shape[1]

        return self.obs_buf

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
        # TODO
        

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
        self.successes[env_ids] = 0
        self.timestep[env_ids] = 0
        self.left_reach_ref_timestep[env_ids] = -1
        self.right_reach_ref_timestep[env_ids] = -1

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

    def visualize(self, replay_times=5, debug=False, vis_metrics=True, vis_mode='ref', append_data=True):
        def visualize_curves(data_dict):
            """
            Visualize each list in the dictionary as a curve in a 2xM matrix of subplots.

            Parameters:
            data_dict (dict): A dictionary where keys are labels and values are lists of data points.
            """
            data_dict =  {key: data_dict[key] for key in sorted(data_dict.keys())}
            num_keys = len(data_dict)
            num_row = 4
            num_col = int(np.ceil(num_keys / num_row))
            fig, axs = plt.subplots(num_row, num_col, figsize=(4*num_col, 3*num_row))
            axs = axs.flatten()
            for i, (key, values) in enumerate(data_dict.items()):
                axs[i].plot(values, label=key)
                axs[i].set_title(key)
                axs[i].legend()
            # Hide any remaining subplots if the number of keys is odd
            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])
            plt.tight_layout()
            plt.show()

        for replay_times in range(1,1+replay_times):
            self._prepare_task(task_id=self.task_id)
            metric_collector = defaultdict(list)
            end_timestep = self.ref_timestep if vis_mode == 'ref' else self.end_timestep
            for i in range(self.init_timestep-1, end_timestep+1):
                self.actions = torch.zeros_like(self.robot_dof_pos)
                self.actions[:, self.both_fingers_dof_indices] = self.both_fingers_dof[i:i+1]
                self.actions[:, self.both_arm_dof_indices] = self.calculate_ik(self.target_left_pose[i:i+1], self.target_right_pose[i:i+1])
                if append_data and i == self.ref_timestep:
                    # append to self.sampled_taco_task_data
                    self.sampled_taco_task_data['left']['ref_fingers_pose'] = self.left_fingertip_pose[0].tolist()
                    self.sampled_taco_task_data['right']['ref_fingers_pose'] = self.right_fingertip_pose[0].tolist()
                    self.dataset_taco_data[self.task_id] = self.sampled_taco_task_data
                    with open(self.cfg['dataset']['meta_data_path'], 'w') as f:
                        json.dump(self.dataset_taco_data, f, indent=4)
                # step dataset in the environment
                # 1.set dof state
                self.robot_dof_pos[:] = self.actions
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_state))
                # self.prev_targets[:] = self.robot_dof_pos[:]
                # self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(self.both_robot_dof_indices.to(torch.int32)), len(self.both_robot_dof_indices))
        
                # 2.step object
                self.root_state_tensor[self.tool_indices, 0:3] = self.dataset_tool_pose[i, 0:3]
                self.root_state_tensor[self.tool_indices, 3:7] = self.dataset_tool_pose[i, 3:7]
                self.root_state_tensor[self.object_indices, 0:3] = self.dataset_object_pose[i, 0:3]
                self.root_state_tensor[self.object_indices, 3:7] = self.dataset_object_pose[i, 3:7]
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
                # 3.step simulation
                self.render()
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.compute_observations()

                
                if debug:
                    # save image
                    if self.cfg['env']['enableCameraSensors']:
                        self.gym.render_all_camera_sensors(self.sim)
                        color_image = self.gym.get_camera_image(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
                        cv2.imwrite(os.path.join(self.cfg['env']['imageSaveDir'], f'{i}.jpg'), color_image)
                    print('-'*50, f'step:{i}', '-'*50,)
                    print('actions:', self.actions)
                    print('object state:', self.root_state_tensor[self.object_indices],)
                    print('tool state:', self.root_state_tensor[self.tool_indices],)
                    if i % 5 == 0:
                        input()
                if vis_metrics:    
                    # compute metrics
                    # if i == self.ref_timestep:
                    #     breakpoint()
                    metrics = self.compute_reward(mode='s12')
                    for k,v in metrics.items():  # for visualize metrics
                        metric_collector[k].append(v.float().mean().item())

                
            if vis_metrics:
                visualize_curves(metric_collector)

    