import numpy as np
from scipy.spatial.transform import Rotation as R
from isaacgymenvs.utils.torch_jit_utils import *
import torch

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
    b_position: torch.Tensor,
    b_orientation: torch.Tensor,
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




def transformation_inverse_np(quat, pos):
    """Invert a transformation.
    
    Args:
        quat: Quaternion of the transformation (shape: (4,))
        pos: Position of the transformation (shape: (3,))

    Returns:
        Inverted quaternion and position.
    """
    quat_inv = R.from_quat(quat).inv().as_quat()
    pos_inv = -R.from_quat(quat_inv).apply(pos)
    return quat_inv, pos_inv

def compute_relative_position_np(a_position, b_position, b_orientation):
    """Compute the position of `a` in frame `b`.
    
    Args:
        a_position: Position of `a` in world frame (shape: (3,))
        b_position: Position of `b` in world frame (shape: (3,))
        b_orientation: Orientation of `b` in world frame (shape: (4,))

    Returns:
        Position of `a` in frame `b` (shape: (3,))
    """
    # Compute the inverse of the transformation from `b` to world
    w2b_rotation, w2b_translation = transformation_inverse_np(b_orientation, b_position)

    # Apply the inverse transformation to `a_position`
    position_in_b = R.from_quat(w2b_rotation).apply(a_position) + w2b_translation
    
    return position_in_b

ap = np.array([[1, 2, 3],[1, 2, 3]])    
bp = np.array([[4, 5, 6],])
bo = np.array([[0, 1, 0, 0]])
print(compute_relative_position(torch.tensor(ap).unsqueeze(0), torch.tensor(bp).unsqueeze(0), torch.tensor(bo).unsqueeze(0)))
print(compute_relative_position_np(ap, bp, bo))

