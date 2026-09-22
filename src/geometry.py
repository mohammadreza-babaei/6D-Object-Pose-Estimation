"""Geometry utilities for 6D pose estimation."""

import torch
import torch.nn.functional as F


def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""

    q = F.normalize(quaternion, dim=-1)

    w, x, y, z = q.unbind(dim=-1)

    rotation_matrix = torch.stack([
        1 - 2 * (y**2 + z**2),
        2 * (x * y - z * w),
        2 * (x * z + y * w),

        2 * (x * y + z * w),
        1 - 2 * (x**2 + z**2),
        2 * (y * z - x * w),

        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x**2 + y**2),
    ], dim=-1)

    return rotation_matrix.reshape(*q.shape[:-1], 3, 3)
def rotation_matrix_to_quaternion(matrix):
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""

    m = matrix

    w = torch.sqrt(torch.clamp(1 + m[0, 0] + m[1, 1] + m[2, 2], min=0)) / 2
    x = torch.sqrt(torch.clamp(1 + m[0, 0] - m[1, 1] - m[2, 2], min=0)) / 2
    y = torch.sqrt(torch.clamp(1 - m[0, 0] + m[1, 1] - m[2, 2], min=0)) / 2
    z = torch.sqrt(torch.clamp(1 - m[0, 0] - m[1, 1] + m[2, 2], min=0)) / 2

    x = torch.copysign(x, m[2, 1] - m[1, 2])
    y = torch.copysign(y, m[0, 2] - m[2, 0])
    z = torch.copysign(z, m[1, 0] - m[0, 1])

    return F.normalize(torch.stack([w, x, y, z]), dim=0)