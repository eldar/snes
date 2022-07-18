import numpy as np
import torch


def homogenise_np(p):
    _1 = np.ones((p.shape[0], 1), dtype=p.dtype)
    return np.concatenate([p, _1], axis=-1)


def inside_axis_aligned_box(pts, box_min, box_max):
    return torch.all(torch.cat([pts >= box_min, pts <= box_max], dim=1), dim=1)


def homogenise_torch(p):
    _1 = torch.ones_like(p[:, [0]])
    return torch.cat([p, _1], dim=-1)


def transform_points(T, pts):
    pts = homogenise_torch(pts)
    pts = torch.einsum('ji,ni->nj', T, pts)[:, :3]
    return pts