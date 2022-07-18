import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import so3_exp_map

def mask_translation(cfg, t_):
    cfg_r = cfg.model.renderer
    if cfg_r.symmetry_translation_1dof:
        t_ = t_ * t_.new_tensor([[0., 1., 0.]])
    elif cfg_r.symmetry_translation_2dof:
        t_ = t_ * t_.new_tensor([[0., 1., 1.]])
    return t_

def compute_4x4_transform(r, t, return_inverse=False):
    R = so3_exp_map(r)
    T = torch.cat([R, t[:, :, None]], dim=-1)
    T = torch.cat([T, torch.zeros_like(T[:, [0], :])], dim=1)
    T[..., 3, 3] = 1.0
    if return_inverse:
        Rtr = R.transpose(-2, -1)
        Rtrt = torch.einsum('...ij,...jk->...ik', Rtr, t[:, :, None])
        invT = torch.cat([Rtr, -Rtrt], dim=-1)
        invT = torch.cat([invT, torch.zeros_like(invT[:, [0], :])], dim=1)
        invT[..., 3, 3] = 1.0
        return T, invT
    return T

class TransformManager(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.r_ = nn.Parameter(torch.zeros((1, 3)).cuda())
        self.t_ = nn.Parameter(torch.zeros((1, 3)).cuda())
        self.r_.requires_grad_()
        self.t_.requires_grad_()

        # TODO: implement state to avoid recomputation
        self.T = None

    def get_transform(self, return_inverse=False):
        """ Compute transformation matrix from t and r.
        """
        t = mask_translation(self.cfg, self.t_)
        return compute_4x4_transform(self.r_, t, return_inverse)
    
    def apply_transform(self, x, is_dir=False):
        """ Applies transformation to a batch of 3D points.
        Args:
            x: point cloud with shape (..., 3)
        Returns:
            x_transformed: transformed point cloud with shape (..., 3)
        """
        pad_val = 0.0 if is_dir else 1.0 # 0 for directions, 1 for points
        x = F.pad(x, (0, 1), mode='constant', value=pad_val) # homogenise
        T = self.get_transform()
        return torch.einsum('...ij,...j->...i', T, x)[..., :3]

class SymmetryManager(TransformManager):
    """ Computes and applies a canonical symmetry transformation.
    The transformation T^-1 * S * T is applied to homogeneous coordinates.
    While not a minimal parametrisation, this approach allows us to use the
    transformation for other purposes as well, such as ground plane fitting and
    bounding box alignment, if desired. Minimal parametrisations are presented
    in the supplementary material.
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        cfg_r = cfg.model.renderer
        if cfg_r.symmetry_type == 'planar_reflection':
            # Set canonical planar reflection matrix
            # I - 2 * e[dim] * e[dim]^T
            # that reflects points about the plane perpendicular to the axis
            # corresponding to reflection_dim.
            self.reflection_dim = cfg_r.symmetry_axis
            self.canonical_symmetry_matrix = torch.eye(4).cuda()
            self.canonical_symmetry_matrix[self.reflection_dim, self.reflection_dim] *= -1
            self.canonical_symmetry_matrix = self.canonical_symmetry_matrix.unsqueeze(0) # add batch dim
            # Alternatively, use T = [I - 2*n*n^T, 2*delta*n; 0, 1]
            # where n is the unit normal vector and delta is the scalar offset
            # defining the plane; a minimal parametrisation.
        elif cfg_r.symmetry_type == 'line_reflection':
            raise NotImplementedError("line reflections not implemented")
        elif cfg_r.symmetry_type == 'point_reflection':
            self.canonical_symmetry_matrix = -torch.eye(4).cuda()
            # Alternatively, use T = [-I, 2*p; 0, 1]
            # where p is the 3D point to reflect about.
        elif cfg_r.symmetry_type == 'rotation':
            self.rotation_angle = cfg_r.symmetry_angle
            raise NotImplementedError("rotation symmetries not implemented")
    
    def get_symmetry_transform(self):
        """ Compute symmetry transformation.
        """
        T, invT = self.get_transform(return_inverse=True) # Get transformation to canonical coordinates
        invTRT = invT @ (self.canonical_symmetry_matrix @ T)
        return invTRT

    def apply_symmetry_transform(self, x, is_dir=False):
        """ Applies symmetry transformation to a batch of 3D points.
        Args:
            x: point cloud with shape (..., 3)
        Returns:
            transformed point cloud with shape (..., 3)
        """
        pad_val = 0.0 if is_dir else 1.0 # 0 for directions, 1 for points
        x = F.pad(x, (0, 1), mode='constant', value=pad_val) # homogenise points
        invTRT = self.get_symmetry_transform()
        return torch.einsum('...ij,...j->...i', invTRT, x)[..., :3]
    
    def vis_symmetry_plane(self):
        _, invT = self.get_transform(return_inverse=True)

        reflection_dim = self.reflection_dim
        height = 0.5
        v = torch.tensor([
            [-1, -1, -height],
            [-1, -1, height],
            [1, 1, height],
            [1, 1, -height]
        ], dtype=torch.float32, device=invT.device)
        v[:, reflection_dim] = 0

        v = F.pad(v, (0, 1), mode='constant', value=1.0) # homogenise
        v_w = torch.einsum('...ij,...j->...i', invT, v)[..., :3]

        tris = torch.tensor([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=torch.int32, device=invT.device)
        return v_w.detach().cpu(), tris.cpu()