import torch
import torch.nn as nn

from pytorch3d.transforms import so3_exp_map
from pytorch3d.transforms import Scale

from pytorch3d.renderer.utils import TensorProperties


def invert_intrinsics(K):
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    px = K[:, 0, 2]
    py = K[:, 1, 2]
    _0 = torch.zeros_like(fx)
    _1 = torch.ones_like(fx)
    K_inv = [
        [1/fx,    _0, -px/fx],
        [  _0,  1/fy, -py/fy],
        [  _0,    _0,     _1]
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in K_inv], dim=-2)


class RayBundle:
    """
    RayBundle parametrizes points along projection rays by storing ray `origins`,
    `directions` vectors and `lengths` at which the ray-points are sampled.
    Furthermore, the xy-locations (`xys`) of the ray pixels are stored as well.
    Note that `directions` don't have to be normalized; they define unit vectors
    in the respective 1D coordinate systems; see documentation for
    :func:`ray_bundle_to_ray_points` for the conversion formula.
    """

    origins: torch.Tensor
    directions: torch.Tensor
    xys: torch.Tensor

    def __init__(self, xys=None, origins=None, directions=None):
        self.xys = xys
        self.origins = origins
        self.directions = directions
    
    def split(self, chunk_size):
        xys = self.xys.split(chunk_size)
        if self.origins is not None:
            origins = self.origins.split(chunk_size)
            directions = self.directions.split(chunk_size)
        else:
            origins = [None]*len(xys)
            directions = [None]*len(xys)
        rays = [RayBundle(*r) for r in zip(xys, origins, directions)]
        return rays

class PerspectiveCamera(TensorProperties):
    def __init__(self, image_size, intrinsics, pose):
        super().__init__()

        self.image_size = image_size.clone().type(torch.int32).reshape(-1, 2)
        self.intrinsics = intrinsics.reshape(-1, 4, 4)
        self.inv_intrinsics = invert_intrinsics(self.intrinsics)
        self.pose = pose.reshape(-1, 4, 4)
    
    def get_image_size(self):
        return self.image_size

    def get_intrinsics(self):
        return self.intrinsics

    def get_inverse_intrinsics(self):
        return self.inv_intrinsics

    def get_pose_matrix(self):
        """
        returns: camera to world transformation matrix
        """
        return self.pose
    
    @staticmethod
    def from_pytorch3d(cam, image_size=None):
        if image_size is None:
            image_size = tuple(cam.image_size[0].tolist())
        img_h = int(image_size[0])
        img_w = int(image_size[1])
        f = cam.focal_length.squeeze()
        K = torch.zeros((4, 4))
        K[0, 0] = f[0] * (img_w-1) / 2
        K[1, 1] = f[1] * (img_h-1) / 2
        # K[:2, 2] = cam.principal_point.squeeze()
        K[0, 2] = (img_w) / 2
        K[1, 2] = (img_h) / 2
        K[2, 2] = 1
        K[3, 3] = 1

        E1 = torch.transpose(cam.get_world_to_view_transform().get_matrix(), 1, 2)
        scale = Scale(x=-1, y=-1, z=1).to(E1.device)
        E2 = torch.matmul(scale.get_matrix(), E1).squeeze()
        c2w = torch.linalg.inv(E2)

        return PerspectiveCamera(torch.tensor((img_h, img_w)), K, c2w)
    
    def left_transformed(self, t):
        pose = torch.matmul(t, self.get_pose_matrix())
        return PerspectiveCamera(self.get_image_size(),
                                 self.get_intrinsics(),
                                 pose)

    @staticmethod
    def from_list(cameras):
        return PerspectiveCamera(
            torch.cat([c.image_size for c in cameras]),
            torch.cat([c.intrinsics for c in cameras]),
            torch.cat([c.pose for c in cameras])
        )
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = torch.tensor([idx]).type(torch.int64)
        return PerspectiveCamera(
            self.image_size[idx, ...],
            self.intrinsics[idx, ...],
            self.pose[idx, ...],
        )

class CameraManager(torch.nn.Module):
    def __init__(self, cameras, cfg):
        super().__init__()

        self.cfg = cfg
        self.cameras = cameras
        self.batched_cameras = PerspectiveCamera.from_list(cameras).cuda()

    def get_camera(self, idx):
        return self.cameras[idx]

    def get_cameras(self, indices):
        return self.batched_cameras[indices]
