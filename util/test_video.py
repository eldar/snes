from typing import Tuple
import math
import torch

from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform


def generate_eval_video_cameras(
    cfg,
    cameras,
    traj_radius = 2.5,
    n_eval_cams: int = 100,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    """
    loosely based on
    https://github.com/facebookresearch/pytorch3d/blob/main/projects/nerf/nerf/eval_video_utils.py
    """

    height = 0.5
    angle = torch.linspace(0, 2.0 * math.pi, n_eval_cams+1)[:-1]
    traj = torch.stack(
        (traj_radius*angle.cos(), traj_radius*angle.sin(), torch.ones_like(angle)*height), dim=-1
    )

    # point all cameras towards the center of the scene
    R, T = look_at_view_transform(
        eye=traj,
        at=(scene_center,),  # (1, 3)
        up=(up,),  # (1, 3)
        device=traj.device,
    )

    # choose focal length and image size
    if cfg.test.fixed_test_cameras:
        image_size = tuple(cfg.test.camera_image_size)
        focal_actual = torch.tensor(cfg.test.camera_focal_length).cpu()
        focal = focal_actual * 2 / (torch.tensor(image_size)[[1, 0]] - 1).cpu()
    else:
        # get the average focal length and principal point
        cam = cameras[0]
        focal = torch.diag(cam.get_intrinsics())[:2].cpu() * 2 \
            / (torch.tensor(cam.get_image_size())[[1, 0]] - 1).cpu()
        image_size = cameras[0].get_image_size()

    # assemble the dataset
    test_cameras = [
        PerspectiveCameras(
            focal_length=focal[None, :],
            R=R_[None],
            T=T_[None],
            image_size=(image_size,)
        )
        for (R_, T_) in zip(R, T)
    ]

    return test_cameras


def _figure_eight_knot(t: torch.Tensor, z_scale: float = 0.5):
    x = (2 + (2 * t).cos()) * (3 * t).cos()
    y = (2 + (2 * t).cos()) * (3 * t).sin()
    z = (4 * t).sin() * z_scale
    return torch.stack((x, y, z), dim=-1)


def _trefoil_knot(t: torch.Tensor, z_scale: float = 0.5):
    x = t.sin() + 2 * (2 * t).sin()
    y = t.cos() - 2 * (2 * t).cos()
    z = -(3 * t).sin() * z_scale
    return torch.stack((x, y, z), dim=-1)


def _figure_eight(t: torch.Tensor, z_scale: float = 0.5):
    x = t.cos()
    y = (2 * t).sin() / 2
    z = t.sin() * z_scale
    return torch.stack((x, y, z), dim=-1)