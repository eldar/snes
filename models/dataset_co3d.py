import os
import sys
from pathlib import Path
import logging
import pickle

import numpy as np
import torch
import torch.nn.functional as F

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(code_dir, "3rdparty/co3d"))

from dataset.co3d_dataset import Co3dDataset as Co3dData
from models.camera import PerspectiveCamera
from util.coord import homogenise_torch


def to_global_path(path):
    # Converts a local path to a path relative to the source tree root
    path = Path(path)
    if not path.exists():
        path = Path(code_dir).joinpath(path)
    return path


def load_co3d(cfg):
    category = cfg.category
    dataset_root = to_global_path(cfg.data_dir)
    logging.info(f"Dataset root: {dataset_root}")
    logging.info(f"Category: {category}")
    logging.info(f"Instance: {cfg.instance}")
    frame_file = os.path.join(dataset_root, category, "frame_annotations.jgz")
    sequence_file = os.path.join(dataset_root, category, "sequence_annotations.jgz")
    sequence_name = cfg.instance
    dataset = Co3dData(
        frame_annotations_file=frame_file,
        sequence_annotations_file=sequence_file,
        dataset_root=dataset_root,
        image_height=None,
        image_width=None,
        box_crop=False,
        load_point_clouds=True,
        pick_sequence=[sequence_name]
    )
    return dataset


def load_auto_bbox_scale(cfg, margin_scale, apply_scaling):
    data_extra_dir = to_global_path(cfg.data_extra_dir)
    path = data_extra_dir.joinpath(cfg.category, cfg.instance, "alignment.npy")
    data = np.load(path, allow_pickle=True).item()
    T = data["T"]
    box = data["box_size"]

    s = np.zeros((4,4), np.float32)
    if apply_scaling:
        max_sz = np.max(box)
        scale = 2.0 / max_sz * margin_scale
        np.fill_diagonal(s, scale)
        s[3, 3] = 1.0
        total = s @ T
    else:
        np.fill_diagonal(s, 1.0)
        total = T

    return total, box, s


class Dataset:
    def __init__(self, cfg, split="train", other=None, device='cuda'):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device(device)
        self.cfg = cfg

        if other is not None:
            co3d = other.co3d
        else:
            co3d = load_co3d(cfg)
        self.co3d = co3d

        point_cloud = co3d[0].sequence_point_cloud.to("cpu")
        self.point_cloud_xyz = point_cloud.points_padded()
        self.point_cloud_rgb = point_cloud.features_padded()
        self.point_cloud_quality_score = co3d[0].point_cloud_quality_score

        num_frames = len(co3d)
        self.n_images = num_frames

        self.pytorch3d_cameras = [co3d[idx].camera for idx in range(num_frames)]

        filenames = [co3d.frame_annots[i]["frame_annotation"].image.path for i in range(num_frames)]
        filenames = [os.path.split(f)[-1] for f in filenames]
        self.filenames = filenames

        auto_box = cfg.use_auto_box
        if auto_box:
            scaling_factor = cfg.scaling_factor
            scale_obj, box, scale = load_auto_bbox_scale(cfg, scaling_factor, cfg.apply_scaling)
            scale_obj = torch.from_numpy(scale_obj) # C2W
            self.bbox_scale_transform = scale

            max_sz = np.max(box)
            box_scaled = box / max_sz
            object_bbox_min = -box_scaled * np.array([1.0, 1.1, 1.0])
            object_bbox_max = box_scaled * np.array([1.0, 1.1, 1.1])
            self.raw_bbox_min = -box / max_sz * scaling_factor
            self.raw_bbox_max =  box / max_sz * scaling_factor

            pts = homogenise_torch(self.point_cloud_xyz.squeeze())
            pts = torch.einsum('ji,ni->nj', scale_obj, pts)[:, :3]

            bbox_min = torch.from_numpy(self.raw_bbox_min)
            bbox_max = torch.from_numpy(self.raw_bbox_max)

            mask = torch.cat([pts >= bbox_min, pts <= bbox_max], dim=1)
            mask = torch.all(mask, dim=-1)

            self.point_cloud_xyz_canonical = pts[mask].to(self.device)
        else:
            scale_obj = None
            object_bbox_min = np.array([-1.01, -1.01, -1.01])
            object_bbox_max = np.array([ 1.01,  1.01,  1.01])

        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]

        self.global_alignment = scale_obj

        self.images = [None] * num_frames
        self.masks = [None] * num_frames
        self.co3d_masks = [None] * num_frames
        self.co3d_depth_masks = [None] * num_frames
        self.co3d_depth_maps = [None] * num_frames
        self.cameras = []

        for idx in range(num_frames):
            frame_data = co3d[idx]
            img = frame_data.image_rgb
            self.images[idx] = img.permute(1, 2, 0).cpu()
            # self.masks[idx] = frame_data.fg_probability.cpu()
            self.masks[idx] = torch.ones_like(self.images[idx]).cpu()
            self.co3d_masks[idx] = frame_data.fg_probability
            self.co3d_depth_masks[idx] = frame_data.depth_mask
            self.co3d_depth_maps[idx] = frame_data.depth_map

            cam = frame_data.camera
            new_cam = PerspectiveCamera.from_pytorch3d(cam, img.shape[1:])
            if scale_obj is not None:
                new_cam = new_cam.left_transformed(scale_obj)
            new_cam.to(self.device)
            self.cameras.append(new_cam)
        
        if cfg.trainval_split:
            data_extra_dir = to_global_path(cfg.data_extra_dir)
            path = data_extra_dir.joinpath(cfg.category, cfg.split_file)
            split_data = pickle.load(open(path, "rb"))
            instance = split_data[cfg.instance]
            ids = instance[split]
            self.n_images = len(ids)

            slice_ = lambda arr, idx: [arr[i] for i in idx]
            self.cameras = slice_(self.cameras, ids)
            self.images = slice_(self.images, ids)
            self.masks = slice_(self.masks, ids)
            self.filenames = slice_(self.filenames, ids)
            self.co3d_masks = slice_(self.co3d_masks, ids)
            self.co3d_depth_masks = slice_(self.co3d_depth_masks, ids)
            self.co3d_depth_maps = slice_(self.co3d_depth_maps, ids)

        print('Load data: End')

    def get_cameras(self):
        return self.cameras

    def get_ground_plane_z(self):
        return self.raw_bbox_min[2].item()

    def near_far_from_sphere(self, rays):
        rays_o = rays.origins
        rays_d = rays.directions
        a = torch.sum(rays_d**2, dim=-1, keepdim=True).sqrt()
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        far = mid + 1.0
        min_depth = float(self.cfg.min_depth)
        if min_depth == -1:
            near = mid - 1.0
        else:
            near = torch.ones_like(far) * min_depth
        return near, far

    # def near_far_from_sphere(self, rays):
    #     # Assumes unit sphere
    #     rays_o = rays.origins
    #     rays_d = rays.directions
    #     rays_d = F.normalize(rays_d, p=2.0, dim=-1)
    #     mid = -torch.sum(rays_o * rays_d, dim=-1, keepdim=True) # distance to the perpendicular bisector of ray and sphere origin
    #     far = mid + 1.0
    #     min_depth = float(self.cfg.min_depth)
    #     if min_depth == -1:
    #         near = mid - 1.0
    #     elif min_depth == 0: # near starts at sphere, if ray intersects sphere
    #         # for rays that do not intersect unit sphere, use near = mid - 1.0,
    #         # otherwise use first intersection of ray and sphere
    #         delta2 = (rays_o ** 2).sum(dim=-1, keepdim=True) - mid ** 2 # squared distance to perpendicular bisector
    #         t = 1.0 - delta2
    #         t[t < 0] = 0.0
    #         near = torch.where(delta2 < 1.0, mid - torch.sqrt(t), mid - 1.0)
    #     else:
    #         near = torch.ones_like(far) * min_depth
    #     return near, far