import startup

import os
import logging
import argparse
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from sklearn.covariance import MinCovDet
from sklearn import linear_model

from util.config import construct_config

from models.factory import dataset_factory
from util.min_bounding_rect import minBoundingRect
from util.coord import homogenise_np
from util.point_cloud import filter_pointcloud


def _up_direction(cameras):
    w2c = [cam.get_world_to_view_transform().get_matrix().transpose(-1, -2).squeeze() \
               for cam in cameras]
    C2W = torch.stack([torch.linalg.inv(W2C) for W2C in w2c])
    f_pts = torch.FloatTensor([[0., 0., 0.],
                               [0., 1., 0.]])
    pts = torch.cat([f_pts, torch.ones_like(f_pts[:, 0:1])], dim=1)
    t = torch.einsum('bji,pi->bpj', C2W, pts)
    camera_up = (t[:, 1, :] - t[:, 0, :])[..., :3]
    return camera_up


def camera_centers_up(cameras):
    cam_centers = torch.cat(
        [cam.get_camera_center() for cam in cameras]
    )

    # fit plane to the camera centers
    cov = torch.cov(cam_centers.transpose(-2, -1))
    # plane_mean = cam_centers.mean(dim=0)
    # cam_centers_c = cam_centers - plane_mean[None]
    # cov = (cam_centers_c.t() @ cam_centers_c) / cam_centers_c.shape[0] # biased
    _, e_vec = torch.linalg.eigh(cov)
    up = e_vec[:, 0]
    
    # plane_normal = _up_direction(plane_normal, train_dataset)
    camera_up = _up_direction(cameras)

    cos = torch.einsum('ji,i->j', camera_up, up) / (torch.norm(camera_up, dim=1) * torch.norm(up))
    up = up if torch.mean(cos) > 0 else -up
    return up


def skew_symmetric(v):
    return np.array(
        [[0,    -v[2],  v[1]],
         [v[2],     0, -v[0]],
         [-v[1], v[0],     0]],
        dtype=v.dtype
    )


def ground_plane_rotation(up_vec):
    # align with Z axis
    unit_z = np.array([0., 0., 1.], np.float32)
    up_vec = up_vec / np.linalg.norm(up_vec)

    v = np.cross(up_vec, unit_z)
    s = np.linalg.norm(v)
    c = np.dot(up_vec, unit_z)
    v_x = skew_symmetric(v)

    I = np.eye(3, dtype=np.float32)
    R = I + v_x + v_x @ v_x * (1-c) / s**2
    return R


def estimate_ground_plane(co3d):
    num_frames = len(co3d)
    cameras = [co3d[i].camera for i in range(num_frames)]
    camera_up = _up_direction(cameras)
    up_vector = torch.mean(camera_up, axis=0)
    up_vector_2 = camera_centers_up(cameras)
    up_vector = (up_vector+up_vector_2)*0.5

    R = ground_plane_rotation(up_vector)
    return R


def fit_box_2d(hull_points):
    R, rot_angle, rect, area, width, height, center_point, corner_points = minBoundingRect(hull_points)
    min_x, min_y, max_x, max_y = rect
    if (max_x - min_x) < (max_y - min_y):
        r = Rotation.from_rotvec(-np.pi/2 * np.array([0, 0, 1])).as_matrix()[:2, :2]
        R = r @ R
    return R


def vis_bev_box_fit(points, hull_points, R, center_xy, rect, vis_path):
    min_x, max_x, min_y, max_y = rect[:4]
    center_xy = center_xy.reshape((-1, 2))
    proj_points = np.einsum('ji,ni->nj', R, points)
    proj_hull_points = np.einsum('ji,ni->nj', R, hull_points)

    corner_points_orig = np.zeros((4,2)) # empty 2 column array
    corner_points_orig[0] = np.array([max_x, min_y])
    corner_points_orig[1] = np.array([min_x, min_y])
    corner_points_orig[2] = np.array([min_x, max_y])
    corner_points_orig[3] = np.array([max_x, max_y])

    proj_points += center_xy
    proj_hull_points += center_xy
    corner_points_orig += center_xy

    def split_xy(pts):
        return pts[:, 0], pts[:, 1]

    plt.plot(*split_xy(proj_points), 'o')
    plt.plot(*split_xy(proj_hull_points), 'ro')

    poly = Polygon(corner_points_orig, closed=True) # , fill=False
    p = PatchCollection([poly], alpha=0.4)
    ax = plt.gca()
    ax.add_collection(p)
    
    plt.axis('scaled')
    plt.savefig(f"{vis_path}_bev.png")


def vis_side_box_fit(pcl_orig, T, size, vis_path):
    s = size / 2
    w, h = s[0], s[2]
    corners = np.array([[-w, -h], [-w, h], [w, h], [w, -h]])
    poly = Polygon(corners, closed=True) # , fill=False
    p = PatchCollection([poly], alpha=0.4)
    ax = plt.gca()
    ax.add_collection(p)

    pcl = np.einsum('ji,ni->nj', T, homogenise_np(pcl_orig))[:, :3]
    plt.plot(pcl[:, 0], pcl[:, 2], 'o')
    plt.axis('scaled')

    plt.savefig(f"{vis_path}_side.png")

def align_linefit(pcl_2d, residual_threshold=0.05):
    """ Extract the line with greatest support using RANSAC and align to y axis
    """
    # Find slope of line with greatest support
    ransac = linear_model.RANSACRegressor(residual_threshold=0.05)
    ransac.fit(pcl_2d[:, :1], pcl_2d[:, 1:])
    m = ransac.estimator_.coef_[0, 0]
    # Form rotation matrix
    r1 = np.stack((1.0, m), axis=0)
    r1 = r1 / np.linalg.norm(r1)
    r2 = np.sqrt(1.0 / (1.0 + (r1[0] / r1[1]) ** 2))
    r2 = np.stack((r2, -r2 * r1[0] / r1[1]), axis=0)
    R = np.stack((r2, r1), axis=-1)
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    R = R.transpose()
    R1 = R[0, :]
    R2 = R[1, :]
    R = np.stack([R2, -R1], axis=0) # rotate by 90deg clockwise
    inlier_indices = ransac.inlier_mask_
    print(f'num line fit inliers: {sum(ransac.inlier_mask_)}')

    # If the selected edge was at the front or back of the car, rotate by 90
    pcl_rot = np.einsum('ji,ni->nj', R, pcl_2d)
    width_x = pcl_rot[:, 0].max() - pcl_rot[:, 0].min()
    width_y = pcl_rot[:, 1].max() - pcl_rot[:, 1].min()
    if width_x < width_y:
        r = Rotation.from_rotvec(-np.pi/2 * np.array([0, 0, 1])).as_matrix()[:2, :2]
        R = r @ R
    return R, inlier_indices

def fit_3d_box(pcl_orig, co3d, vis_path="", ransac_threshold=0.05):
    R = estimate_ground_plane(co3d)
    T1 = np.eye(4, dtype=np.float32)
    T1[:3, :3] = R

    # pcl = np.einsum('ji,ni->nj', scale, homogenise_np(pcl))[:, :3]
    pcl = np.einsum('ji,ni->nj', R, pcl_orig)

    # points in the bottom may extend beyond the object bounding box
    # we exclude them for computing of object extend in the XY plane
    remove_bottom_10_perc = True
    if remove_bottom_10_perc:
        z = pcl[:, 2]
        print("percentile", np.percentile(z, 20), np.percentile(z, 80))
        z_min = z.min()
        z_max = z.max()
        z_cutoff = z_min + 0.1 * (z_max-z_min)
        mask = z >= z_cutoff
        print(z_cutoff, np.count_nonzero(mask))
        pcl = pcl[mask, :]
        pcl_orig_wo_bottom = pcl_orig[mask, :]
    else:
        pcl_orig_wo_bottom = pcl_orig

    pcl_2d = pcl[:, :2]
    hull = ConvexHull(pcl_2d)
    hull_points = pcl_2d[hull.vertices, :]

    # Align dominant line in point cloud to y axis, using RANSAC
    # Overly preferences dense regions, so may fail in unusual cases
    R, inlier_indices = align_linefit(pcl_2d, residual_threshold=ransac_threshold)

    # Or fit box to convex hull of points
    # R = fit_box_2d(hull_points)

    # Or enforce that the (largest) eigenvector of the covariance matrix
    # is parallel to the y axis
    # # cov = np.cov(pcl_2d, rowvar=False)
    # cov = MinCovDet().fit(pcl_2d).covariance_ # robust covariance
    # _, v = np.linalg.eigh(cov) # in ascending order of eigenvalues
    # R = v
    # if np.linalg.det(R) < 0:
    #     R[:, 0] = -R[:, 0]
    # R = R.transpose()
    # R1 = R[0, :]
    # R2 = R[1, :]
    # R = np.stack([R2, -R1], axis=0) # rotate by 90deg clockwise

    T2 = np.eye(4, dtype=np.float32)
    T2[:2, :2] = R

    rotated_pcl = np.einsum('ji,ni->nj', T2 @ T1, homogenise_np(pcl_orig))[:, :3]
    rotated_pcl_wo_bottom = np.einsum('ji,ni->nj', T2 @ T1, homogenise_np(pcl_orig_wo_bottom))[:, :3]

    min_x, min_y, _ = np.min(rotated_pcl_wo_bottom, axis=0)
    max_x, max_y, _ = np.max(rotated_pcl_wo_bottom, axis=0)

    _, _, min_z = np.min(rotated_pcl, axis=0)
    _, _, max_z = np.max(rotated_pcl, axis=0)

    rect = np.array([min_x, max_x, min_y, max_y, min_z, max_z])
    center = - (rect[[0, 2, 4]] + rect[[1, 3, 5]]) / 2.0

    plt.clf()
    vis_bev_box_fit(pcl_2d, hull_points, R, center[:2], rect, vis_path)
    
    T3 = np.eye(4, dtype=np.float32)
    T3[:3, 3] = center

    T = T3 @ T2 @ T1

    size = rect[[1, 3, 5]] - rect[[0, 2, 4]]

    plt.clf()
    vis_side_box_fit(pcl_orig, T, size, vis_path)

    return T, size


def fit_box(category, case):
    print("case:", case)
    cwd = os.getcwd()

    init_cfg = OmegaConf.create({
        "config" : {"exp_name": "co3d/tmp"},
        "dataset": {
            "data_extra_dir": "/work/eldar/data/datasets/co3d_extra_new_heuristic",
            "category": category,
            "instance": case,
            "use_auto_box": False
        },
    })
    cfg = construct_config("config/config.yaml", init_cfg)

    dataset_cfg = cfg.dataset
    dataset = dataset_factory(dataset_cfg)(dataset_cfg, device="cpu")

    category = cfg.dataset.category
    instance_id = cfg.dataset.instance

    pcl = np.squeeze(dataset.point_cloud_xyz.numpy())

    dist = np.linalg.norm(pcl, axis=1)
    dist_threshold = 4
    valid = dist <= dist_threshold
    pcl = pcl[valid, :]

    inlier_indices = filter_pointcloud(pcl, distance_threshold=0.2, max_points=20000)
    pcl = pcl[inlier_indices]

    vis_path = os.path.join(cfg.dataset.data_extra_dir, "vis_box_fit", category)
    Path(vis_path).mkdir(parents=True, exist_ok=True)
    vis_path = f"{vis_path}/{instance_id}"

    T, box_size = fit_3d_box(pcl, dataset.co3d, vis_path)

    pcl = np.einsum('ji,ni->nj', T, homogenise_np(pcl))[:, :3]

    out_base_dir = os.path.join(cfg.dataset.data_extra_dir, category)
    out_dir = os.path.join(out_base_dir, instance_id)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print("out_dir", out_dir)
    np.save(os.path.join(out_dir, "alignment"), {
        "T": T,
        "box_size": box_size
    })

    os.chdir(cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default='')
    parser.add_argument('--category', type=str, default='car')
    
    args = parser.parse_args()

    fit_box(args.category, args.instance)
