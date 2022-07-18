import startup

import argparse
import numpy as np
import torch
from skimage.transform import downscale_local_mean

from models.factory import dataset_factory
from util.coord import homogenise_np
from util.test_video import generate_eval_video_cameras
from util.webvis import default_camera, start_http_server
import scenepic as sp
from models.camera import PerspectiveCamera

from omegaconf import OmegaConf


"""
Code borrowed from:
https://github.com/Kai-46/nerfplusplus/blob/master/camera_visualizer/visualize_cameras.py
"""

def get_camera_frustum(camera, frustum_length=0.5, color=[0., 1., 0.]):
    C2W = camera.get_pose_matrix().squeeze().cpu().numpy()
    K = camera.get_intrinsics().squeeze().cpu().numpy()
    H, W = camera.get_image_size()[0].cpu().numpy()

    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def vis_pointcloud(scene, pts, rgb):
    mesh = scene.create_mesh("mesh")
    mesh.shared_color = np.array([0.7, 0.7, 0.7])
    
    mesh.add_cube()
    mesh.apply_transform(sp.Transforms.Scale(0.002)) 
    mesh.enable_instancing(positions=pts,
                           colors=rgb)
    return mesh


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors
    return merged_points, merged_lines, merged_colors


def visualise_cameras_3d(scene, cameras, color, frustum_size=0.5, nth=1):
    frustums = [get_camera_frustum(cam, frustum_size, color) for cam in cameras[::nth]]
    points, lines, colors = frustums2lineset(frustums)
    lines = lines.astype(np.int64)

    mesh = scene.create_mesh(shared_color=sp.Color(color[0], color[1], color[2]))
    mesh.add_lines(
        start_points=points[lines[:, 0], :],
        end_points=points[lines[:, 1], :]
    )

    return mesh


def visualise_virtual_test_cameras(cfg, dataset, color=[0., 0., 1.]):
    bbox_scale = torch.from_numpy(dataset.bbox_scale_transform)
    train_cameras = dataset.get_cameras()

    traj_radius = 2.0 / bbox_scale[0, 0]

    test_cameras = generate_eval_video_cameras(cfg, train_cameras, traj_radius, n_eval_cams=50)
    test_cameras = [PerspectiveCamera.from_pytorch3d(cam) for cam in test_cameras]
    test_cameras = [cam.left_transformed(bbox_scale) for cam in test_cameras]
    test_cameras_vis = visualise_cameras_3d(test_cameras, color)
    return test_cameras_vis


def visualise_images(scene, dataset, frustum_size=0.5, nth=1):
    train_cameras = dataset.get_cameras()

    meshes = []
    for k in range(0, len(train_cameras), nth):
        texture_id = f"frame_{k:02}"
        image1 = scene.create_image(image_id=texture_id)
        im_data = dataset.images[k].numpy()
        im_data_scaled = downscale_local_mean(im_data, (4, 4, 1))
        image1.from_numpy(im_data_scaled)

        cam = train_cameras[k]
        stuff = get_camera_frustum(cam, frustum_size, [0., 1., 0.])
        frustum_points = stuff[0]
        frustum_image_points = frustum_points[1:, :]

        mesh = scene.create_mesh(texture_id=texture_id)
        mesh.double_sided = True
        mesh.add_mesh_without_normals(
            frustum_image_points,
            np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),
            uvs=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        )
        meshes.append(mesh)

    return meshes


def main(args):
    instance, nth, frustum_size = args.instance, args.nth, args.frustum_size
    
    device = torch.device('cpu')

    init_cfg = OmegaConf.create({
        "dataset": {
            "category": args.category,
            "instance": instance
        },
    })

    cfg = OmegaConf.merge(OmegaConf.load("config/config.yaml"), init_cfg)

    dataset_cfg = cfg.dataset
    ds = dataset_factory(dataset_cfg)(dataset_cfg, device=device)

    pts = ds.point_cloud_xyz.squeeze().numpy()
    rgb = ds.point_cloud_rgb.squeeze().numpy()
    if ds.global_alignment is not None:
        canonical_alignment = ds.global_alignment.cpu().numpy()
        pts = np.einsum('ji,ni->nj', canonical_alignment, homogenise_np(pts))[:, :3]

        mask = np.all(
            np.concatenate(
                [pts >= ds.object_bbox_min, pts <= ds.object_bbox_max],
                axis=1),
            axis=1)
        # pts = pts[mask, :]
        # rgb = rgb[mask, :]

    RED = [1., 0., 0.]
    GREEN = [0., 1., 0.]
    BLUE = [0., 0., 1.]

    train_cameras = ds.get_cameras()

    scene = sp.Scene()
    point_cloud = vis_pointcloud(scene, pts, rgb)

    main = scene.create_canvas_3d(width=1200, height=800,
                                  shading=sp.Shading(bg_color=sp.Colors.White))
    main.camera = default_camera()

    image_meshes = visualise_images(scene, ds, frustum_size=frustum_size, nth=nth)

    all_cameras_vis = [visualise_cameras_3d(scene, train_cameras, GREEN,
                                            frustum_size=frustum_size, nth=nth)]

    all_meshes = [point_cloud]+all_cameras_vis+image_meshes

    if args.vis_axes:
        axes = scene.create_mesh(f"axes")
        axes.add_coordinate_axes()
        all_meshes += [axes]

    frame1 = main.create_frame(meshes=all_meshes)

    # scene.link_canvas_events(main)
    filename = "index.html"
    scene.save_as_html(filename, title=instance)

    start_http_server(".", 8888) # cfg.visualisation.port


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise training camera frustums and images.')
    parser.add_argument(
        '--instance', 
        type=str,
        default="157_17286_33548",
        help=("Instance in CO3D dataset")
    )
    parser.add_argument(
        '--category', 
        type=str,
        default="car",
        help=("CO3D category")
    )
    parser.add_argument(
        '--nth', 
        type=int,
        default=4,
        help=("Number of rames to skip.")
    )
    parser.add_argument(
        '--frustum_size',
        type=float,
        default=1.0,
        help="Default frustum size",
    )
    parser.add_argument(
        '--structured-split',
        action='store_true',
        help=("Visualise cameras in the structured split only.")
    )
    parser.add_argument(
        '--x-is-reflection-axis',
        action='store_true'
    )
    parser.add_argument('--vis-axes', action='store_true')
    args = parser.parse_args()
    main(args)