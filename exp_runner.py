import startup

from itertools import chain
import subprocess
import socket
import os
import sys
import time
import logging
from itertools import chain, cycle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
import trimesh
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import imageio

import util.config as config
from logger.factory import create_logger

from models.factory import load_datasets
from models.renderer_factory import factory as renderer_factory
from models.raysampler import sample_random_rays, rays_to_world, sample_image_pixels, sample_rays_on_grid
from models.camera import CameraManager, PerspectiveCamera
from models.transform import TransformManager, SymmetryManager
from models.dataset_wrapper import DatasetWrapper
from models.camera import RayBundle

from util.test_video import generate_eval_video_cameras
from util.checkpoint import delete_old_checkpoints
from util.coord import transform_points
from util.webvis import start_http_server, vis_mesh
from util.filesystem import mkdir_shared


def find_latest_checkpoint(ckpt_dir):
    model_list_raw = os.listdir(ckpt_dir)
    model_list = []
    for model_name in model_list_raw:
        if model_name[-3:] == 'pth':
            model_list.append(model_name)
    model_list.sort()
    latest_model_name = model_list[-1]
    return latest_model_name


class Runner:
    def __init__(self, cfg):
        self.device = torch.device('cuda')

        # Configuration
        self.cfg = cfg

        mode, is_continue = cfg.mode, cfg.is_continue

        self.fp16 = cfg.fp16 # Half precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        self.base_exp_dir = os.getcwd()
        self.dataset, self.val_dataset = load_datasets(cfg)

        self.camera_manager = CameraManager(self.dataset.cameras, cfg)
        self.learn_symmetry = cfg.model.renderer.learn_symmetry
        if self.learn_symmetry:
            self.transform_manager = SymmetryManager(cfg)
        else:
            self.transform_manager = TransformManager(cfg)

        self.iter_step = 0

        # Weights
        self.is_continue = is_continue
        self.mode = mode
        
        self.end_iter = cfg.train.end_iter

        self.renderer = renderer_factory(cfg)(
            cfg,
            bbox_min=self.dataset.object_bbox_min,
            bbox_max=self.dataset.object_bbox_max
        )
        self.renderer.set_device(self.device)

        # Networks
        nets = self.renderer.get_networks()
        for key, net in nets.items():
            nets[key] = net.to(self.device)
        
        # Pretrain SDF network
        cfg.model.sdf_network.pretrain_sdf = getattr(cfg.model.sdf_network, "pretrain_sdf", False)
        if cfg.model.sdf_network.pretrain_sdf:
            nets['sdf_network'].pretrain(self.device)
        
        if cfg.mode == 'train':
            self.create_optimizer()
        nets["camera_manager"] = self.camera_manager
        nets["transform_manager"] = self.transform_manager
        self.networks = nets

        # Load checkpoint
        init_model = cfg.train.init_model
        if is_continue:
            ckpt_dir = self.get_checkpoint_dir()
            if cfg.test.checkpoint == - 1:
                checkpoint_file = find_latest_checkpoint(ckpt_dir)
            else:
                checkpoint_file = self.get_checkpoint_name(cfg.test.checkpoint)
            logging.info('Find checkpoint: {}'.format(checkpoint_file))
            if checkpoint_file is not None:
                checkpoint = ckpt_dir.joinpath(checkpoint_file)
                self.load_checkpoint(checkpoint)
        elif init_model:
            ckpt_dir = Path(init_model, cfg.dataset.instance, 'checkpoints')
            latest_model_name = find_latest_checkpoint(ckpt_dir)
            checkpoint = ckpt_dir.joinpath(latest_model_name)
            logging.info(f"Initialising model from: {checkpoint}")
            logging.info(f"Loading networks: {cfg.train.init_networks}")
            self.load_checkpoint(checkpoint, True, cfg.train.init_networks)

    def create_optimizer(self):
        cfg_t = self.cfg.train
        nets = self.renderer.get_networks()

        variance_group = ["variance_network"]
        slow_group = cfg_t.ramp_lr_nets
        groups = [variance_group, slow_group]
        # the rest
        constant_group = [x for x in nets.keys() if x not in chain(*groups)]
        groups += [constant_group]
        group_names = ["variance_group", "slow_group", "constant_group"]

        lrs = {
            "slow_group": cfg_t.learning_rate,
            "constant_group": cfg_t.learning_rate,
            "variance_group": cfg_t.learning_rate_variance
        }

        param_groups = []
        for k, group in enumerate(groups):
            name = group_names[k]
            params_to_train = list(
                chain.from_iterable(
                    nets[name].parameters() for name in group
                )
            )
            param_groups.append({
                "params": params_to_train,
                "lr": lrs[name],
                "name": name
            })

        if (self.learn_symmetry and not cfg_t.freeze_symmetry_transform): # also learn params if estimating ground plane
            transform_params = self.transform_manager.parameters()
            transform_lr = cfg_t.learning_rate_symmetry
            param_groups += [{
                "params": transform_params,
                "lr": transform_lr,
                "name": "global_alignment"
            }]

        self.optimizer = torch.optim.Adam(param_groups, lr=cfg_t.learning_rate)

    def train_single_view(self):
        cfg = self.cfg
        cfg_t = cfg.train
        self.logger = create_logger(cfg)
        print("Experiment name:", cfg.config.exp_name)

        self.update_learning_rate()
        res_step = cfg_t.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        dataset = self.dataset

        for iter_i in tqdm(range(res_step)):
            image_idx = image_perm[self.iter_step % len(image_perm)]
            camera = self.camera_manager.get_camera(image_idx).cuda()
            
            xy = sample_random_rays(camera, cfg_t.batch_size, self.device)
            rays = rays_to_world(camera, xy)
            true_rgb = sample_image_pixels(dataset.images[image_idx], rays)

            self.training_step(rays, true_rgb)

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def train(self):
        # save checkpoint with the initial network parameters
        # for subsequent reinitialization
        self.train_loop()
        self.save_checkpoint()

    def train_loop(self):
        cfg = self.cfg
        cfg_t = cfg.train
        self.logger = create_logger(cfg)
        print("Experiment name:", cfg.config.exp_name)

        self.update_learning_rate()
        res_step = cfg_t.end_iter - self.iter_step
        dataset = self.dataset
        ds_wrapper = DatasetWrapper(cfg, dataset)
        data_loader = DataLoader(ds_wrapper,
                                 shuffle=True,
                                 num_workers=4,
                                 batch_size=cfg.train.batch_size,
                                 pin_memory=True)

        for batch in tqdm(cycle(data_loader), total=res_step):
            batch = batch.cuda()
            frame_idx = batch[:, 0].type(torch.int64)
            camera = self.camera_manager.get_cameras(frame_idx)
            xy = RayBundle(xys=batch[:, 1:3])
            rays = rays_to_world(camera, xy)
            true_rgb = batch[:, 3:]

            self.training_step(rays, true_rgb)

            if self.iter_step == cfg_t.end_iter:
                break

    def training_step(self, rays, true_rgb):
        cfg = self.cfg
        cfg_t = cfg.train
        batch_size = true_rgb.shape[0]
        renderer = self.renderer
        dataset = self.dataset

        renderer.set_training_step(self.iter_step)
        near, far = dataset.near_far_from_sphere(rays)

        background_rgb = None
        if cfg_t.use_white_bkgd:
            background_rgb = torch.ones([1, 3])

        mask = torch.ones((batch_size, 1), dtype=torch.float32).cuda()

        inputs = self.form_extra_inputs()

        if getattr(cfg_t, "sfm_supervision_weight", 0) > 0:
            pcl = dataset.point_cloud_xyz_canonical
            transform = self.camera_manager.get_learnable_4x4_transform().squeeze() # Fails now
            perm = torch.randperm(pcl.shape[0])
            sfm_batch_size = cfg_t.sfm_batch_size
            idx = perm[:sfm_batch_size]
            pcl = pcl[idx, ...]
            pcl = pcl.cuda()
            pcl = transform_points(transform, pcl)
            inputs["points_xyz"] = pcl
        
        # Initial mesh and renders
        if self.iter_step == 0:
            self.save_intermediate_mesh()

        with torch.cuda.amp.autocast(enabled=self.fp16):
            render_out = renderer.render(rays, near, far,
                                        inputs=inputs,
                                        background_rgb=background_rgb,
                                        cos_anneal_ratio=self.get_cos_anneal_ratio())

            loss, to_log = renderer.evaluate_loss(render_out, true_rgb, mask)

        self.optimizer.zero_grad()
        # self.optimizer2.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        # self.scaler.step(self.optimizer2)
        self.scaler.update()

        self.iter_step += 1

        # log learned camera parameters
        to_log.update({
            't/x': self.transform_manager.t_[0, 0],
            't/y': self.transform_manager.t_[0, 1],
            't/z': self.transform_manager.t_[0, 2],
            't/norm': self.transform_manager.t_[0, :].norm(),
            'r/x': self.transform_manager.r_[0, 0],
            'r/y': self.transform_manager.r_[0, 1],
            'r/z': self.transform_manager.r_[0, 2],
            'r/norm': self.transform_manager.r_[0, :].norm(),
        })

        self.logger.log(to_log, self.iter_step)

        if self.iter_step % cfg_t.report_freq == 0:
            print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

        if self.iter_step % cfg_t.save_freq == 0:
            self.save_checkpoint()

        if self.iter_step % cfg_t.val_mesh_freq == 0:
            self.save_intermediate_mesh()

        if self.iter_step % cfg.train.render_views_freq == 0 or self.iter_step == 10000:
            torch.cuda.empty_cache()
            renderer.set_inference_mode(True)
            images = self.render_test_video_impl(4, 4)
            renderer.set_inference_mode(False)
            for img_idx, img in enumerate(images):
                img = img.astype(np.float32) / 255
                self.logger.upload_image(f"render/{img_idx:01}", img)
            torch.cuda.empty_cache()

        self.update_learning_rate()

    def form_extra_inputs(self):
        inputs = {
            "ground_plane_offset": self.dataset.get_ground_plane_z(), # fixed offset from transformed origin
            "transform_manager": self.transform_manager,
        }
        return inputs

    def get_cos_anneal_ratio(self):
        cfg_t = self.cfg.train
        if cfg_t.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / cfg_t.anneal_end])

    def update_learning_rate(self):
        cfg_t = self.cfg.train
        warm_up_end = cfg_t.warm_up_end

        if self.iter_step < warm_up_end:
            t = self.iter_step / warm_up_end
            ramp_lr_end = 1.0
            learning_factor = cfg_t.ramp_lr_start * (1.0 - t) + t * ramp_lr_end
            learning_factors = {
                "slow_group": learning_factor,
                "constant_group": 1.0,
                "global_alignment": learning_factor,
                "variance_group": learning_factor
            }
        else:
            alpha = cfg_t.learning_rate_alpha
            progress = (self.iter_step - warm_up_end) / (self.end_iter - warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            learning_factors = {
                "slow_group": learning_factor,
                "constant_group": learning_factor,
                "global_alignment": learning_factor,
                "variance_group": learning_factor
            }

        for g in self.optimizer.param_groups:
            g['lr'] = cfg_t.learning_rate * learning_factors[g["name"]]

    def load_checkpoint(self, checkpoint_file, init_model=False, network_names=None):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        names = self.networks.keys() if network_names is None else network_names
        for k in names:
            if k in checkpoint:
                if k == "camera_manager" and not hasattr(self.networks[k], "r_") and "r_" in checkpoint[k]: # some neus experiments have dummy r_ 
                    continue
                self.networks[k].load_state_dict(checkpoint[k])
        if not init_model and self.cfg.mode == 'train':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.iter_step = checkpoint['iter_step']
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler']) 
        logging.info('End')

    def get_checkpoint_dir(self):
        return Path(self.base_exp_dir, 'checkpoints')

    def save_checkpoint(self):
        checkpoint = {k: v.state_dict() for k, v in self.networks.items()}
        checkpoint.update({
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'scaler': self.scaler.state_dict()
        })
        ckpt_dir = self.get_checkpoint_dir()
        mkdir_shared(ckpt_dir)
        ckpt_file = ckpt_dir.joinpath(self.get_checkpoint_name(self.iter_step))
        torch.save(checkpoint, str(ckpt_file.resolve()))
        if not self.cfg.train.keep_old_checkpoints:
            delete_old_checkpoints(ckpt_dir)

    def get_checkpoint_name(self, iter_step):
        return 'ckpt_{:0>6d}.pth'.format(iter_step)

    def render_test_video_impl(self, resolution_level, num_cameras, out_dir=None):
        ds = self.dataset
        cfg = self.cfg
        test_batch_size = cfg.test.batch_size
        device = self.device

        inputs = self.form_extra_inputs()

        bbox_scale = torch.from_numpy(ds.bbox_scale_transform)
        traj_radius = 2.0 / bbox_scale[0, 0]
        test_cameras = generate_eval_video_cameras(cfg, ds.get_cameras(), traj_radius, n_eval_cams=num_cameras)
        test_cameras = [PerspectiveCamera.from_pytorch3d(cam) for cam in test_cameras]
        test_cameras = [cam.left_transformed(bbox_scale) for cam in test_cameras]
        for cam in test_cameras:
            cam.cuda()

        if self.learn_symmetry:
            _, T_inv = self.transform_manager.get_transform(return_inverse=True)
            test_cameras = [cam.left_transformed(T_inv.squeeze()) for cam in test_cameras]

        images = []
        for frame_idx, cam in enumerate(tqdm(test_cameras)):
            xys, H, W = sample_rays_on_grid(cam, resolution_level, device)
            rays = rays_to_world(cam, xys)
            rays = rays.split(test_batch_size)
            out_rgb = []
            for rays_batch in rays:
                near, far = self.dataset.near_far_from_sphere(rays_batch)
                background_rgb = torch.ones([1, 3], device=device) if cfg.test.white_bkgd else None
                render_out = self.renderer.render(rays_batch, near, far,
                                                  inputs=inputs,
                                                  background_rgb=background_rgb,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())
                if cfg.test.rendering_output != "full":
                    color_out = render_out['custom_color']
                else:
                    color_out = render_out['color']
                out_rgb.append(color_out.detach().cpu().numpy())

            img = (np.concatenate(out_rgb, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
            img = img.astype(np.uint8)
            images.append(img)
            if out_dir is not None:
                imageio.imwrite(out_dir.joinpath(f"{frame_idx:04}.png"), img)
        return images

    def render_test_video(self):
        cfg = self.cfg
        self.renderer.set_inference_mode(True)
        out_dir = Path(cfg.test.video_out_dir)
        mkdir_shared(out_dir)
        print(f"Writing frames8 to: {str(out_dir.resolve())}")
        resolution_level = cfg.test.nvs_resolution

        num_cameras = cfg.test.num_cams
        if not out_dir.joinpath(f"{num_cameras-1:04}.png").exists():
            self.render_test_video_impl(resolution_level, num_cameras, out_dir)
        else:
            print("frames already rendered, skipping")
        
        self.gen_video_file(out_dir)

    def gen_video_file(self, out_dir):
        out_video_file = out_dir.joinpath(f"video.mp4")
        # use ffmpeg from conda
        ffmpeg_exec = f"{os.path.dirname(sys.executable)}/ffmpeg"
        print("ffmpeg exec", ffmpeg_exec)
        cmd = [ffmpeg_exec, "-y", "-f", "image2", "-i", out_dir.joinpath("%04d.png"), "-b:v", "6000k", "-c:v", "libopenh264", out_video_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        print("Generated video file", out_video_file.resolve())

    def make_mesh_dir(self):
        path = Path("meshes")
        mkdir_shared(path)
        return path

    def out_mesh_file(self):
        path = self.make_mesh_dir()
        return path.joinpath("mesh.ply")

    def visualise_mesh(self, world_space=False, resolution=None, export=False, out_filename=None):
        threshold = self.cfg.test.mcube_threshold
        if resolution is None:
            resolution = self.cfg.test.mcube_resolution
        dataset = self.dataset

        bound_min = torch.tensor(dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(dataset.object_bbox_max, dtype=torch.float32)
        if not self.cfg.test.nvs_cut_box:
            bound_min = torch.min(bound_min).repeat(3)
            bound_max = torch.max(bound_max).repeat(3)

        learnt_t = self.transform_manager.t_.detach().to(bound_min.device).squeeze() 
        learnt_t[1] = 0
        bound_min += learnt_t # TODO: verify correctness of sign here
        bound_max += learnt_t

        renderer = self.renderer
        renderer.set_inference_mode(False)

        inputs = self.form_extra_inputs()
        renderer.setup_rendering(inputs)

        vertices, triangles =\
            renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

        if vertices.shape[0] == 0 or triangles.shape[0] == 0:
            return None

        if world_space:
            vertices = vertices * dataset.scale_mats_np[0][0, 0] + dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)

        if self.cfg.test.vis_symm_plane:
            verts_symm, tris_symm = self.transform_manager.vis_symmetry_plane()
            verts_symm = verts_symm.numpy().astype(np.float64)
            tris_symm = tris_symm.numpy().astype(np.uint64)
            mesh_symm = trimesh.Trimesh(verts_symm, tris_symm)
            mesh = trimesh.util.concatenate([mesh, mesh_symm])

        if export:
            if not out_filename:
                out_filename = self.out_mesh_file()
            if out_filename.exists():
                os.remove(out_filename)
            mesh.export(out_filename)

        return mesh

    def save_intermediate_mesh(self):
        cfg = self.cfg
        cfg_t = cfg.train
        vis_bbox = cfg.visualisation.show_bounding_box
        mcube_resolution = 128
        torch.cuda.empty_cache()
        mesh = self.visualise_mesh(export=True, resolution=mcube_resolution)
        if mesh is not None:
            if cfg_t.save_all_meshes:
                html_file = f"index_{self.iter_step:06}.html"
            else:
                html_file = "index.html"
            path = self.make_mesh_dir()
            html_path = path.joinpath(html_file)
            if html_path.exists():
                os.remove(html_path)
            bbox_size = self.dataset.raw_bbox_max if vis_bbox else None
            vis_mesh(str(html_path),
                     mesh,
                     half_bbox_size=bbox_size,
                     vis_axes=cfg.visualisation.show_axes)
            self.logger.upload_file("mesh", str(html_path))
            torch.cuda.empty_cache()
        else:
            logging.info('Mesh generation failed')


@config.main(default_config="config/config.yaml")
def main(cfg: DictConfig) -> None:
    print(f"HOST: {socket.gethostname()}")

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    torch.cuda.set_device(cfg.gpu)
    print("GPU:", cfg.gpu)
    if cfg.mode in ['visualise_mesh', 'test_video']:
        cfg.is_continue = True
    runner = Runner(cfg)

    if cfg.mode == 'train':
        if cfg.train.multi_view_batch:
            runner.train()
        else:
            runner.train_single_view()
        runner.visualise_mesh(export=True)
    elif cfg.mode == 'visualise_mesh':
        mesh = runner.visualise_mesh(export=True)
        if cfg.test.web_vis:
            path = Path("meshes")
            mkdir_shared(path)
            html_path = path.joinpath("index.html")
            if html_path.exists():
                os.remove(html_path)
            vis_mesh(str(html_path), mesh)
            start_http_server(path, cfg.visualisation.port)
    elif cfg.mode == 'test_video':
        runner.render_test_video()


if __name__ == '__main__':
    main()