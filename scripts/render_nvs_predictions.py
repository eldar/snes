import startup

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import imageio
from tqdm import tqdm
from omegaconf import OmegaConf
from models.renderer import compute_integral

from util.config import construct_config
from util.visualise_image import normalize_depth_for_display

from models.camera import CameraManager
from models.raysampler import rays_to_world, sample_rays_on_grid, rays_to_cam
from exp_runner import Runner

from util.data import iterate_exp
from util.filesystem import mkdir_shared


def render_views(runner, resolution_level, args): # if sum(weights) < transparency_threshold, assume transparent
    self = runner
    ds = runner.val_dataset
    test_batch_size = 256
    n_sdf_samples = 128
    device = runner.device
    transparency_threshold = args.transparency_threshold
    print("transparency threshold", transparency_threshold)

    self.renderer.set_inference_mode(True)

    cameras = ds.get_cameras()
    num_cameras = len(cameras)

    sub_dir = Path(args.result_dir)
    mkdir_shared(sub_dir)

    out_render_dir = sub_dir.joinpath(f"eval_renders")
    mkdir_shared(out_render_dir)

    out_mask_dir = sub_dir.joinpath(f"eval_masks")
    mkdir_shared(out_mask_dir)

    out_depth_dir = sub_dir.joinpath(f"eval_depths")
    mkdir_shared(out_depth_dir)

    out_depth_vis_dir = sub_dir.joinpath(f"eval_depth_vis")
    mkdir_shared(out_depth_vis_dir)

    def compute_max_depth(cam):
        xys, _, _ = sample_rays_on_grid(cam, resolution_level, device)
        rays_all = rays_to_world(cam, xys)
        _, far_all = ds.near_far_from_sphere(rays_all)
        max_depth = torch.max(far_all) * 2
        return max_depth

    for frame_idx in range(num_cameras):
        print(f"evaluate camera {frame_idx}, {frame_idx+1}/{num_cameras}")
        cam = ds.get_cameras()[frame_idx]
        cam = cam.cuda()
        max_depth = compute_max_depth(cam)
        xys, H, W = sample_rays_on_grid(cam, resolution_level, device)
        rays = rays_to_world(cam, xys)
        rays = rays.split(test_batch_size)
        
        # Get rays in camera coordinates
        rays_cam = rays_to_cam(cam, xys)
        rays_cam = rays_cam.split(test_batch_size)

        out_rgb = []
        out_mask = []
        out_depth = []
        for i, rays_batch in enumerate(tqdm(rays)):
            near, far = self.dataset.near_far_from_sphere(rays_batch)
            inputs = self.form_extra_inputs()
            background_rgb = torch.ones([1, 3], device=device) if self.cfg.test.white_bkgd else None
            render_out = self.renderer.render(rays_batch, near, far,
                                              inputs=inputs,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())
            color_out = render_out['color']
            out_rgb.append(color_out.detach().cpu().numpy())
            with_mask = 'mask_fg' in render_out
            if with_mask:
                out_mask.append(render_out['mask_fg'].detach().cpu().numpy())

            # Compute depths:
            # Normalise weights if above a cutoff threshold; foreground weights
            # should sum to 1 if the ray intersects a surface (no transparency allowed)
            # Alternative: take the first depth with weight above a threshold
            weights = render_out["weights"][:, :n_sdf_samples] # samples within sphere
            weights_sum = weights.sum(dim=-1, keepdim=True) # should be < 1
            weights = torch.where(weights_sum > transparency_threshold, weights / weights_sum, weights)
            z_vals = render_out["mid_z_vals"][:, :, None]
            dist = compute_integral(z_vals, max_depth, weights)
            r = dist * rays_cam[i].directions
            depth = r[..., [2]] # dot product with optical axis [0, 0, 1]
            depth = depth.abs() # assume all distances are positive (in front of camera)
            out_depth.append(depth.detach().cpu().numpy())

        filename = ds.filenames[frame_idx]
        basename = os.path.splitext(filename)[0]
        name = f"{basename}.png"

        img = (np.concatenate(out_rgb, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        imageio.imwrite(out_render_dir.joinpath(name), img)

        if with_mask:
            mask = (np.concatenate(out_mask, axis=0).reshape([H, W, 1]) * 255).clip(0, 255).astype(np.uint8)
            imageio.imwrite(out_mask_dir.joinpath(name), mask)
        
        depth_img = np.squeeze(np.concatenate(out_depth, axis=0).reshape([H, W, 1]))
        np.save(out_depth_dir.joinpath(basename), depth_img)

        depth_rgb = normalize_depth_for_display(depth_img) #, normalizer=1.0)
        imageio.imwrite(out_depth_vis_dir.joinpath(name), depth_rgb)

# "co3d/ablation/neus"
# "co3d/ablation/symmetric_ramp_lr"
# "lists/complete_360.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='co3d/eccv/neus_co3d_split')
    parser.add_argument('--instance-list', type=str, default='lists/co3d_testset.txt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--instance', type=str, default='')
    parser.add_argument('--transparency-threshold', type=float, default=0.1)
    parser.add_argument('--ckpt-iter', type=int, default=300000)
    parser.add_argument('--structured-split', action='store_true')
    parser.add_argument('--result-dir', type=str, default='.')
    parser.add_argument('--nvs_cut_box', action='store_true')
    parser.add_argument('--white_bkgd', action='store_true')
    args = parser.parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(args.gpu)

    instance_list = args.instance_list
    if args.instance:
        instance_list = [args.instance]

    resolution = 1

    init_cfg = {
        "test": {
            "render_foreground_mask": True,
            "nvs_cut_box": args.nvs_cut_box,
            "pre_sphere_transparency": True,
            "checkpoint": args.ckpt_iter,
            "white_bkgd": args.white_bkgd
        },
    }
    if args.structured_split:
        init_cfg.update({
            "dataset": {
                "split_file": "train_val_split_partial_view_filtered.pkl"
            }
        })

    iterator = iterate_exp(args.exp, instance_list, init_cfg, with_runner=True)
    for runner in iterator:
        render_views(runner, resolution, args)


if __name__ == '__main__':
    main()