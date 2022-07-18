import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes

from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from util.coord import inside_axis_aligned_box


@torch.no_grad()
def extract_fields(bound_min, bound_max, resolution, query_func, device=None):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

@torch.no_grad()
def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, device=None):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func, device)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    device = weights.device
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def compute_alpha(sdf, dists, dirs, gradients, inv_s, cos_anneal_ratio):
    num = sdf.shape[0]
    inv_s = inv_s.expand(num, 1)

    true_cos = (dirs * gradients).sum(-1, keepdim=True)

    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    # Estimate signed distances at section points
    estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
    estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    
    return alpha, c


def compute_transmittance(alpha):
    _1 = torch.ones([alpha.shape[0], 1], device=alpha.device)
    return alpha * torch.cumprod(torch.cat([_1, 1. - alpha + 1e-7], -1), -1)[:, :-1]


def compute_integral(input_color, background_rgb, weights, weights_sum=None):
    if weights_sum is None:
        weights_sum = weights.sum(dim=-1, keepdim=True)
    color = (input_color * weights[:, :, None]).sum(dim=1)
    if background_rgb is not None:    # Fixed background, usually black
        color = color + background_rgb * (1.0 - weights_sum)
    return color


def color_loss(color, gt, mask):
    mask_sum = mask.sum() + 1e-5
    color_error = (color - gt) * mask
    return F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum


class Renderer:
    def __init__(self, cfg, **kwargs):
        cfg_r = cfg.model.renderer
        self.cfg = cfg
        self.set_bounding_box(kwargs["bbox_min"], kwargs["bbox_max"])
        self.create_networks()
        self.n_samples = cfg_r.n_samples
        self.n_importance = cfg_r.n_importance
        self.n_outside = cfg_r.n_outside
        self.up_sample_steps = cfg_r.up_sample_steps
        self.perturb = cfg_r.perturb
        self.set_inference_mode(cfg.mode != "train")
        self.training_step = 0

    def create_networks(self):
        cfg = self.cfg
        if cfg.model.renderer.scale_input_coords:
            max_side = torch.max(self.bbox_max)
            scaler = max_side / self.bbox_max
        else:
            scaler = None
        networks = {
            'nerf': NeRF(cfg.model.nerf),
            'sdf_network': SDFNetwork(cfg.model.sdf_network, inputs_scale=scaler),
            'variance_network': SingleVarianceNetwork(cfg.model.variance_network),
            'color_network': RenderingNetwork(cfg.model.rendering_network)
        }

        self.nerf = networks['nerf']
        self.color_network = networks['color_network']
        self.sdf_network = networks['sdf_network']
        self.deviation_network = networks['variance_network']
        self.networks = networks

    def renderer_config(self):
        return self.cfg.model.renderer

    def set_bounding_box(self, bbox_min, bbox_max):
        def convert(x):
            return torch.from_numpy(x)[None, :].type(torch.float32).cuda()
        self.bbox_min = convert(bbox_min)
        self.bbox_max = convert(bbox_max)

    def set_device(self, device):
        self.device = device

    def get_networks(self):
        return self.networks

    def set_inference_mode(self, inference_mode):
        self.inference_mode = inference_mode
    
    def set_training_step(self, step):
        self.training_step = step

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape
        device = z_vals.device

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        sample_dist_t = torch.tensor([sample_dist], device=device)
        dists = torch.cat([dists, sample_dist_t.expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sigma = F.softplus(density.reshape(batch_size, n_samples)) # opacity
        alpha = 1.0 - torch.exp(-sigma * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
            'sigma': sigma,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        device = z_vals.device
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3))
            new_sdf = new_sdf.reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size, device=z_vals.device)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def preprocess_points(self, pts, dirs):
        return pts, dirs, dirs
    
    def evaluate_color(self, pts, gradients, dirs, feature_vector):
        out = self.color_network(pts, gradients, dirs, feature_vector)
        out = torch.sigmoid(out)
        return {
            "full_color": out
        }

    def compute_dists(self, z_vals, sample_dist):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        sample_dist = torch.tensor([sample_dist], dtype=torch.float32, device=dists.device)
        dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        return dists, mid_z_vals

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    inputs,
                    sample_dist,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape
        cfg = self.cfg

        is_test = self.inference_mode

        # Section length
        dists, mid_z_vals = self.compute_dists(z_vals, sample_dist)

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts, dirs, dirs_actual = self.preprocess_points(pts, dirs)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        dirs_actual = dirs_actual.reshape(-1, 3)

        sdf_output = self.sdf_network(pts)
        sdf = sdf_output["signed_distance"]
        feature_vector = sdf_output["feature"]

        gradients = self.sdf_network.gradient(pts).squeeze()
        colors = self.evaluate_color(pts, gradients, dirs, feature_vector)
        sampled_color = colors["full_color"].reshape(batch_size, n_samples, 3)
        
        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        alpha, c = compute_alpha(sdf, dists, dirs_actual, gradients, inv_s, cos_anneal_ratio)
        alpha = alpha.reshape(batch_size, n_samples)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        if cfg.test.nvs_cut_box:
            inside = self.limit_rendering_volume(pts).reshape(batch_size, n_samples)
            alpha = alpha * inside

        outputs = dict()

        if is_test and cfg.test.render_foreground_mask:
            weights_fg = compute_transmittance(alpha)
            mask_fg = weights_fg.sum(dim=-1, keepdim=True)
            outputs.update({
                'mask_fg': mask_fg
            })

        # Render with background
        if background_alpha is not None and background_rgb is None:
            outside_sphere = 1.0 - inside_sphere
            if self.inference_mode and cfg.test.pre_sphere_transparency:
                # Set all background_alpha before the first nonzero value of inside_sphere to zero, unless entire row is zero (does not intersect sphere)
                # These alphas should become zero during training anyway, if seen from other views
                # May prevent modelling of ground plane immediately outside sphere
                background_mask = ((torch.cumsum(inside_sphere, dim=-1) + (1.0 - inside_sphere.sum(dim=-1, keepdim=True)).clamp(min=0.0)) > 0).float()
                outside_sphere *= background_mask
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * outside_sphere
            # alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = compute_transmittance(alpha)
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = compute_integral(sampled_color, background_rgb, weights, weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        outputs.update({
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf_fine': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
        })

        return outputs

    def render(self, rays, near, far, inputs=dict(), perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        rays_o = rays.origins
        rays_d = rays.directions
        device = rays_o.device

        self.setup_rendering(inputs)

        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=device)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]], device=device)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3))
                sdf = sdf.reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']
            background_sigma = ret_outside['sigma']

        # Render core
        results = self.render_core(rays_o,
                                   rays_d,
                                   z_vals,
                                   inputs,
                                   sample_dist,
                                   background_rgb=background_rgb,
                                   background_alpha=background_alpha,
                                   background_sampled_color=background_sampled_color,
                                   cos_anneal_ratio=cos_anneal_ratio)

        weights = results['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        results.update({
            'z_vals': z_vals,
            'mid_z_vals': results['mid_z_vals'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
        })
        
        return results

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        query_func = lambda pts: -self.sdf_network.sdf(pts)
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=query_func,
                                device=self.device)

    def limit_rendering_volume(self, pts):
        return inside_axis_aligned_box(pts, self.bbox_min, self.bbox_max)

    def setup_rendering(self, inputs):
        if "ground_plane_offset" in inputs:
            self.ground_plane_offset = inputs["ground_plane_offset"] # fixed offset from transformed origin
        if "transform_manager" in inputs:
            self.transform_manager = inputs["transform_manager"]

    def evaluate_loss(self, render_out, true_rgb, mask):
        cfg = self.cfg

        color = render_out['color']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        mask_sum = mask.sum() + 1e-5

        # Loss
        color_error = (color - true_rgb) * mask
        color_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        psnr = 20.0 * torch.log10(1.0 / (((color - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        eikonal_loss = gradient_error

        loss = color_loss + eikonal_loss * cfg.train.eikonal_weight

        if cfg.train.mask_weight > 0:
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            loss += mask_loss * cfg.train.mask_weight
        
        log = {
            'Loss/color': color_loss,
            'Loss/eikonal': eikonal_loss,
            'Statistics/s_val': s_val.mean(),
            'Statistics/cdf': (cdf_fine[:, :1] * mask).sum() / mask_sum,
            'Statistics/weight_max': (weight_max * mask).sum() / mask_sum,
            'Statistics/psnr': psnr,
        }

        log.update({
            'Loss/total': loss,
        })

        return loss, log
