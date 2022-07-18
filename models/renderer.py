import torch
import torch.nn.functional as F
import numpy as np
import mcubes

from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from util.coord import inside_axis_aligned_box


#####################
# Helper functions
#####################
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
    """ Obtain samples from a probability density function.
    Implementation from nerf-pytorch: github.com/yenchenlin/nerf-pytorch
    """
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
    """ Compute the discrete opacity given the signed distances.
    Implementation from NeuS: https://github.com/Totoro97/NeuS
    """
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


def reflected_rays(dirs, normals):
    v = dirs
    n = F.normalize(normals, dim=-1)
    refl = v - 2 * (v * n).sum(-1, keepdim=True) * n
    return refl


def compute_transmittance(alpha):
    """ Compute the transmittance from the discrete opacity.
    """
    _1 = torch.ones([alpha.shape[0], 1], device=alpha.device)
    return alpha * torch.cumprod(torch.cat([_1, 1. - alpha + 1e-7], -1), -1)[:, :-1]


def compute_integral(input_color, background_rgb, weights, weights_sum=None):
    """ Compute the (discretised) integral given the weights and colors.
    """
    if weights_sum is None:
        weights_sum = weights.sum(dim=-1, keepdim=True)
    color = (input_color * weights[:, :, None]).sum(dim=1)
    if background_rgb is not None:    # Fixed background, usually black
        color = color + background_rgb * (1.0 - weights_sum)
    return color


def combine_color_with_bg(color, background_color, inside_sphere):
    """ Combine foreground and background colors according to a mask.
    """
    n_samples = color.shape[1]
    color = color * inside_sphere[:, :, None] +\
                    background_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
    return torch.cat([color, background_color[:, n_samples:]], dim=1)


def compute_color(albedo, reflectivity, diffuse, specular, clamp=True):
    """ Combine appearance components into full color.
    """
    full_color = diffuse * albedo + reflectivity * specular
    if clamp:
        full_color = torch.clamp(full_color, min=0.0, max=1.0)
    else:
        full_color = torch.sigmoid(full_color)
    return full_color


def compute_diffuse_color(albedo, diffuse, clamp=True):
    diffuse_color = diffuse * albedo
    if clamp:
        diffuse_color = torch.clamp(diffuse_color, min=0.0, max=1.0)
    else:
        diffuse_color = torch.sigmoid(diffuse_color)
    return diffuse_color


def raw_to_val(raw):
    val = {}
    if 'diffuse' in raw:
        val['diffuse'] = 2.0 * torch.sigmoid(raw['diffuse']) # (0, 2), init: 1
    if 'albedo' in raw:
        val['albedo'] = torch.sigmoid(raw['albedo']) # (0, 1), init: 0.5
    if 'reflectivity' in raw:
        val['reflectivity'] = F.softplus(raw['reflectivity'], beta=100) # (0, inf), init: 0.0069
        # val['reflectivity'] = squareplus(raw['reflectivity'], b=1e-4) # (0, inf), init: 0.005
    if 'specular' in raw:
        val['specular'] = torch.sigmoid(raw['specular']) # (0, 1), init: 0.5
    return val


def squareplus(x, b=4.0):
    return 0.5 * (x + (x ** 2 + b).sqrt())


def compute_fg_mask(sdf, dists, dirs, gradients, inv_s, cos_anneal_ratio, n_samples):
    """ Compute the foreground mask for evaluation.
    """
    alpha, c = compute_alpha(sdf, dists, dirs, gradients, inv_s, cos_anneal_ratio)
    alpha = alpha.reshape(-1, n_samples)
    weights_fg = compute_transmittance(alpha)
    mask_fg = weights_fg.sum(dim=-1, keepdim=True)
    return mask_fg


def masked_loss(color, gt, mask):
    mask_sum = mask.sum() + 1e-5
    color_error = (color - gt) * mask
    return F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum


def masked_mean(loss, mask):
    return (mask * loss).sum() / (mask.sum() + 1e-5)


class SymTensor:
    """ Class to wrap tensor with chunk accessor methods.
    Uses global variable has_transform.
    """
    def __init__(self, data, state):
        self.data = data
        self.state = state
        self.num_chunks = 2 if self.has_transform() else 1
        self.iS = 0
        self.iT = 1
    
    def __call__(self):
        return self.data

    def has_transform(self):
        return self.state.has_transform

    def data(self):
        return self.data

    def chunk(self):
        return torch.chunk(self.data, self.num_chunks, dim=0)

    def S(self):
        if self.has_transform():
            return self.chunk()[self.iS]
        else:
            return self.data
    
    def T(self):
        if self.has_transform():
            return self.chunk()[self.iT]
        else:
            return None
    
    def append(self, x):
        self.num_chunks += 1
        self.data = torch.cat((self.data, x), dim=0)
    
    def permute(self, indices):
        if len(indices) == 2:
            self.iS, self.iT = indices
    
    def get(self, spec):
        if spec == "S":
            return self.S()
        elif spec == "T":
            return self.T()
    
    def reshape(self, *args):
        self.data = self.data.reshape(*args)
        return self


def raw_to_material(raw, with_sigmoid=True):
    """ Convert raw material network outputs to scaled values.
    """
    albedo_color = raw[..., :3]
    reflectivity = raw[..., [3]]
    if with_sigmoid:
        albedo_color = torch.sigmoid(albedo_color) # (0, 1), init: 0.5
        reflectivity = torch.sigmoid(reflectivity) # (0, 1), init: 0.5
    # reflectivity = squareplus(raw[..., [3]], b=1e-4) # (0, inf), init: 0.005 # TODO
    return albedo_color, reflectivity


def raw_to_diffuse(raw, with_sigmoid=True):
    """ Convert raw diffuse lighting network output to scaled values.
    """
    res = raw
    if with_sigmoid:
        res = 2.0 * torch.sigmoid(raw) # (0, 2), init: 1
    # return squareplus(raw) # (0, inf), init: 1 # TODO
    return res


def raw_to_specular(raw, with_sigmoid=True):
    """ Convert raw specular lighting network output to scaled values.
    """
    res = raw
    if with_sigmoid:
        res = torch.sigmoid(raw) # (0, 1), init: 0.5
    return res


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

        self.render_types = set(cfg.train.render_types) # SS, TS, ST, TT

        # Set if tensors include symmetry-transformed elements
        self.has_transform = False

    def wrap_sym_tensor(self, x):
        """ Converts a tensor or sequence of tensors into SymTensors
        """
        if isinstance(x, (list, tuple)):
            return [SymTensor(xi, self) for xi in x]
        else:
            return SymTensor(x, self)

    def renderer_config(self):
        return self.cfg.model.renderer

    def create_networks(self):
        cfg = self.cfg

        if cfg.model.renderer.scale_input_coords:
            max_side = torch.max(self.bbox_max)
            scaler = max_side / self.bbox_max
        else:
            scaler = None

        nets = {
            "nerf": NeRF(cfg.model.nerf),
            "sdf_network": SDFNetwork(cfg.model.sdf_network, inputs_scale=scaler),
            "variance_network": SingleVarianceNetwork(cfg.model.variance_network),
            "diffuse_network": RenderingNetwork(cfg.model.diffuse_network),
            "specular_network": RenderingNetwork(cfg.model.specular_network),
            "material_network": RenderingNetwork(cfg.model.material_network),
            "diffuse_sym_network": RenderingNetwork(cfg.model.diffuse_network),
            "specular_sym_network": RenderingNetwork(cfg.model.specular_network),
        }

        self.nerf = nets['nerf']
        self.sdf_network = nets['sdf_network']
        self.deviation_network = nets['variance_network']
        self.material_network = nets['material_network']
        self.diffuse_network = nets['diffuse_network']
        self.specular_network = nets['specular_network']
        self.diffuse_sym_network = nets['diffuse_sym_network']
        self.specular_sym_network = nets['specular_sym_network']

        self.networks = nets

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
            new_sdf, *_ = self.evaluate_sdf(pts.reshape(-1, 3))
            new_sdf = new_sdf.reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size, device=z_vals.device)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def compute_dists(self, z_vals, sample_dist):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        sample_dist = torch.tensor([sample_dist], dtype=torch.float32, device=dists.device)
        dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        return dists, mid_z_vals

    def preprocess_points(self, pts, dirs, dists):
        self.has_transform = not self.inference_mode 

        if self.inference_mode:
            return pts, dirs, dists

        pts_s = self.transform_manager.apply_symmetry_transform(pts)
        dirs_s = self.transform_manager.apply_symmetry_transform(dirs, is_dir=True)

        # [source, symmetric]
        pts = torch.cat([pts, pts_s], dim=0)
        dirs = torch.cat([dirs, dirs_s], dim=0)
        dists = torch.cat([dists]*2, dim=0)

        return pts, dirs, dists

    def evaluate_sdf(self, pts):
        """ Evaluate SDF values at the sampled points.
        If fitting a ground plane, take the SDF value as
        min{object model signed distance, ground model signed distance}
        """
        cfg_p = self.renderer_config()

        sdf_out = self.sdf_network(pts)
        sdfnet_vals = sdf_out["signed_distance"]
        sdfnet_feats = sdf_out["feature"]

        sdf_vals = sdfnet_vals

        if cfg_p.fit_ground_plane:
            sdf_object = sdfnet_vals
            z_transformed = self.transform_manager.apply_transform(pts)[:, [2]] # z coordinate of transformed points
            sdf_ground = z_transformed - self.ground_plane_offset # distance from transformed pt to ground plane; positive above ground
            sdf_all = torch.cat([sdf_ground, sdf_object], axis=1)
            ids = torch.argmin(sdf_all, dim=1, keepdim=True) # ID of ground (0) or object (1), depending on which is most negative 
            all_ground = torch.zeros_like(ids)
            ids = torch.where(sdf_ground < 0, all_ground, ids) # Force ID to be ground if under the ground plane
            sdf_vals = torch.gather(sdf_all, 1, ids) # Choose ground or object model, depending on which is more negative

        return sdf_vals, sdfnet_vals, sdfnet_feats

    def render_color(self, spec, weights, colors, color_bg, color_bg_far,
                     inside_sphere, n_samples, with_bg):
        color = colors[spec].reshape(-1, n_samples, 3)
        if with_bg:
            color = combine_color_with_bg(color, color_bg, inside_sphere.get(spec[0]))
        return compute_integral(color, color_bg_far, weights.get(spec[0]))

    def setup_rendering(self, inputs):
        if "ground_plane_offset" in inputs:
            self.ground_plane_offset = inputs["ground_plane_offset"] # fixed offset from transformed origin
        if "transform_manager" in inputs:
            self.transform_manager = inputs["transform_manager"]

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
                sdf, *_ = self.evaluate_sdf(pts.reshape(-1, 3))
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

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

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
        if 'color' not in results:
            results['color'] = results['color_recon_SS']
        
        return results

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
        cfg_p = self.renderer_config()
        as_sym_tensor = lambda x : self.wrap_sym_tensor(x)

        is_train = not self.inference_mode
        is_test = self.inference_mode

        # Section length
        dists, mid_z_vals = self.compute_dists(z_vals, sample_dist)

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        pts.requires_grad_(True)

        pts, dirs, dists = as_sym_tensor(
            self.preprocess_points(pts, dirs, dists))

        sdf_vals, sdf_fg, sdfnet_feats = as_sym_tensor(
            self.evaluate_sdf(pts())
        )

        sdf_grad_input = sdf_fg() if cfg_p.reuse_sdf_graph else None
        gradients = self.sdf_network.gradient(pts(), sdf_grad_input).squeeze()
        gradients = as_sym_tensor(gradients)
        # ToDo: unscale gradients? https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-penalty

        colors = self.evaluate_color(pts, gradients, dirs, sdfnet_feats)

        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6) # Single parameter
        alpha, c = compute_alpha(sdf_vals(), dists(), dirs(), gradients(), inv_s, cos_anneal_ratio)
        c = as_sym_tensor(c)
        alpha = alpha.reshape(-1, n_samples)

        pts_norm = torch.linalg.norm(pts(), ord=2, dim=-1, keepdim=True).reshape(-1, n_samples)
        inside_sphere = as_sym_tensor((pts_norm < 1.0).float().detach())
        relax_inside_sphere = as_sym_tensor((pts_norm < 1.2).float().detach())

        if cfg.test.nvs_cut_box and is_test:
            inside = self.limit_rendering_volume(pts).reshape(-1, n_samples)
            alpha = alpha * inside
        
        alpha = as_sym_tensor(alpha)

        outputs = dict()

        if is_test and cfg.test.render_foreground_mask:
            mask_fg = compute_fg_mask(sdf_fg(), dists(), dirs(), gradients(), inv_s, cos_anneal_ratio, n_samples)
            outputs.update({
                'mask_fg': mask_fg
            })

        # Render with background
        with_bg = background_alpha is not None and background_rgb is None
        if with_bg:
            background_alpha = background_alpha if is_test else torch.cat([background_alpha]*2)
            background_alpha = as_sym_tensor(background_alpha)
            alpha = as_sym_tensor(self.combine_fg_bg_alpha(
                alpha(), background_alpha(), inside_sphere(), n_samples))

        weights = as_sym_tensor(compute_transmittance(alpha()))

        def render_color_(spec, weights, color_bg):
            return self.render_color(spec, weights, colors, color_bg,
                                     background_rgb, inside_sphere, n_samples, with_bg)
            
        for spec in self.render_types:
            if spec == "SS" or is_train:
                bg = background_sampled_color
                outputs[f"color_recon_{spec}"] = render_color_(spec, weights, bg)
                if cfg.train.color_diffuse_weight > 0.0 and is_train:
                    outputs[f"color_diffuse_{spec}"] = render_color_(spec + 'd', weights, bg)
                if cfg.train.color_symmetric_lighting_weight > 0.0 and is_train and spec[-1] == "T": # ST and TT only
                    outputs[f"color_symmetric_lighting_{spec}"] = render_color_(spec + 's', weights, bg)

        if is_train:
            gradients_b = gradients.reshape(-1, n_samples, 3)
            gradient_error = (torch.linalg.norm(gradients_b(), ord=2, dim=-1) - 1.0) ** 2
            gradient_error = as_sym_tensor(gradient_error)
            outputs.update({
                'gradient_error': masked_mean(gradient_error.S(), relax_inside_sphere.S()),
                'gradient_error_sym': masked_mean(gradient_error.T(), relax_inside_sphere.T())
            })

        if cfg.test.rendering_output != "full" and is_test:
            custom_col = colors["custom_color"].reshape(batch_size, n_samples, 3)
            custom_col = compute_integral(custom_col, background_rgb, weights[:, :n_samples])
            outputs["custom_color"] = custom_col

        outputs.update({
            'sdf': sdf_vals,
            'dists': dists,
            'gradients': gradients.reshape(-1, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights.S(),
            'cdf': c.reshape(-1, n_samples).S(),
            'inside_sphere': inside_sphere
        })

        return outputs

    def evaluate_color(self, pts, normals, dirs, features):
        is_train = not self.inference_mode
        is_test = self.inference_mode
        cfg_r = self.cfg.model.renderer

        as_sym_tensor = lambda x : self.wrap_sym_tensor(x)

        if cfg_r.use_reflected_view_directions:
            refdirs = as_sym_tensor(reflected_rays(dirs, normals))
            refdirs_S, refdirs_T = refdirs.S(), refdirs.T()
        else:
            refdirs_S, refdirs_T = None, None

        def eval_material(**kwargs):
            return raw_to_material(self.material_network(**kwargs), not cfg_r.late_sigmoid)
        def raw2diff(raw):
            return raw_to_diffuse(raw, not cfg_r.late_sigmoid)
        def raw2spec(raw):
            return raw_to_specular(raw, not cfg_r.late_sigmoid)
        def comp_color(*args):
            return compute_color(*args, clamp=not cfg_r.late_sigmoid)
        def comp_diffuse_color(*args):
            return compute_diffuse_color(*args, clamp=not cfg_r.late_sigmoid)

        albedo, reflectivity = as_sym_tensor(eval_material(points=pts(), features=features()))

        diffuse_S = raw2diff(self.diffuse_network(pts.S(), normals.S(), dirs.S(), features.S()))
        specular_S = raw2spec(self.specular_network(pts.S(), normals.S(), dirs.S(), features.S(), refdirs_S))

        output = {
            "SS": comp_color(albedo.S(), reflectivity.S(), diffuse_S, specular_S)
        }

        if is_train:
            diffuse_T = raw2diff(self.diffuse_sym_network(pts.T(), normals.T(), dirs.T(), features.T()))
            specular_T = raw2spec(self.specular_sym_network(pts.T(), normals.T(), dirs.T(), features.T(), refdirs_T))
            diffuse_Ts = raw2diff(self.diffuse_network(pts.T(), normals.T(), dirs.T(), features.T()))
            specular_Ts = raw2spec(self.specular_network(pts.T(), normals.T(), dirs.T(), features.T(), refdirs_T))
            output.update({
                "TS": comp_color(albedo.T(), reflectivity.T(), diffuse_S, specular_S),
                "ST": comp_color(albedo.S(), reflectivity.S(), diffuse_T, specular_T),
                "TT": comp_color(albedo.T(), reflectivity.T(), diffuse_T, specular_T),
                "SSd": comp_diffuse_color(albedo.S(), diffuse_S),
                "TSd": comp_diffuse_color(albedo.T(), diffuse_S),
                "STd": comp_diffuse_color(albedo.S(), diffuse_T),
                "TTd": comp_diffuse_color(albedo.T(), diffuse_T),
                "STs": comp_color(albedo.S(), reflectivity.S(), diffuse_Ts, specular_Ts),
                "TTs": comp_color(albedo.T(), reflectivity.T(), diffuse_Ts, specular_Ts)
            })

        if is_test:
            out_type = self.cfg.test.rendering_output
            if out_type == "d":
                out = albedo.S()
            elif out_type == "d2":
                out = albedo.S() * diffuse_S
            elif out_type == "s":
                out = specular_S
            elif out_type == "s1":
                out = specular_S * reflectivity.S()
            else:
                out = output["SS"]
            output["custom_color"] = out

        return output

    def combine_fg_bg_alpha(self, alpha, alpha_bg, inside_sphere, n_samples):
        outside_sphere = 1.0 - inside_sphere
        if self.inference_mode and self.cfg.test.pre_sphere_transparency:
            # Set all background_alpha before the first nonzero value of inside_sphere to zero, unless entire row is zero (does not intersect sphere)
            # These alphas should become zero during training anyway, if seen from other views
            # May prevent modelling of ground plane immediately outside sphere
            background_mask = ((torch.cumsum(inside_sphere, dim=-1) + (1.0 - inside_sphere.sum(dim=-1, keepdim=True)).clamp(min=0.0)) > 0).float()
            outside_sphere *= background_mask
        alpha = alpha * inside_sphere + alpha_bg[:, :n_samples] * outside_sphere
        alpha = torch.cat([alpha, alpha_bg[:, n_samples:]], dim=-1)
        return alpha

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        if self.cfg.test.mcube_render_road:
            query_func = lambda pts: -self.evaluate_sdf(pts)[0]
        else:
            query_func = lambda pts: -self.sdf_network.sdf(pts)
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=query_func,
                                device=self.device)

    def limit_rendering_volume(self, pts):
        return inside_axis_aligned_box(pts, self.bbox_min, self.bbox_max)

    def evaluate_loss(self, render_out, true_rgb, mask):
        cfg = self.cfg

        color_SS = render_out['color_recon_SS']
        s_val = render_out['s_val']
        cdf_val = render_out['cdf']
        eikonal_loss = render_out['gradient_error']
        weight_max = render_out['weight_max']
        mask_sum = mask.sum() + 1e-5

        psnr = 20.0 * torch.log10(1.0 / (((color_SS - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        log = {
            'Statistics/s_val': s_val.mean(),
            'Statistics/cdf': (cdf_val[:, :1] * mask).sum() / mask_sum,
            'Statistics/weight_max': (weight_max * mask).sum() / mask_sum,
            'Statistics/psnr': psnr,
        }

        loss = 0

        def get_sym_weight(spec):
            return cfg.train.symmetricity if spec[0] == 'T' else 1.0

        # Reconstruction losses
        for spec in self.render_types:
            loss_name = f"color_recon_{spec}"
            loss_component = masked_loss(render_out[loss_name], true_rgb, mask)
            loss_weight = get_sym_weight(spec)
            loss += loss_weight * loss_component
            log.update({f"Loss/{loss_name}": loss_component})

        if cfg.train.color_diffuse_weight > 0.0:
            for spec in self.render_types:
                loss_name = f"color_diffuse_{spec}"
                loss_component = masked_loss(render_out[loss_name], true_rgb, mask)
                loss_weight = cfg.train.color_diffuse_weight * get_sym_weight(spec)
                loss += loss_weight * loss_component
                log.update({f"Loss/{loss_name}": loss_component})

        if cfg.train.color_symmetric_lighting_weight > 0.0:
            for spec in {"ST", "TT"}:
                loss_name = f"color_symmetric_lighting_{spec}"
                loss_component = masked_loss(render_out[loss_name], true_rgb, mask)
                loss_weight = cfg.train.color_symmetric_lighting_weight * get_sym_weight(spec)
                loss += loss_weight * loss_component # Note multiplicative weights
                log.update({f"Loss/{loss_name}": loss_component})

        log['Loss/eikonal'] = eikonal_loss
        loss += eikonal_loss * cfg.train.eikonal_weight

        sym_weight = cfg.train.symmetricity
        if sym_weight > 0.0:
            eikonal_loss_sym = render_out["gradient_error_sym"]
            loss += cfg.train.eikonal_weight * sym_weight
            log.update({"Loss/eikonal_sym": eikonal_loss_sym})
        
        # If ground plane goes AWOL, apply a corrective force up to the initial position:
        ground_retrieval_weight = getattr(cfg.train, "ground_retrieval_weight", 0)
        if ground_retrieval_weight > 0:
            ground_plane = self.transform_manager.t_[0, 2]
            if ground_plane > getattr(cfg.train, "ground_retrieval_turnon_z", 0.1): # if the ground plane has descended by 0.1 below the bbox
                self.ground_retrieval = True # turn on
            elif ground_plane < 0.0: # if the ground plane is back at the bbox
                self.ground_retrieval = False # turn off
            if getattr(self, "ground_retrieval", False): # if ground retrieval triggered, apply loss
                loss += ground_retrieval_weight * ground_plane # apply downward pressure on tz, to raise ground plane
                log.update({f"Loss/ground_retrieval": ground_plane})

        loss *= 0.25

        log.update({
            'Loss/total': loss,
        })

        return loss, log
