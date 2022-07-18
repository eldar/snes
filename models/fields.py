import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


# Implementation from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self, cfg, inputs_scale=None):
        super(SDFNetwork, self).__init__()

        multires = cfg.multires
        dims = [cfg.d_in] + [cfg.d_hidden for _ in range(cfg.n_layers)] + [cfg.d_out]

        if inputs_scale is None:
            self.scale = 1.0
            base_freq = 1.0
        else:
            self.scale = inputs_scale
            base_freq = 1.0 / inputs_scale.flatten()

        self.embed_fn = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=cfg.d_in, base_freq=base_freq)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = cfg.skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if cfg.geometric_init:
                if l == self.num_layers - 2:  # final hidden layer
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001) # sqrt(pi/out_dim) + noise
                    torch.nn.init.constant_(lin.bias, -cfg.bias) # -radius
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0) # zero out last (input_dim - 3) elements of skip connection, but not (x, y, z) skip inputs; this deviates from the initialisation proposed in SAL
                else:
                    torch.nn.init.constant_(lin.bias, 0.0) # b_i = 0
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)) # N(0, sqrt(2) / sqrt(out_dim))

            if cfg.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
    
    def forward(self, inputs):
        inputs = inputs * self.scale # scale SDF inputs
        if self.embed_fn is not None:
            inputs = self.embed_fn(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        
        return {"signed_distance": x[:, :1], "feature": x[:, 1:]}

    def sdf(self, x):
        return self.forward(x)["signed_distance"]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x, y=None):
        if y is None:
            x.requires_grad_(True)
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.mode = cfg.mode
        self.mode = self.mode.split("_")
        dims = [cfg.d_in + cfg.d_feature] + \
            [cfg.d_hidden for _ in range(cfg.n_layers)] + [cfg.d_out]
        self.out_scale = cfg.out_scale if "out_scale" in cfg else 1.0
        self.out_bias = cfg.out_bias if "out_bias" in cfg else 0.0

        self.embedview_fn = None
        if cfg.multires_view > 0:
            embedview_fn, input_ch = get_embedder(cfg.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        
        self.embedrefview_fn = None
        if 'refviewdirs' in cfg.mode and cfg.multires_refview > 0:
            embedrefview_fn, input_ch = get_embedder(cfg.multires_refview)
            self.embedrefview_fn = embedrefview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim) # uniform intialisation U(-sqrt(1/dims[l]), sqrt(1/dims[l]))

            if cfg.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points=None, normals=None, view_dirs=None, features=None, refview_dirs=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        if self.embedrefview_fn is not None:
            refview_dirs = self.embedrefview_fn(refview_dirs)

        inputs = {"points": points, "viewdirs": view_dirs, "normals": normals,
                  "refviewdirs": refview_dirs, "feats": features}
        rendering_input = torch.cat([inputs[inp] for inp in self.mode], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self, cfg):
        super(NeRF, self).__init__()
        D = self.D = cfg.D
        W = self.W = cfg.W
        self.d_in = cfg.d_in
        self.d_in_view = cfg.d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if cfg.multires > 0:
            embed_fn, input_ch = get_embedder(cfg.multires, input_dims=cfg.d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if cfg.multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(cfg.multires_view, input_dims=cfg.d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = cfg.skips
        self.use_viewdirs = cfg.use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1) # init U(-sqrt(1/W), -sqrt(1/W)); alpha goes through softplus
        if cfg.use_viewdirs:
            ### Implementation according to the official code release
            ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

            ### Implementation according to the paper
            # self.views_linears = nn.ModuleList(
            #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.rgb_linear = nn.Linear(W, 3)
        
        if cfg.rgb_bias != 0.0:
            torch.nn.init.constant_(self.rgb_linear.bias.data, cfg.rgb_bias) # alter background color at initialization
        if cfg.alpha_bias != 0.0:
            torch.nn.init.constant_(self.alpha_linear.bias.data, cfg.alpha_bias) # alter background color at initialization

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            rgb = self.rgb_linear(feature)
            return alpha, rgb


# Implementation from NeuS: https://github.com/Totoro97/NeuS
class SingleVarianceNetwork(nn.Module):
    def __init__(self, cfg):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(cfg.init_val)))

    def forward(self, x):
        device = self.variance.device
        return torch.ones([len(x), 1], device=device) * torch.exp(self.variance * 10.0)
