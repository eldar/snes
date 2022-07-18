import math

import torch
from torch.nn import functional as F
from typing import Optional, Tuple


def eval_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    get_best_scale: bool = True,
    mask_thr: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the depth error between the prediction `pred` and the ground
    truth `gt`.
    Args:
        pred: A tensor of shape (N, 1, H, W) denoting the predicted depth maps.
        gt: A tensor of shape (N, 1, H, W) denoting the ground truth depth maps.
        mask: A mask denoting the valid regions of the gt depth.
        get_best_scale: If `True`, estimates a scaling factor of the predicted depth
            that yields the best mean squared error between `pred` and `gt`.
            This is typically enabled for cases where predicted reconstructions
            are inherently defined up to an arbitrary scaling factor.
        mask_thr: A constant used to threshold the `mask` to specify the valid
            regions.
    Returns:
        mse_depth: Mean squared error between `pred` and `gt`.
        abs_depth: Mean absolute difference between `pred` and `gt`.
    """

    if mask is not None:
        # mult gt by mask
        gt = gt * (mask > mask_thr).float()

    dmask = (gt > 0.0).float()
    dmask_mass = torch.clamp(dmask.sum(), 1e-4)

    if get_best_scale:
        # mult preds by a scalar "scale_best"
        # 	s.t. we get best possible mse error
        xy = pred * gt * dmask
        xx = pred * pred * dmask
        scale_best = xy.mean() / torch.clamp(xx.mean(), 1e-4)
        pred = pred * scale_best

    df = gt - pred

    mse_depth = (dmask * (df ** 2)).sum() / dmask_mass
    abs_depth = (dmask * df.abs()).sum() / dmask_mass

    return mse_depth, abs_depth, dmask_mass


def calc_psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y, mask=mask)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def calc_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    if mask is None:
        return torch.mean((x - y) ** 2)
    else:
        return (((x - y) ** 2) * mask).sum() / mask.expand_as(x).sum().clamp(1e-5)


def iou(
    predict: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This is a great loss because it emphasizes on the active
    regions of the predict and targets
    """
    dims = tuple(range(predict.dim())[1:])
    if mask is not None:
        predict = predict * mask
        target = target * mask
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-4
    return (intersect / union).sum() / intersect.numel()