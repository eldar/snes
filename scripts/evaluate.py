import startup

import os
from pathlib import Path
import argparse

import numpy as np
import torch
import imageio
import lpips

from util.data import iterate_exp
from util.metric import eval_depth, calc_mse, iou
from models.factory import load_datasets


def get_scale(instance, margin_scale=0.8, dir='/work/eldar/data/datasets/co3d_extra_new_heuristic', category='car'):
    path = os.path.join(dir, category, instance)
    data = np.load(os.path.join(path, "alignment.npy"), allow_pickle=True).item()
    box = data["box_size"]
    max_sz = np.max(box)
    scale = 2.0 / max_sz * margin_scale
    return scale # c2w


def mse_to_psnr(mse):
    psnr = -10.0 * torch.log10(torch.tensor(mse))
    return psnr.item()


def img_to_pytorch(im):
    im = im.cuda().permute(2, 0, 1).unsqueeze(0)
    return 2.0 * im.cuda().clamp(0.0, 1.0) - 1.0


def mask_to_box(mask):
    # columns = torch.split(mask, 1, dim=1)
    mask = mask.squeeze()
    mask = mask > 0.5
    
    column = torch.max(mask, dim=1)[0]
    ys = torch.nonzero(column)
    min_y = torch.min(ys).item()
    max_y = torch.max(ys).item()

    row = torch.max(mask, dim=0)[0]
    xs = torch.nonzero(row)
    min_x = torch.min(xs).item()
    max_x = torch.max(xs).item()

    return min_y, max_y, min_x, max_x


def main(instance_list, args):
    delimiter = args.delimiter

    lpips_model = lpips.LPIPS(net="vgg")
    lpips_model = lpips_model.cuda()

    exp, per_frame = args.exp, args.per_frame
    with_iou = not args.without_iou
    cfg_file = args.config
    init_cfg = {
        "dataset": {
            "trainval_split": True
        }
    }

    iterator = iterate_exp(exp,
                           instance_list,
                           init_cfg,
                           cfg_file=cfg_file,
                           with_runner=False)
    scenes = []
    # num_instances = 11
    # for i, cfg in zip(range(num_instances), iterator):
    for i, cfg in enumerate(iterator):
        print("scene", i)
        
        if args.scale_type == 'snes':
            scale_c2w = get_scale(cfg.dataset.instance, dir=cfg.dataset.data_extra_dir, category=cfg.dataset.category)
            print('Our scaling factor', scale_c2w)
            get_best_scale = False
        else:
            get_best_scale = True

        _, dataset = load_datasets(cfg)

        num_frames = len(dataset.filenames)

        sub_dir = Path(args.result_dir)
        depth_dir = sub_dir.joinpath("eval_depths")
        render_dir = sub_dir.joinpath("eval_renders")
        mask_dir = sub_dir.joinpath("eval_masks")
        frames = []
        lpips_frames = []
        
        for k in range(num_frames):
            filename = dataset.filenames[k]
            basename = os.path.splitext(filename)[0]
            out = dict()
            # depth
            depth_pred_file = depth_dir.joinpath(f"{basename}.npy")
            if not depth_pred_file.exists():
                print("Missing prediction:", cfg.dataset.instance, filename)
                return

            depth_pred = torch.from_numpy(np.load(depth_pred_file)).reshape(-1)
            
            if args.scale_type == 'snes':
                depth_pred = depth_pred / scale_c2w # Rescale depth predictions

            depth_gt = dataset.co3d_depth_maps[k].reshape(-1)
            depth_mask = dataset.co3d_depth_masks[k].reshape(-1)
            out.update({
                "depth_pred": depth_pred,
                "depth_gt": depth_gt,
                "depth_mask": depth_mask
            })

            # iou
            if with_iou:
                mask_pred_file = mask_dir.joinpath(f"{basename}.png")
                mask_pred = torch.from_numpy(imageio.imread(mask_pred_file)) / 255
                mask_gt = dataset.co3d_masks[k].squeeze()
                threshold = 0.5
                mask_pred = (mask_pred > threshold).type(torch.float32).reshape(-1, 1)
                mask_gt = (mask_gt > threshold).type(torch.float32).reshape(-1, 1)
                iou_mask = iou(
                    mask_pred.reshape(1, -1),
                    mask_gt.reshape(1, -1)
                )
                out.update({
                    "iou_mask": iou_mask.item(),
                })

            # rgb
            rgb_pred_file = render_dir.joinpath(f"{basename}.png")
            rgb_pred = torch.from_numpy(imageio.imread(rgb_pred_file)) / 255
            rgb_gt = dataset.images[k]
            out.update({
                "rgb_pred": rgb_pred.reshape(-1, 3),
                "rgb_gt": rgb_gt.reshape(-1, 3),
                "rgb_mask": dataset.co3d_masks[k].reshape(-1, 1)
            })
            frames += [out]

            min_y, max_y, min_x, max_x = mask_to_box(dataset.co3d_masks[k])
            rgb_gt_pyt = img_to_pytorch(rgb_gt)[:, :, min_y:max_y, min_x:max_x]
            rgb_pred_pyt = img_to_pytorch(rgb_pred)[:, :, min_y:max_y, min_x:max_x]
            lpips_frames += [lpips_model.forward(rgb_gt_pyt, rgb_pred_pyt).item()]

            if per_frame:
                mse_depth, mae_depth, _ = eval_depth(depth_pred, depth_gt, depth_mask, get_best_scale=get_best_scale)
                mse_depth = mse_depth.item()
                mae_depth = mae_depth.item()
                print(f"{filename}{delimiter}{mse_depth} [mse]")
                print(f"{filename}{delimiter}{mae_depth} [mae]")

                rgb_mask = out["rgb_mask"] # (2248000, 1)
                rgb_pred = out["rgb_pred"] # (2248000, 1)
                rgb_gt = out["rgb_gt"] # (2248000, 1)
                mse_rgb = calc_mse(rgb_pred, rgb_gt, rgb_mask)
                mse_rgb_wo_mask = calc_mse(rgb_pred, rgb_gt, mask=None)
                psnr = mse_to_psnr(mse_rgb)
                psnr_wo_mask = mse_to_psnr(mse_rgb_wo_mask)
                print(f"{filename}{delimiter}{mse_rgb.item()} [mse rgb]")
                print(f"{filename}{delimiter}{mse_rgb_wo_mask.item()} [mse w/o mask rgb]")
                print(f"{filename}{delimiter}{psnr} [psnr]")
                print(f"{filename}{delimiter}{psnr_wo_mask} [psnr w/o mask]")
                if with_iou:
                    print(f"{filename}{delimiter}{iou_mask.item()} [IoU]")

        def cat_frames(arr, key):
            return torch.cat([x[key] for x in arr]).cuda()

        rgb_mask = cat_frames(frames, "rgb_mask")
        rgb_mask_sum = rgb_mask.sum().clamp(1e-5)
        mse_rgb = calc_mse(
            cat_frames(frames, "rgb_pred"), cat_frames(frames, "rgb_gt"), rgb_mask
        )
        mse_depth, mae_depth, depth_mask_sum = eval_depth(
            cat_frames(frames, "depth_pred"),
            cat_frames(frames, "depth_gt"),
            cat_frames(frames, "depth_mask"),
            get_best_scale=get_best_scale
        )
        scene = {
            "instance": cfg.dataset.instance,
            "mse_depth": mse_depth.item(),
            "mae_depth": mae_depth.item(),
            "depth_mask_sum": depth_mask_sum.item(),
            "rgb_mask_sum": rgb_mask_sum.item(),
            "mse_rgb": mse_rgb.item(),
            "lpips": np.array(lpips_frames)
        }
        if with_iou:
            scene["iou_mask"] = [ou["iou_mask"] for ou in frames]

        scenes += [scene]
        
    print(exp)

    header = ["Scene", "PSNR", "MSE", "LPIPS", "MSE_Depth", "MAE_Depth"]
    if with_iou:
        header += ["IoU"]
    header = delimiter.join(header)
    print(header)
    
    for i, sc in enumerate(scenes):
        id = sc["instance"]
        mse_depth = sc["mse_depth"]
        mae_depth = sc["mae_depth"]
        mse_rgb = sc["mse_rgb"]
        psnr = mse_to_psnr(mse_rgb)
        lpips_metric = sc["lpips"].mean().item()
        metrics = [psnr, mse_rgb, lpips_metric, mse_depth, mae_depth]
        if with_iou:
            iou_mask = np.array(sc["iou_mask"]).mean()
            metrics += [iou_mask]
        instance_res = delimiter.join([id] + [str(m) for m in metrics])
        print(instance_res)
    
    mse_depth_sum = np.array([s["mse_depth"] for s in scenes])
    depth_mask_sum = np.array([s["depth_mask_sum"] for s in scenes])
    mse_depth_mean = (mse_depth_sum * depth_mask_sum).sum() / depth_mask_sum.sum()

    mae_depth_sum = np.array([s["mae_depth"] for s in scenes])
    mae_depth_mean = (mae_depth_sum * depth_mask_sum).sum() / depth_mask_sum.sum()

    mse_rgb_sum = np.array([s["mse_rgb"] for s in scenes])
    rgb_mask_sum = np.array([s["rgb_mask_sum"] for s in scenes])
    mse_rgb_mean = (mse_rgb_sum * rgb_mask_sum).sum() / rgb_mask_sum.sum()

    psnr_mean = mse_to_psnr(mse_rgb_mean)

    lpips_mean = np.concatenate([sc["lpips"] for sc in scenes]).mean()

    metrics = [psnr_mean, mse_rgb_mean, lpips_mean, mse_depth_mean, mae_depth_mean]

    if with_iou:
        iou_mask_mean = np.concatenate([sc["iou_mask"] for sc in scenes]).mean()
        metrics += [iou_mask_mean]

    mean_res = delimiter.join(["Mean"] + [str(m) for m in metrics])

    # mean_res = f"Mean{delimiter}{psnr_mean}{delimiter}{mse_rgb_mean}{delimiter}{lpips_mean}{delimiter}{mse_depth_mean}{delimiter}{mae_depth_mean}{delimiter}{iou_mask_mean}"
    print(mean_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp', type=str, default='co3d/eccv/neus_co3d_split')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--instance-list', type=str, default='lists/co3d_testset.txt')
    parser.add_argument('--instance', type=str, default='')
    parser.add_argument('--per-frame', action='store_true')
    parser.add_argument('--scale-type', type=str, default='snes')
    parser.add_argument('--delimiter', type=str, default=', ')
    parser.add_argument('--without-iou', action='store_true')
    parser.add_argument('--result-dir', type=str, default='.')
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(args.gpu)
    instance_list = args.instance_list
    if args.instance:
        instance_list = [args.instance]
    main(instance_list, args)
