import os
import sys

import numpy as np
import torch


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset
       
        cameras = dataset.get_cameras()

        self.num_rays = []
        for cam in cameras:
            img_w, img_h = cam.get_image_size()[0].tolist()
            self.num_rays += [img_w*img_h]
        self.total_num_rays = np.sum(np.array(self.num_rays))

    def __len__(self):
        return self.total_num_rays

    def ray_idx_to_img_ray(self, idx):
        for i, num in enumerate(self.num_rays):
            if idx < num:
                return i, idx
            idx -= num
        assert False

    def __getitem__(self, idx):
        img_idx, ray_idx = self.ray_idx_to_img_ray(idx)
        cam = self.dataset.get_cameras()[img_idx]
        img = self.dataset.images[img_idx]
        img_h, img_w = cam.get_image_size()[0].tolist()
        pix_y = ray_idx // img_w
        pix_x = ray_idx % img_w
        return torch.cat([torch.tensor([img_idx, pix_x, pix_y]),
                          img[pix_y, pix_x, :]])
