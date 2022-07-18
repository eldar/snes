import matplotlib.pyplot as plt
import numpy as np


CMAP_DEFAULT = 'plasma'


def gray2rgb(im, cmap=CMAP_DEFAULT):
    cmap = plt.get_cmap(cmap)
    result_img = cmap(im.astype(np.float32))
    if result_img.shape[2] > 3:
        result_img = np.delete(result_img, 3, 2)
    return result_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None,
                                cmap=CMAP_DEFAULT):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.

    depth = np.squeeze(depth)

    depth_f = depth.flatten()
    depth_f = depth_f[depth_f != 0]
    disp_f = 1.0 / (depth_f + 1e-6)
    percentile = np.percentile(disp_f, pc)

    disp = 1.0 / (depth + 1e-6)
    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (percentile + 1e-6)
    disp = np.clip(disp, 0, 1)
    disp = gray2rgb(disp, cmap=cmap)
    keep_h = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_h]
    return disp