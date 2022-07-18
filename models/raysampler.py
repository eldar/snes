import torch

from models.camera import RayBundle


def _make_bundle(pixels_x, pixels_y):
    xy = torch.stack([pixels_x, pixels_y], dim=-1)
    return RayBundle(xys=xy)


def sample_random_rays(camera, batch_size, device=None):
    im_size = camera.get_image_size()[0]
    H = im_size[0].item()
    W = im_size[1].item()
    pixels_x = torch.randint(low=0, high=W, size=[batch_size], device=device)
    pixels_y = torch.randint(low=0, high=H, size=[batch_size], device=device)
    return _make_bundle(pixels_x, pixels_y)


def sample_rays_on_grid(camera, resolution_level=1, device=None):
    # This is actually sampling pixel indices, not coordinates (offset by 0.5)
    im_size = camera.get_image_size()[0]
    H = im_size[0].item()
    W = im_size[1].item()
    l = resolution_level
    H_s = H // l
    W_s = W // l
    tx = torch.linspace(0, W - 1, W_s, device=device)
    ty = torch.linspace(0, H - 1, H_s, device=device)
    pixels_y, pixels_x = torch.meshgrid(ty, tx)
    return _make_bundle(pixels_x.reshape((-1)), pixels_y.reshape((-1))), H_s, W_s


def rays_to_world(camera, rays):
    # TODO: Offset pixels_x and pixels_y by +0.5 to move to centre of the pixel
    xys = rays.xys
    pixels_x = xys[:, 0]
    pixels_y = xys[:, 1]
    intrinsics_inv = camera.get_inverse_intrinsics()
    pose = camera.get_pose_matrix()
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
    p = torch.matmul(intrinsics_inv[:, :3, :3], p[:, :, None]).squeeze(2) # batch_size, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
    rays_v = torch.matmul(pose[:, :3, :3], rays_v[:, :, None]).squeeze(2)  # batch_size, 3
    rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)    # batch_size, 3
    rays_o = pose[:, :3, 3].expand(rays_v.shape) # batch_size, 3
    return RayBundle(origins=rays_o, directions=rays_v, xys=xys)


def rays_to_cam(camera, rays):
    # Convert "rays" (pixel indices) into rays, in the camera coordinates
    # TODO: Offset pixels_x and pixels_y by +0.5 to move to centre of the pixel
    xys = rays.xys
    pixels_x = xys[:, 0]
    pixels_y = xys[:, 1]
    intrinsics_inv = camera.get_inverse_intrinsics()
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
    p = torch.matmul(intrinsics_inv[:, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
    rays_o = torch.zeros_like(rays_v)
    return RayBundle(origins=rays_o, directions=rays_v, xys=xys)


def pixels_to_cam(camera, pixel_indices):
    # TODO: Offset pixels_x and pixels_y by +0.5 to move to centre of the pixel
    pixels_x = pixel_indices.xys[:, 0]
    pixels_y = pixel_indices.xys[:, 1]
    intrinsics_inv = camera.get_inverse_intrinsics()
    x_cam = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
    x_cam = torch.matmul(intrinsics_inv[:, :3, :3], x_cam[:, :, None]).squeeze() # batch_size, 3
    return x_cam


def sample_image_pixels(image, rays):
    xys = rays.xys
    xs = xys[:, 0]
    ys = xys[:, 1]
    return image[(ys, xs)].cuda()