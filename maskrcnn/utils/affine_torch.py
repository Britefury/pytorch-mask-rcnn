import numpy as np
import torch
import torch.nn.functional as F

from maskrcnn.utils import affine_transforms


def torch_reflect_grid(x):
    """
    Apply reflection padding to co-ordinates that are to be used by `torch.nn.functional.grid_sample`
    :param x: co-ordinates as a Torch tensor
    :return: tensor
    """
    return torch.abs(torch.fmod(torch.abs(x-1), 4.0)-2)-1

def torch_replicate_grid(x):
    """
    Apply replication padding to co-ordinates that are to be used by `torch.nn.functional.grid_sample`
    :param x: co-ordinates as a Torch tensor
    :return: tensor
    """
    return torch.clamp(x, -1.0, 1.0)

def torch_pad_grid(x, pad_mode):
    """
    Apply padding to co-ordinates that are to be used by `torch.nn.functional.grid_sample`
    :param x: co-ordinates as a Torch tensor
    :param pad_mode: 'reflect' for reflection padding, 'replicate' for replication padding or 'constant' for constant
    (zero) padding
    :return: tensor
    """
    if pad_mode == 'reflect':
        return torch_reflect_grid(x)
    elif pad_mode == 'replicate':
        return torch_replicate_grid(x)
    elif pad_mode == 'constant':
        return x
    else:
        raise ValueError('Unknown pad_mode \'{}\'; should be reflect, replicate or constant'.format(pad_mode))


def torch_grid_sample_nearest(input, grid):
    """
    Helper function that performs nearest neighbour sampling using `torch.nn.functional.grid_sample`
    by clamping grid co-ordinates in order to emulate nearest sampling using the bi-linear sampling
    that is provided.

    Use it like `torch.nn.functional.grid_sample`.

    :param input: Input image tensor to sample
    :param grid: Sample co-ordinates as a Torch tensor
    :return: Sampled image as a Torch tensor
    """
    scale_factor = (np.array([input.size(3), input.size(2)])-1)
    sf_var = torch.tensor(scale_factor[None, None, None, :], dtype=torch.float, device=input.device)
    grid_n = (((grid + 1) * 0.5 * sf_var).round()) / sf_var * 2.0 - 1.0
    return F.grid_sample(input, grid_n, 'nearest')


def torch_scaled_grid(xform, size, scale_factor, torch_device):
    grid_scale = np.array([float(size[2]-scale_factor) / float(size[2]-1),
                           float(size[3]-scale_factor) / float(size[3]-1)])
    xf_scale = np.zeros_like(xform)
    xf_scale[:, 0, 0] = grid_scale[0]
    xf_scale[:, 1, 1] = grid_scale[1]
    xf = affine_transforms.cat_nx2x3(xform, xf_scale)
    xf_var = torch.tensor(xf, dtype=torch.float, device=torch_device)
    grid = F.affine_grid(xf_var, torch.Size((size[0], size[1], size[2] // scale_factor, size[3] // scale_factor)))
    return grid


def torch_matmul_imagetensor_colour_matrix(X, colour_matrix):
    X_out = []
    for i in range(X.size()[0]):
        img = X[i, :, :, :].permute(1, 2, 0)
        img_m = torch.matmul(img, colour_matrix[i, :, :]).permute(2, 0, 1)
        X_out.append(img_m[None, ...])

    X_out = torch.cat(X_out)

    return X_out

