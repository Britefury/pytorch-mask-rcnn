import numpy as np
import torch
import torch.nn.functional as F

from maskrcnn.utils import affine_transforms


def t_identity_xf(N, torch_device):
    """
    Construct N identity 2x3 transformation matrices
    :return: Torch tensor of shape (N, 2, 3)
    """
    xf = torch.zeros([N, 2, 3], dtype=torch.float, device=torch_device)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    return xf

def t_inv_nx2x2(X):
    """
    Invert the N 2x2 transformation matrices stored in X; a (N,2,2) torch tensor
    :param X: transformation matrices to invert, (N,2,2) array
    :return: inverse of X
    """
    rdet = 1.0 / (X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1])
    y = torch.zeros_like(X)
    y[:, 0, 0] = X[:, 1, 1] * rdet
    y[:, 1, 1] = X[:, 0, 0] * rdet
    y[:, 0, 1] = -X[:, 0, 1] * rdet
    y[:, 1, 0] = -X[:, 1, 0] * rdet
    return y

def t_inv_nx2x3(m):
    """
    Invert the N 2x3 transformation matrices stored in X; a (N,2,3) array
    :param X: transformation matrices to invert, (N,2,3) array
    :return: inverse of X
    """
    m2 = m[:, :, :2]
    mx = m[:, :, 2:3]
    m2inv = t_inv_nx2x2(m2)
    mxinv = torch.matmul(m2inv, -mx)
    return torch.cat([m2inv, mxinv], dim=2)

def t_cat_nx2x3_2(a, b):
    """
    Multiply the N 2x3 transformations stored in `a` with those in `b`
    :param a: transformation matrices, (N,2,3) array
    :param b: transformation matrices, (N,2,3) array
    :return: `a . b`
    """
    a2 = a[:, :, :2]
    b2 = b[:, :, :2]

    ax = a[:, :, 2:3]
    bx = b[:, :, 2:3]

    ab2 = torch.matmul(a2, b2)
    abx = ax + torch.matmul(a2, bx)
    return torch.cat([ab2, abx], dim=2)

def t_cat_nx2x3(*x):
    """
    Multiply the N 2x3 transformations stored in the arrays in `x`
    :param x: transformation matrices, tuple of (N,2,3) arrays
    :return: `x[0] . x[1] . ... . x[N-1]`
    """
    y = x[0]
    for i in range(1, len(x)):
        y = t_cat_nx2x3_2(y, x[i])
    return y

def t_transform_points(xf, points_xy):
    """
    Apply the transformation matrix `xf` to the points in `points_xy`.

    :param xf: transformation as a (2, 3) Torch tensor
    :param points_xy: points as a (N, 2) Torch tensor where each point is of the form (x, y), not (y, x)
    :return: (N, 2) Torch Tensor
    """
    return torch.matmul(xf[:2, :2], points_xy.permute(1, 0)).permute(1, 0) + xf[:, 2][None, :]

def t_transform_vectors(xf, vectors_xy):
    """
    Apply the transformation matrix `xf` to the vectors in `vectors_xy`.
    Like `transform_points` except that the translation component of `xf` is not applied.

    :param xf: transformation as a (2, 3) Torch tensor
    :param vectors_xy: vectors as a (N, 2) Torch tensor where each vector is of the form (x, y), not (y, x)
    :return: (N, 2) Torch tensor
    """
    return torch.matmul(xf[:2, :2], vectors_xy.permute(1, 0)).permute(1, 0)


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

