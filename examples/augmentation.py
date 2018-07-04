import math
import numpy as np
from .affine_transforms import *


def round_up_shape(shape, tile_shape):
    if len(shape) != len(tile_shape):
        raise ValueError('shape ({}) and tile_shape ({}) should have the same number of dimensions'.format(
            len(shape), len(tile_shape)
        ))
    return tuple([int(math.ceil(float(s) / t) * t) for s, t in zip(shape, tile_shape)])

def _dim_pad_crop(source_size, dest_size):
    diff = dest_size - source_size
    p0 = diff // 2
    return p0, diff - p0

def compute_pad_or_crop(source_shape, dest_shape):
    """
    Compute the padding or cropping required to transform a block of shape `source_shape` into
    a block of shape `dest_shape`. Note that `source_shape` and `dest_shape` must have they same
    number of dimensions; e.g. `len(source_shape) == len(dest_shape)`.

    :param source_shape: the shape of the source block as a sequence of integers
    :param dest_shape: the shape of the destination block as a sequence of integers
    :return: per dimension padding/cropping, where each entry is a 2-tuple that gives the padding/cropping
        before/left/above and after/right/below. The values are positive for padding and negative for cropping.
        e.g. `[(pad_crop_before_0, pad_crop_after_0), (pad_crop_before_1, pad_crop_after_1), ...]`
    """
    if len(source_shape) != len(dest_shape):
        raise ValueError('source_shape has {} dimensions, dest_shape has {}; should be the same'.format(len(source_shape), len(dest_shape)))
    return [_dim_pad_crop(s, d) for s, d in zip(source_shape, dest_shape)]

def compute_padding(shape, padded_shape):
    return [(0, p-s) for p, s in zip(padded_shape, shape)]

def padding_to_slice(p):
    start = p[0] if p[0] > 0 else None
    stop = -p[1] if p[1] > 0 else None
    return slice(start, stop)

def crop_to_slice(c):
    start = -c[0] if c[0] > 0 else None
    stop = c[1] if c[1] > 0 else None
    return slice(start, stop)


def random_shift(img_size, min_shift, out_size, rng=None):
    """
    Compute the padding and cropping required to select a `out_size` sized portion
    of an image whose original size is `img_size` and should randomly shift by
    at least `min_shift` pixels.

    Example:
    Take an image as an array and compute the padding+cropping needed to give
    a shift of at least 16 pixels resulting in a 256x256 image:
    >>> pad0, crop0 = random_shift(x.shape[0], 16, 256)
    >>> pad1, crop1 = random_shift(x.shape[1], 16, 256)
    >>> x = np.pad(x, [pad0, pad1], mode='constant')
    >>> x = x[crop0, crop1]

    :param img_size: the size of the original image
    :param min_shift: the amount of variation desired
    :param out_size: the fixed output size
    :return: tuple `(pad, crop)`, where
        pad is `(pad_start, pad_end)` and crop is a python slice
    """
    if rng is None:
        rng = np.random

    if (img_size + min_shift) <= out_size:
        # Size is small enough that random padding will suffice
        space = out_size - img_size
        pad_start = int(rng.randint(0, space))
        pad = pad_start, space - pad_start
        crop = slice(None)
    elif img_size >= (out_size + min_shift):
        # Size is big enough that random cropping will suffice
        extra = img_size - out_size
        pad = 0, 0
        crop_start = int(rng.randint(0, extra))
        crop = slice(crop_start, crop_start + out_size)
    else:
        # Add random padding to one side or the other
        pad_x = int(rng.randint(-min_shift, min_shift))
        if pad_x >= 0:
            # Pad start, crop from 0 to fixed_size
            pad_start = pad_x
            pad_end = max(out_size - (img_size + pad_start), 0)
            crop = slice(0, out_size)
        else:
            # Pad end, crop from `padded_size - fixed_size` to end
            pad_end = -pad_x
            pad_start = max(out_size - (img_size + pad_end), 0)
            padded_size = img_size + pad_start + pad_end
            crop = slice(padded_size - out_size, None)
        pad = pad_start, pad_end
    return pad, crop


class ImageAugmentation (object):
    def __init__(self, hflip, vflip, hvflip, xlat_range, affine_std=0.0, rot_range_mag=0.0,
                 light_scl_std=0.0, light_off_std=0.0,
                 scale_u_range=None, scale_x_range=None, scale_y_range=None,
                 scaled_rotation=False):
        self.hflip = hflip
        self.vflip = vflip
        self.hvflip = hvflip
        self.xlat_range = xlat_range
        self.affine_std = affine_std
        self.rot_range_mag = rot_range_mag
        self.light_scl_std = light_scl_std
        self.log_light_scl_std = math.log(light_scl_std + 1.0)
        self.light_off_std = light_off_std
        self.scale_u_range = scale_u_range
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.scaled_rotation = scaled_rotation


    def aug_xforms(self, N, image_size):
        xf = identity_xf(N)

        if self.hflip:
            x_hflip = np.random.binomial(1, 0.5, size=(N,)) * 2 - 1
            xf[:, 0, 0] = x_hflip.astype(np.float32)

        if self.vflip:
            x_vflip = np.random.binomial(1, 0.5, size=(N,)) * 2 - 1
            xf[:, 1, 1] = x_vflip.astype(np.float32)

        if self.hvflip:
            x_hvflip = np.random.binomial(1, 0.5, size=(N,)) != 0
            xf[x_hvflip, :, :] = xf[x_hvflip, ::-1, :]

        if self.scale_u_range is not None and self.scale_u_range[0] is not None:
            scl = np.random.uniform(low=self.scale_u_range[0], high=self.scale_u_range[1], size=(N,))
            xf[:, 0, 0] *= scl
            xf[:, 1, 1] *= scl
        if self.scale_x_range is not None and self.scale_x_range[0] is not None:
            xf[:, 0, 0] *= np.random.uniform(low=self.scale_x_range[0], high=self.scale_x_range[1], size=(N,))
        if self.scale_y_range is not None and self.scale_y_range[0] is not None:
            xf[:, 1, 1] *= np.random.uniform(low=self.scale_y_range[0], high=self.scale_y_range[1], size=(N,))

        if self.affine_std > 0.0:
            xf[:, :, :2] += np.random.normal(scale=self.affine_std, size=(N, 2, 2))

        xlat_y_bounds = self.xlat_range * 2.0 / float(image_size[0])
        xlat_x_bounds = self.xlat_range * 2.0 / float(image_size[1])
        xf[:, 0, 2] += np.random.uniform(low=-xlat_x_bounds, high=xlat_x_bounds, size=(N,))
        xf[:, 1, 2] += np.random.uniform(low=-xlat_y_bounds, high=xlat_y_bounds, size=(N,))

        if self.rot_range_mag > 0.0:
            thetas = np.random.uniform(low=-self.rot_range_mag, high=self.rot_range_mag, size=(N,))
            rot_xf = rotation_matrices(thetas)

            if self.scaled_rotation:
                # Original
                xf = cat_nx2x3(xf, rot_xf)
            else:
                # Maybe more correct
                xf = cat_nx2x3(rot_xf, xf)

        return xf.astype(np.float32)

    def aug_colour_xforms(self, N):
        light_offset = np.zeros((N,))
        if self.light_off_std > 0.0:
            light_offset = np.random.normal(scale=self.light_off_std, size=(N,))

        light_scale = np.ones((N,))
        if self.light_scl_std > 0.0:
            light_scale = np.exp(np.random.normal(loc=0.0, scale=self.log_light_scl_std, size=(N,)))

        return light_scale.astype(np.float32), light_offset.astype(np.float32)

