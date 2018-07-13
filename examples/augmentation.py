import math

from maskrcnn.utils.affine_transforms import *


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

