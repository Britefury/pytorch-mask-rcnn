import os
import pickle
import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from skimage.util import img_as_float

from maskrcnn.utils.affine_transforms import *
from maskrcnn.utils.ground_truths import label_image_to_boundaries
from examples import settings


def _get_ellipses_root_dir(exists=True):
    return settings.get_config_dir('ellipses_root', exists=exists)


def make_circle(n_verts):
    return np.stack([np.cos(np.linspace(0.0, 2.0 * np.pi, n_verts)),
                     np.sin(np.linspace(0.0, 2.0 * np.pi, n_verts))], axis=1)[:-1] * 0.5


def _get_rng(rng):
    if rng is None:
        return np.random
    elif isinstance(rng, int):
        return np.random.RandomState(rng)
    else:
        return rng


def random_object_matrices(n, image_size, rot_range, size_range, aspect_bound, rng=None):
    rng = _get_rng(rng)
    # Rotations
    rotations = rng.uniform(rot_range[0], rot_range[1], size=(n,))

    # Draw random aspect ratios, keeping area the same
    log_aspect_bound = np.log(aspect_bound) / 2
    ratios = np.exp(rng.uniform(-log_aspect_bound, log_aspect_bound, size=(n,)))
    # Draw random sizes
    sizes = np.exp(rng.uniform(np.log(size_range[0]), np.log(size_range[1]), size=(n,)))
    scales = np.stack([sizes * ratios, sizes / ratios], axis=1)

    # Limit position to lie inside the image with a border whose size is the average object size
    pos_border = (size_range[0] + size_range[1]) * 0.5
    pos_x_range = pos_border, image_size[1] - pos_border
    pos_y_range = pos_border, image_size[0] - pos_border
    pos = np.stack([rng.uniform(pos_x_range[0], pos_x_range[1], size=(n,)),
                    rng.uniform(pos_y_range[0], pos_y_range[1], size=(n,))], axis=1)
    return cat_nx2x3(translation_matrices(pos), rotation_matrices(rotations), scale_matrices(scales), )

def group_object_matrices(n, rot_range, base_scale, aspect_bound, pos_offset_range, pos_step, rng=None):
    rng = _get_rng(rng)
    # Rotations as before, except that the bounds passed will be tighter
    rotations = rng.uniform(rot_range[0], rot_range[1], size=(n,))

    # Draw random aspect rations (more constrained)
    log_aspect_range = np.log(aspect_bound) / 2
    ratios = np.exp(rng.uniform(-log_aspect_range, log_aspect_range, size=(n,)))
    # Multiply by a base scale factor that provides the main contribution of aspect ratio
    scales = np.stack([base_scale[0] * ratios, base_scale[1] / ratios], axis=1)

    # Space the objects in the group equally along the X-axis, with some random perturbation
    pos_x = (np.arange(n) - n // 2) * pos_step + rng.uniform(-pos_offset_range, pos_offset_range, size=(n,))
    pos_y = rng.uniform(-pos_offset_range, pos_offset_range, size=(n,))
    pos = np.stack([pos_x, pos_y], axis=1)
    return cat_nx2x3(translation_matrices(pos), rotation_matrices(rotations), scale_matrices(scales), )

def random_groups(n_groups, n_objects_per_group, image_size, rot_range, size_range, aspect_bound, rng=None):
    rng = _get_rng(rng)
    # Random rotations
    rotations = rng.uniform(rot_range[0], rot_range[1], size=(n_groups,))

    # Vary aspect ratio between -log_aspect_bound and -log_aspect_bound/2; tend to generate ellipses
    log_aspect_bound = np.log(aspect_bound) / 2
    ratios = np.exp(rng.uniform(-log_aspect_bound, -log_aspect_bound * 0.5, size=(n_groups,)))
    # Draw random object sizes
    sizes = np.exp(rng.uniform(np.log(size_range[0]), np.log(size_range[1]), size=(n_groups,)))
    scales = np.stack([sizes * ratios, sizes / ratios], axis=1)

    # Compute the per-group step size, border and central position
    step_sizes = scales[:, 0]
    borders = step_sizes * (n_objects_per_group // 2)
    pos = rng.uniform(0.0, 1.0, size=(n_groups, 2))
    pos = borders[:, None] + pos * (np.array(image_size)[None, :] - borders[:, None])

    # Compute per-group matrices
    group_m = cat_nx2x3(translation_matrices(pos), rotation_matrices(rotations))

    all_m = []
    for g_i in range(len(group_m)):
        g_m = group_object_matrices(n_objects_per_group, (-np.radians(7.5), np.radians(7.5)),
                                    scales[g_i], 1.2, sizes[g_i] * 0.15, step_sizes[g_i], rng=rng)
        # Concatenate object matriecs with group matrix
        group_xf = np.repeat(group_m[g_i:g_i + 1], len(g_m), axis=0)
        all_m.append(cat_nx2x3_2(group_xf, g_m))
    return np.concatenate(all_m, axis=0)


def make_sample(image_size, n_random_objects=20, n_groups=4, n_objs_per_group=7, size_range=(10.0, 16.0),
                aspect_bound=4.0, n_circle_verts=65, rng=None):
    rng = _get_rng(rng)

    # Make circle polygon
    circle = make_circle(n_circle_verts)

    draw_objects = True

    # QHull will generate errors if the object is degenerate or in some other circumstances.
    # In these situations, generate another random image
    while draw_objects:
        group_ms = random_groups(n_groups, n_objs_per_group, image_size=image_size, rot_range=(-np.pi * 0.5, np.pi * 0.5),
                                 size_range=size_range, aspect_bound=aspect_bound, rng=rng)
        obj_ms = random_object_matrices(n_random_objects, image_size=image_size, rot_range=(-np.pi * 0.5, np.pi * 0.5),
                                        size_range=size_range, aspect_bound=aspect_bound, rng=rng)
        m = np.append(group_ms, obj_ms, axis=0)

        rgb_img = Image.new('RGB', image_size[::-1])
        label_img = Image.new('I', image_size[::-1])
        rgb_draw = ImageDraw(rgb_img)
        label_draw = ImageDraw(label_img)

        for i in range(len(m)):
            colour = tuple(rng.randint(127, 255, (3,)).astype(np.uint8).tolist())
            p = transform_points(m[i], circle)
            rgb_draw.polygon([tuple(v) for v in p.tolist()], fill=tuple(colour))
            label_draw.polygon([tuple(v) for v in p.tolist()], fill=i + 1)

        x = np.array(rgb_img)
        y = np.array(label_img)

        try:
            boundaries = label_image_to_boundaries(y)
        except QhullError:
            draw_objects = True
        else:
            draw_objects = False

    return x, y, boundaries


class EllipsesDataset (object):
    class ImageAccessor (object):
        def __init__(self, paths, image_load_fn):
            self.paths = paths
            self.image_load_fn = image_load_fn

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, item):
            if isinstance(item, int):
                return self.image_load_fn(self.paths[item])
            else:
                xs = []
                if isinstance(item, slice):
                    indices = range(*item.indices(len(self)))
                elif isinstance(item, np.ndarray):
                    indices = item
                else:
                    raise TypeError('item should be an int/long, a slice or an array, not a {}'.format(
                        type(item)
                    ))
                for i in indices:
                    img = self.image_load_fn(self.paths[i])
                    xs.append(img)
                return xs


    def __init__(self, root_dir, range01=True, rgb_order=True):
        self.root_dir = root_dir

        self.n_images = 0
        self.rgb_paths = []
        self.label_paths = []

        # Find the images
        while True:
            rgb_path = os.path.join(root_dir, 'rgb_{:06d}.png'.format(self.n_images))
            label_path = os.path.join(root_dir, 'labels_{:06d}.png'.format(self.n_images))
            if os.path.exists(rgb_path) and os.path.exists(label_path):
                self.rgb_paths.append(rgb_path)
                self.label_paths.append(label_path)
            else:
                break
            self.n_images += 1

        with open(os.path.join(root_dir, 'convex_hulls.pkl'), 'rb') as f_hulls:
            self.hulls = pickle.load(f_hulls)

        self.range01 = range01
        self.rgb_order = rgb_order

        self.X = self.ImageAccessor(self.rgb_paths, self.load_rgb_image)
        self.y = self.ImageAccessor(self.label_paths, self.load_label_image)


    def get_image_size(self, index):
        path = self.rgb_paths[index]
        img = Image.open(path)
        return img.size[1], img.size[0]

    def load_rgb_image(self, path):
        img = Image.open(path)
        img = np.array(img)
        img = img[:, :, :3]
        if not self.rgb_order:
            img = img[:, :, ::-1]
        if self.range01:
            img = img_as_float(img)
        return img

    def load_label_image(self, path):
        img = Image.open(path)
        return np.array(img)



class EllipsesTrainDataset (EllipsesDataset):
    def __init__(self, range01=True, rgb_order=True):
        super(EllipsesTrainDataset, self).__init__(
            os.path.join(_get_ellipses_root_dir(), 'train'), range01=range01, rgb_order=rgb_order
        )

class EllipsesTestDataset (EllipsesDataset):
    def __init__(self, range01=True, rgb_order=True):
        super(EllipsesTestDataset, self).__init__(
            os.path.join(_get_ellipses_root_dir(), 'test'), range01=range01, rgb_order=rgb_order
        )

