import os
import pickle
import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from skimage.util import img_as_float


def identity_xf(N):
    """
    Construct N identity 2x3 transformation matrices
    :return: array of shape (N, 2, 3)
    """
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    return xf


def inv_nx2x2(X):
    """
    Invert the N 2x2 transformation matrices stored in X; a (N,2,2) array
    :param X: transformation matrices to invert, (N,2,2) array
    :return: inverse of X
    """
    rdet = 1.0 / (X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1])
    y = np.zeros_like(X)
    y[:, 0, 0] = X[:, 1, 1] * rdet
    y[:, 1, 1] = X[:, 0, 0] * rdet
    y[:, 0, 1] = -X[:, 0, 1] * rdet
    y[:, 1, 0] = -X[:, 1, 0] * rdet
    return y

def inv_nx2x3(m):
    """
    Invert the N 2x3 transformation matrices stored in X; a (N,2,3) array
    :param X: transformation matrices to invert, (N,2,3) array
    :return: inverse of X
    """
    m2 = m[:, :, :2]
    mx = m[:, :, 2:3]
    m2inv = inv_nx2x2(m2)
    mxinv = np.matmul(m2inv, -mx)
    return np.append(m2inv, mxinv, axis=2)

def cat_nx2x3_2(a, b):
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

    ab2 = np.matmul(a2, b2)
    abx = ax + np.matmul(a2, bx)
    return np.append(ab2, abx, axis=2)

def cat_nx2x3(*x):
    """
    Multiply the N 2x3 transformations stored in the arrays in `x`
    :param x: transformation matrices, tuple of (N,2,3) arrays
    :return: `x[0] . x[1] . ... . x[N-1]`
    """
    y = x[0]
    for i in range(1, len(x)):
        y = cat_nx2x3_2(y, x[i])
    return y

def translation_matrices(xlats_xy):
    """
    Generate translation matrices
    :param xlats_xy: translations as an (N, 2) array (x,y)
    :return: translations matrices, (N,2,3) array
    """
    N = len(xlats_xy)
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    xf[:, :, 2] = xlats_xy
    return xf

def scale_matrices(scale_xy):
    """
    Generate rotation matrices
    :param scale_xy: X and Y scale factors as an (N, 2) array (x,y)
    :return: scale matrices, (N,2,3) array
    """
    N = len(scale_xy)
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = scale_xy[:, 0]
    xf[:, 1, 1] = scale_xy[:, 1]
    return xf

def rotation_matrices(thetas):
    """
    Generate rotation matrices
    :param thetas: rotation angles in radians as a (N,) array
    :return: rotation matrices, (N,2,3) array
    """
    N = len(thetas)
    c = np.cos(thetas)
    s = np.sin(thetas)
    rot_xf = np.zeros((N, 2, 3), dtype=np.float32)
    rot_xf[:, 0, 0] = rot_xf[:, 1, 1] = c
    rot_xf[:, 1, 0] = s
    rot_xf[:, 0, 1] = -s
    return rot_xf

def transform_points(xf, points_xy):
    """
    Apply the transformation matrix `xf` to the points in `points`.

    :param xf: transformation as a (2, 3) NumPy array
    :param points_xy: points as a (N, 2) array where each point is of the form (x, y), not (y, x)
    :return: (N, 2) array
    """
    return np.tensordot(xf[:2, :2], points_xy, [[1], [1]]).T + xf[:, 2][None, :]



def make_circle(n_verts):
    return np.stack([np.cos(np.linspace(0.0, 2.0 * np.pi, n_verts)),
                     np.sin(np.linspace(0.0, 2.0 * np.pi, n_verts))], axis=1)[:-1] * 0.5



def random_object_matrices(n, image_size, rot_range, size_range, aspect_bound):
    # Rotations
    rotations = np.random.uniform(rot_range[0], rot_range[1], size=(n,))

    # Draw random aspect ratios, keeping area the same
    log_aspect_bound = np.log(aspect_bound) / 2
    ratios = np.exp(np.random.uniform(-log_aspect_bound, log_aspect_bound, size=(n,)))
    # Draw random sizes
    sizes = np.exp(np.random.uniform(np.log(size_range[0]), np.log(size_range[1]), size=(n,)))
    scales = np.stack([sizes * ratios, sizes / ratios], axis=1)

    # Limit position to lie inside the image with a border whose size is the average object size
    pos_border = (size_range[0] + size_range[1]) * 0.5
    pos_x_range = pos_border, image_size[1] - pos_border
    pos_y_range = pos_border, image_size[0] - pos_border
    pos = np.stack([np.random.uniform(pos_x_range[0], pos_x_range[1], size=(n,)),
                    np.random.uniform(pos_y_range[0], pos_y_range[1], size=(n,))], axis=1)
    return cat_nx2x3(translation_matrices(pos), rotation_matrices(rotations), scale_matrices(scales), )

def group_object_matrices(n, rot_range, base_scale, aspect_bound, pos_offset_range, pos_step):
    # Rotations as before, except that the bounds passed will be tighter
    rotations = np.random.uniform(rot_range[0], rot_range[1], size=(n,))

    # Draw random aspect rations (more constrained)
    log_aspect_range = np.log(aspect_bound) / 2
    ratios = np.exp(np.random.uniform(-log_aspect_range, log_aspect_range, size=(n,)))
    # Multiply by a base scale factor that provides the main contribution of aspect ratio
    scales = np.stack([base_scale[0] * ratios, base_scale[1] / ratios], axis=1)

    # Space the objects in the group equally along the X-axis, with some random perturbation
    pos_x = (np.arange(n) - n // 2) * pos_step + np.random.uniform(-pos_offset_range, pos_offset_range, size=(n,))
    pos_y = np.random.uniform(-pos_offset_range, pos_offset_range, size=(n,))
    pos = np.stack([pos_x, pos_y], axis=1)
    return cat_nx2x3(translation_matrices(pos), rotation_matrices(rotations), scale_matrices(scales), )

def random_groups(n_groups, n_objects_per_group, image_size, rot_range, size_range, aspect_bound):
    # Random rotations
    rotations = np.random.uniform(rot_range[0], rot_range[1], size=(n_groups,))

    # Vary aspect ratio between -log_aspect_bound and -log_aspect_bound/2; tend to generate ellipses
    log_aspect_bound = np.log(aspect_bound) / 2
    ratios = np.exp(np.random.uniform(-log_aspect_bound, -log_aspect_bound * 0.5, size=(n_groups,)))
    # Draw random object sizes
    sizes = np.exp(np.random.uniform(np.log(size_range[0]), np.log(size_range[1]), size=(n_groups,)))
    scales = np.stack([sizes * ratios, sizes / ratios], axis=1)

    # Compute the per-group step size, border and central position
    step_sizes = scales[:, 0]
    borders = step_sizes * (n_objects_per_group // 2)
    pos = np.random.uniform(0.0, 1.0, size=(n_groups, 2))
    pos = borders[:, None] + pos * (np.array(image_size)[None, :] - borders[:, None])

    # Compute per-group matrices
    group_m = cat_nx2x3(translation_matrices(pos), rotation_matrices(rotations))

    all_m = []
    for g_i in range(len(group_m)):
        g_m = group_object_matrices(n_objects_per_group, (-np.radians(7.5), np.radians(7.5)),
                                    scales[g_i], 1.2, sizes[g_i] * 0.15, step_sizes[g_i])
        # Concatenate object matriecs with group matrix
        group_xf = np.repeat(group_m[g_i:g_i + 1], len(g_m), axis=0)
        all_m.append(cat_nx2x3_2(group_xf, g_m))
    return np.concatenate(all_m, axis=0)


def label_hulls(labels):
    sample_convex_hulls = [None]

    for label_i in range(1, labels.max() + 1):
        mask = labels == label_i
        mask_y, mask_x = np.where(mask)
        mask_points = np.append(mask_y[:, None], mask_x[:, None], axis=1)
        if len(mask_points) > 0:
            try:
                hull = ConvexHull(mask_points)
            except QhullError:
                raise
            else:
                ch_points = mask_points[hull.vertices, :]
        else:
            ch_points = np.zeros((0, 2))

        sample_convex_hulls.append(ch_points)

    return sample_convex_hulls


def make_sample(image_size, n_random_objects=20, n_groups=4, n_objs_per_group=7, size_range=(8.0, 12.5), aspect_bound=4.0,
                n_circle_verts=65):
    # Make circle polygon
    circle = make_circle(n_circle_verts)

    draw_objects = True

    # QHull will generate errors if the object is degenerate or in some other circumstances.
    # In these situations, generate another random image
    while draw_objects:
        obj_ms = random_object_matrices(n_random_objects, image_size=image_size, rot_range=(-np.pi * 0.5, np.pi * 0.5),
                                        size_range=size_range, aspect_bound=aspect_bound)
        group_ms = random_groups(n_groups, n_objs_per_group, image_size=image_size, rot_range=(-np.pi * 0.5, np.pi * 0.5),
                                 size_range=size_range, aspect_bound=aspect_bound)
        m = np.append(obj_ms, group_ms, axis=0)

        rgb_img = Image.new('RGB', image_size)
        label_img = Image.new('I', image_size)
        rgb_draw = ImageDraw(rgb_img)
        label_draw = ImageDraw(label_img)

        for i in range(len(m)):
            colour = tuple(np.random.randint(127, 255, (3,)).astype(np.uint8).tolist())
            p = transform_points(m[i], circle)
            rgb_draw.polygon([tuple(v) for v in p.tolist()], fill=tuple(colour))
            label_draw.polygon([tuple(v) for v in p.tolist()], fill=i + 1)

        x = np.array(rgb_img)
        y = np.array(label_img)

        try:
            hulls = label_hulls(y)
        except QhullError:
            draw_objects = True
        else:
            draw_objects = False

    return x, y, hulls


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


    def __init__(self, root_dir, n_images, range01, rgb_order):
        self.root_dir = root_dir
        self.n_images = n_images

        self.rgb_paths = [os.path.join(root_dir, 'rgb_{:06d}.png'.format(i)) for i in range(n_images)]
        self.label_paths = [os.path.join(root_dir, 'labels_{:06d}.png'.format(i)) for i in range(n_images)]
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
