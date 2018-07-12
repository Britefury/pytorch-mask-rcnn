import numpy as np


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

    Counter-clockwise, +y points downwards

    Where s = sin(theta) and c = cos(theta)

    M = [[ c   s   0 ]
         [ -s  c   0 ]]

    :param thetas: rotation angles in radians as a (N,) array
    :return: rotation matrices, (N,2,3) array
    """
    N = len(thetas)
    c = np.cos(thetas)
    s = np.sin(thetas)
    rot_xf = np.zeros((N, 2, 3), dtype=np.float32)
    rot_xf[:, 0, 0] = rot_xf[:, 1, 1] = c
    rot_xf[:, 1, 0] = -s
    rot_xf[:, 0, 1] = s
    return rot_xf

def transform_points(xf, points_xy):
    """
    Apply the transformation matrix `xf` to the points in `points`.

    :param xf: transformation as a (2, 3) NumPy array
    :param points_xy: points as a (N, 2) array where each point is of the form (x, y), not (y, x)
    :return: (N, 2) array
    """
    return np.tensordot(xf[:2, :2], points_xy, [[1], [1]]).T + xf[:, 2][None, :]

def centre_xf(xf, size):
    """
    Centre the transformations in `xf` around (0,0), where the current centre is assumed to be at the
    centre of an image of shape `size`
    :param xf: transformation matrices, (N,2,3) array
    :param size: image size
    :return: centred transformation matrices, (N,2,3) array
    """
    height, width = size

    # centre_to_zero moves the centre of the image to (0,0)
    centre_to_zero = np.zeros((1, 2, 3), dtype=np.float32)
    centre_to_zero[0, 0, 0] = centre_to_zero[0, 1, 1] = 1.0
    centre_to_zero[0, 0, 2] = -float(width) * 0.5
    centre_to_zero[0, 1, 2] = -float(height) * 0.5

    # centre_to_zero then xf
    xf_centred = cat_nx2x3(xf, centre_to_zero)

    # move (0,0) back to the centre
    xf_centred[:, 0, 2] += float(width) * 0.5
    xf_centred[:, 1, 2] += float(height) * 0.5

    return xf_centred

def grid_to_image_size_nx2x3(N, size):
    """
    Generate matrices that will transform a [-1, 1] grid to the size of the specified image, so that
    transformations will operate on a co-ordinate frame that is the size of the image and centred
    at the image centre. This way rotations will not result in shearing and translations will be
    performed in pixel co-ordinates.

    Two matrix arrays are returned; the grid-to-image matrices and their inverses.

    :param N: the number of matrices to generate
    :param size: the size of the image being transformed as a `(height, with)` tuple
    :return: tuple of `(mtx, mtx_inv)` where `mtx` and `mtx_inv` are the matrices and their inverses
    respectively. `mtx` and `mtx_inv` are NumPy arrays of shape `(N,2,3)`
    """
    x = float(size[1]) / 2.0
    y = float(size[0]) / 2.0

    mtx = identity_xf(N)
    mtx[:, 0, 0] = x
    mtx[:, 1, 1] = y

    mtx_inv = identity_xf(N)
    mtx_inv[:, 0, 0] = 1.0 / x
    mtx_inv[:, 1, 1] = 1.0 / y

    return mtx, mtx_inv

def grid_to_px_nx2x3(N, src_size, tgt_size=None):
    """
    Generate matrices that will transform a [-1, 1] grid to an image of size `size`.

    :param N: the number of matrices to generate
    :param tgt_size: the size of the output image as a `(height, with)` tuple
    :param src_size: the size of the image being transformed as a `(height, with)` tuple
    :return: tuple of `(grid_to_tgt_px, src_px_to_grid)` where:
        `grid_to_tgt_px` transforms a [-1,1] grid to a `tgt_size` pixel image
        `src_px_to_grid` transforms a `src_size` pixel image to a [-1, 1] grid
    `grid_to_tgt_px` and `src_px_to_grid` are NumPy arrays of shape `(N,2,3)`
    """
    src_x = float(src_size[1]-1) / 2.0
    src_y = float(src_size[0]-1) / 2.0

    if tgt_size is not None:
        tgt_x = float(tgt_size[1]-1) / 2.0
        tgt_y = float(tgt_size[0]-1) / 2.0
    else:
        tgt_x = src_x
        tgt_y = src_y

    grid_to_tgt_px = identity_xf(N)
    grid_to_tgt_px[:, 0, 0] = tgt_x
    grid_to_tgt_px[:, 1, 1] = tgt_y
    grid_to_tgt_px[:, 0, 2] = tgt_x
    grid_to_tgt_px[:, 1, 2] = tgt_y

    src_px_to_grid = identity_xf(N)
    src_px_to_grid[:, 0, 0] = 1.0 / src_x
    src_px_to_grid[:, 1, 1] = 1.0 / src_y
    src_px_to_grid[:, 0, 2] = -1.0
    src_px_to_grid[:, 1, 2] = -1.0

    return grid_to_tgt_px, src_px_to_grid

def apply_grid_to_image(mtx, size):
    """
    Apply apsect ratio correction (see `aspect_nx2x3` function) for images of size `size` to the matrices in `mtx`

    :param mtx: transformation matrices as a `(N,2,3)` arrays
    :param size: image size as a `(height, width)` tuple
    :return: corrected matrices as a `(N,2,3)` array
    """
    asp, asp_inv = grid_to_image_size_nx2x3(len(mtx), size)
    return cat_nx2x3(asp_inv, cat_nx2x3(mtx, asp))


def axis_angle_rotation_matrices(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    :param axis: the axes
    :param theta: the angles in radians
    :return: matrices as an (N,3,3) array
    """
    # Normalize
    axis = axis/np.sqrt(np.sum(axis*axis, axis=1, keepdims=True))
    a = np.cos(theta/2)
    axis_sin_theta = -axis*np.sin(theta/2)[:, None]
    b = axis_sin_theta[:, 0]
    c = axis_sin_theta[:, 1]
    d = axis_sin_theta[:, 2]
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot.transpose(2, 0, 1)


def axis_to_axis_rotation_matrices(axes_a, axes_b, normalize=False, epsilon=1e-12):
    """
    Return the rotation matrices that rotate the axes in `axes_a` onto the axes in `axes_b`

    :param axes_a: source axes
    :param axes_b: target axes
    :param normalize: if True, normalize axes_a and axes_b before use
    :param epsilon: if normalize is True, use this epsilon value to prevent divide by 0
    :return: matrices as an (N,3,3) array
    """
    if normalize:
        axes_a = axes_a / np.maximum(np.sqrt((axes_a * axes_a).sum(axis=1, keepdims=True)), epsilon)
        axes_b = axes_b / np.maximum(np.sqrt((axes_b * axes_b).sum(axis=1, keepdims=True)), epsilon)
    rotation_axes = np.cross(axes_a, axes_b)
    rotation_cos_thetas = (axes_a * axes_b).sum(axis=1)

    # Detect cases where axes are perfectly aligned or opposed
    aligned_or_opposed = (rotation_cos_thetas > (1.0 - epsilon)) | (rotation_cos_thetas < (-1.0 + epsilon))
    if aligned_or_opposed.any():
        # Choose most perpendicular base axis to A
        most_perp_to_a = (np.argmin(abs(axes_a), axis=1)[:, None] == np.array([[0, 1, 2]])).astype(float)
        # Get axis perpendicular to both
        perp_rot_axis = np.cross(axes_a, most_perp_to_a)
        # Normalize
        perp_to_a = perp_rot_axis / np.maximum(np.sqrt((perp_rot_axis * perp_rot_axis).sum(axis=1, keepdims=True)), epsilon)
        # Replace rotation axes with valid axes as the result of the initial cross product will be degenerate
        rotation_axes[aligned_or_opposed] = perp_to_a[aligned_or_opposed]
    rotation_thetas = np.arccos(np.clip(rotation_cos_thetas, -1.0, 1.0))
    return axis_angle_rotation_matrices(rotation_axes, rotation_thetas)



def compute_transformed_image_padding(image_size, xf):
    """
    Compute the padding required to completely contain an image of size `image_size` transformed by the transformations
    in `xf`

    :param image_size: image size as a tuple `(height, width)`
    :param xf: transformations as an `(N, 2, 3)` array
    :return: tuple `(padding, padded_size)` where `padding` is an `(N, 4)` array where each row is
            [pad_top, pad_bottom, pad_left, pad_right] and `padded_size` is an `(N, 2)` array where each
            row is [padded_height, padded_width]
    """
    if len(xf) != 1:
        raise NotImplementedError('Check that this works with batch size > 1')
    height, width = image_size

    xf_corners = centre_xf(xf, image_size)

    # (4, 2) array where each row is a corner; note co-ordinates used for transformations are [x,y] NOT [y,x]
    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    corners_z = np.matmul(xf_corners[:, :, :2], corners.T).transpose(0, 2, 1) + xf_corners[:, :, 2][: None, :]

    left_top = np.floor(corners_z.min(axis=1)).astype(int)
    right_bottom = np.ceil(corners_z.max(axis=1)).astype(int)

    pad_lower = -left_top
    pad_upper = right_bottom - np.array([[width, height]])

    padding = np.stack([pad_lower[:, 1], pad_upper[:, 1], pad_lower[:, 0], pad_upper[:, 0]], axis=1)
    padded_size = right_bottom - left_top

    return padding, padded_size[:, ::-1]

