import numpy as np
import skimage.transform
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


def label_image_to_boundaries(labels):
    """
    Convert a label image to a list of label boundaries, where each boundary is a (N, [y, x]) array that lists
    the points on the boundary.

    Attempts to use `scipy.spatial.ConvexHull` to select the points that lie on the convex hull, uses all
    points that lie within the label otherwise.

    The returned list will be:
    `[None, boundary_1, boundary_2, .... boundary_N]` where `boundary_N` corresponds to the label whose pixels have
    a value of `N`.

    :param labels: a label image as a (height, width) NumPy integer array
    :return: list of (N,2) NumPy arrays. Element 0 will be None, after which there will be one array for each
        non-zero region in `labels`, e.g. the boundary for pixels with a value of 1 in `labels` will be at index 1
    """
    sample_convex_hulls = [None]

    for label_i in range(1, labels.max() + 1):
        mask = labels == label_i
        mask_y, mask_x = np.where(mask)
        mask_points = np.append(mask_y[:, None] + 0.5, mask_x[:, None] + 0.5, axis=1)
        if len(mask_points) > 0:
            # Compute the convex hull to filter out interior points
            try:
                hull = ConvexHull(mask_points)
            except QhullError:
                # Could not construct convex hull; use all points
                ch_points = mask_points
            else:
                ch_points = mask_points[hull.vertices, :]
        else:
            ch_points = np.zeros((0, 2))

        sample_convex_hulls.append(ch_points)

    return sample_convex_hulls


def _object_mask_to_gt(image_size, object_mask, mini_mask_shape=None):
    """
    Convert a binary object mask to a ground truth box and mask for object detection / instance segmentation

    :param image_size: the size of the original image as a `(height, width)` tuple
    :param object_mask: the binary object mask in the context of the original image as a binary (height, width) shaped
        NumPy array
    :param mini_mask_shape: the size of the mini-mask to extract for training Mask R-CNN, or `None` for no mask
    :return: tuple `(box, mini_mask)` where box is the object box in the form of a
        NumPy array [centre_y, centre_x, height, width] and mini_mask is a binary image if size `mini_mask_shape`
        or `None` if `mini_mask_shape` is `None`
    """
    box = mini_mask = None

    horizontal_indicies = np.where(np.any(object_mask, axis=0))[0]
    vertical_indicies = np.where(np.any(object_mask, axis=1))[0]
    if horizontal_indicies.shape[0] > 0 and vertical_indicies.shape[0] > 0:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1

        if y2 > y1 and x2 > x1:
            box = np.array([y1, x1, y2, x2])

            if mini_mask_shape is not None:
                mini_mask = object_mask[y1:y2, x1:x2].astype(float)
                mini_mask = skimage.transform.resize(mini_mask, mini_mask_shape, order=1, mode='constant')
                mini_mask = (mini_mask >= 0.5).astype(np.float32)
            else:
                mini_mask = None

    return box, mini_mask


def label_image_to_gt(labels, image_size, mini_mask_shape=None):
    """
    Convert a label image to ground truth boxes and masks for object detection / instance segmentation

    :param labels: a label image, dtype=int, 0 value = background, positive values identify object labels
    :param image_size: the size of the original image as a `(height, width)` tuple
    :param mini_mask_shape: the size of the mini-mask to extract for training Mask R-CNN, or `None` for no mask
    :return: tuple `(boxes, mini_masks)` where boxes are object boxes in the form of a
        NumPy array [N, (centre_y, centre_x, height, width)] and mini_masks is a binary image if size
        `(N, mini_mask_shape[0], mini_mask_shape[1])` or `None` if `mini_mask_shape` is `None`
    """
    gt_boxes = []
    if mini_mask_shape is not None:
        gt_masks = []
    else:
        gt_masks = None
    for rp_i in range(labels.max() + 1):
        label_i = rp_i + 1
        m = labels == label_i
        box, mask = _object_mask_to_gt(image_size, m, mini_mask_shape=mini_mask_shape)
        if box is not None:
            gt_boxes.append(box)
        if mask is not None:
            gt_masks.append(mask)

    if len(gt_boxes) > 0:
        gt_boxes = np.stack(gt_boxes, axis=0)
        if mini_mask_shape is not None:
            gt_masks = np.stack(gt_masks, axis=0)
    else:
        gt_boxes = np.zeros((0, 4))
        if mini_mask_shape is not None:
            gt_masks = np.zeros((0,) + mini_mask_shape)

    return gt_boxes, gt_masks


