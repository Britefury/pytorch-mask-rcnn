import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops
import skimage.transform
from examples import affine_transforms
import cv2


def _object_mask_to_gt(image_size, object_mask, mini_mask_shape=None, box_border=0.0, box_border_min=0):
    """
    Convert a binary object mask to a ground truth box and mask for object detection / instance segmentation

    :param image_size: the size of the original image as a `(height, width)` tuple
    :param object_mask: the binary object mask in the context of the original image as a binary (height, width) shaped
        NumPy array
    :param mini_mask_shape: the size of the mini-mask to extract for training Mask R-CNN, or `None` for no mask
    :param box_border: grow the size of the box by this proportion - distributed equally on both sides
        e.g. if the objects bounding box extends from 10 to 20 in the y-axis, and box_border=0.4, then the
        box will be expanded to extend from 8 to 22 (its size was 10 pixels, add 40%; 20% above and 20% below)
    :param box_border_min: grow the size of the box by at least this number of pixels (in image space)
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

        if box_border > 0.0 or box_border_min > 0:
            # Grow box
            h = float(y2 - y1)
            w = float(x2 - x1)
            y_border = max(int(round(h * box_border * 0.5)), box_border_min)
            x_border = max(int(round(w * box_border * 0.5)), box_border_min)

            y1 = max(y1 - y_border, 0)
            x1 = max(x1 - x_border, 0)
            y2 = min(y2 + y_border, image_size[0])
            x2 = min(x2 + x_border, image_size[1])

        if y2 > y1 and x2 > x1:
            box = np.array([y1, x1, y2, x2])

            if mini_mask_shape is not None:
                mini_mask = object_mask[y1:y2, x1:x2].astype(float)
                mini_mask = skimage.transform.resize(mini_mask, mini_mask_shape, order=1, mode='constant')
                mini_mask = (mini_mask >= 0.5).astype(np.float32)
            else:
                mini_mask = None

    return box, mini_mask

_ROOT_2 = math.sqrt(2.0)

def label_image_to_gt(labels, image_size, mini_mask_shape=None, box_border=0.0, box_border_min=0):
    """
    Convert a label image to ground truth boxes and masks for object detection / instance segmentation

    :param labels: a label image, dtype=int, 0 value = background, positive values identify object labels
    :param image_size: the size of the original image as a `(height, width)` tuple
    :param mini_mask_shape: the size of the mini-mask to extract for training Mask R-CNN, or `None` for no mask
    :param box_border: grow the size of the box by this proportion - distributed equally on both sides
        e.g. if the objects bounding box extends from 10 to 20 in the y-axis, and box_border=0.4, then the
        box will be expanded to extend from 8 to 22 (its size was 10 pixels, add 40%; 20% above and 20% below)
    :param box_border_min: grow the size of the box by at least this number of pixels (in image space)
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
        box, mask = _object_mask_to_gt(image_size, m, mini_mask_shape=mini_mask_shape, box_border=box_border,
                                       box_border_min=box_border_min)
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
