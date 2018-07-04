import math
import numpy as np
import skimage.transform, skimage.measure
import cv2
import torch, torch.nn.functional as F
from maskrcnn.model.utils import compute_overlaps
from . import affine_transforms, augmentation

def _compute_precision(iou, threshold):
    matches = iou > threshold
    true_pos = matches.sum(axis=1) == 1
    false_pos = matches.sum(axis=0) == 0
    false_neg = matches.sum(axis=1) == 0
    return true_pos.sum(), false_pos.sum(), false_neg.sum()



def mean_precision(true_labels, pred_labels, thresh_low=0.5, thresh_high=1.0):
    # This function was largely copied from Heng CherKeng's code for the 2018 DS bowl competition
    intersection, _, _ = np.histogram2d(true_labels.flatten(), pred_labels.flatten(),
                                        bins=(np.arange(true_labels.max()+2), np.arange(pred_labels.max()+2)))

    area_true, _ = np.histogram(true_labels.flatten(), bins=np.arange(true_labels.max() + 2))
    area_pred, _ = np.histogram(pred_labels.flatten(), bins=np.arange(pred_labels.max() + 2))

    union = area_true[:, None] + area_pred[None, :] - intersection

    # Remove background
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union = np.maximum(union, 1.0e-9)

    iou = intersection / union

    thresholds = np.arange(thresh_low, thresh_high, 0.05)

    prec = []
    for t in thresholds:
        tp, fp, fn = _compute_precision(iou, t)
        p = tp / max((tp + fp + fn), 1.0e-9)
        prec.append(p)

    return np.mean(prec)


def match_labels(true_labels, pred_labels):
    intersection, _, _ = np.histogram2d(true_labels.flatten(), pred_labels.flatten(),
                                        bins=(np.arange(true_labels.max()+2), np.arange(pred_labels.max()+2)))

    matches = []
    while True:
        ix = np.argmax(intersection.flatten())
        match = np.unravel_index(ix, intersection.shape)
        if intersection[match[0], match[1]] == 0:
            break
        matches.append(match)
        intersection[match[0], :] = 0
        intersection[:, match[1]] = 0

    matches = np.array(matches)

    true_to_pred = np.zeros((true_labels.max()+1,), dtype=int)
    pred_to_true = np.zeros((pred_labels.max()+1,), dtype=int)

    true_to_pred.fill(-1)
    true_to_pred[matches[:, 0]] = matches[:, 1]

    pred_to_true.fill(-1)
    pred_to_true[matches[:, 1]] = matches[:, 0]


    return matches, true_to_pred, pred_to_true



def match_labels_by_iou(true_labels, pred_labels, min_iou=None):
    # Largely copied this from Heng CherKeng's code for the 2018 DS bowl competition
    intersection, _, _ = np.histogram2d(true_labels.flatten(), pred_labels.flatten(),
                                        bins=(np.arange(true_labels.max()+2), np.arange(pred_labels.max()+2)))

    area_true, _ = np.histogram(true_labels.flatten(), bins=np.arange(true_labels.max() + 2))
    area_pred, _ = np.histogram(pred_labels.flatten(), bins=np.arange(pred_labels.max() + 2))

    union = area_true[:, None] + area_pred[None, :] - intersection

    # Remove background
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union = np.maximum(union, 1.0e-9)

    iou = intersection / union

    if min_iou is not None:
        iou[iou < min_iou] = 0.0

    matches = []
    while True:
        ix = np.argmax(iou.flatten())
        match = np.unravel_index(ix, iou.shape)
        if iou[match[0], match[1]] == 0.0:
            break
        matches.append(match)
        iou[match[0], :] = 0
        iou[:, match[1]] = 0

    matches = np.array(matches) + 1

    true_to_pred = np.zeros((true_labels.max()+1,), dtype=int)
    pred_to_true = np.zeros((pred_labels.max()+1,), dtype=int)

    true_to_pred.fill(-1)
    if len(matches) > 0:
        true_to_pred[matches[:, 0]] = matches[:, 1]

    pred_to_true.fill(-1)
    if len(matches) > 0:
        pred_to_true[matches[:, 1]] = matches[:, 0]


    return matches, true_to_pred, pred_to_true



def evaluate_box_predictions(det_boxes, true_boxes):
    dets = float(det_boxes.shape[0])
    reals = float(true_boxes.shape[0])

    if len(det_boxes) > 0 and len(true_boxes) > 0:
        overlaps = compute_overlaps(det_boxes, true_boxes)
        hits = (overlaps >= 0.5).sum()
    else:
        hits = 0.0

    acc = float(hits) / float(dets + reals - hits)

    return (acc, float(dets), reals)



def evaluate_box_predictions_from_labels(det_boxes, true_labels, image_size, box_border=0.0, box_border_min=0):
    # Get region props
    rprops = skimage.measure.regionprops(true_labels)
    gt_boxes = []
    for label_i, rp in enumerate(rprops):
        m = true_labels == (label_i + 1)
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
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
                gt_boxes.append(np.array([y1, x1, y2, x2]))
    if len(gt_boxes) > 0:
        gt_boxes = np.stack(gt_boxes, axis=0)
    else:
        gt_boxes = np.zeros((0, 4))

    return evaluate_box_predictions(det_boxes, gt_boxes)


def mrcnn_transformed_image_padding(image_size, xf, net_block_size):
    """
    Compute the padding required to fully contain the transformed version of an image
    of size `image_size` when transformed using the transformation `xf`, with additional
    padding to ensure that the resulting image's size is divisible by `net_block_size`

    :param image_size: the size of the image as a `(height, width)` tuple
    :param xf: The transformation matrix as a (1, 2, 3) matrix
    :param net_block_size: Network block size as a `(block_height, block_width)` tuple
    :return: `(window, final_xf, block_padded_shape)`
        window: `(1, 4)` array giving the valid window size to be passed to Mask-RCNN
        final_xf: the transformation that applies padding and correct centreing, (1, 2, 3) array
        block_padded_shape: `(height, width)` tuple giving the size of the padded image
    """
    image_centre_xy = np.array(image_size[::-1]) * 0.5

    # Compute the padding required to completely contain the transformed image
    xf_padding, xf_padded_size = affine_transforms.compute_transformed_image_padding(image_size, xf)
    xf_padding = xf_padding[0]
    xf_padded_size = xf_padded_size[0]

    # Compute padding require to fit neural network blocks
    block_padded_shape = augmentation.round_up_shape(xf_padded_size, net_block_size)
    block_padding = augmentation.compute_padding(xf_padded_size, block_padded_shape)

    # Total image padding
    img_padding = [(block_padding[0][0] + xf_padding[0], block_padding[0][1] + xf_padding[1]),
                   (block_padding[1][0] + xf_padding[2], block_padding[1][1] + xf_padding[3])]
    pad_offset_xy = np.array([img_padding[1][0], img_padding[0][0]])
    padded_centre_xy = pad_offset_xy + image_centre_xy

    # Combine transformation with translations that apply centreing and padding
    centre_to_origin = augmentation.translation_matrices(-image_centre_xy[None, :])
    origin_to_pad_centre = augmentation.translation_matrices(padded_centre_xy[None, :])
    final_xf = augmentation.cat_nx2x3(origin_to_pad_centre, xf, centre_to_origin)

    # y1, x1, y2, x2
    window = np.array([
        float(block_padding[0][0]), float(block_padding[1][0]),
        float(block_padded_shape[0] - block_padding[0][1]), float(block_padded_shape[1] - block_padding[1][1])])

    window = window[None, ...]

    return window, final_xf, block_padded_shape


def mrcnn_detections_to_label_image(image_size, det_scores, det_class_id, det_boxes, mrcnn_mask, mask_nms_thresh=0.9):
    """
    Generate a label image from Mask-RCNN detections

    :param image_size: the size of the label image as `(height, width)`
    :param det_scores: Detection scores; (1, N) or (N,) array
    :param det_class_id: Detection class IDs; (1, N) or (N,) array
    :param det_boxes: Detection boxes; (1, N, 4) or (N, 4) array, each box is [y1, x1, y2, x2]
    :param mrcnn_mask: Detection masks; (1, N, H, W) or (N, H, W) array, where H,W is the mask size
    :param mask_nms_thresh: Mask NMS threshold
    :return: (label_img, cls_img)
        label_img; (height, width) label image with different integer ID for each instance
        cls_img; (height, width) with pixels labelled by class
    """
    if det_scores.ndim == 2:
        # Remove batch dimension
        det_boxes = det_boxes[0]
        det_scores = det_scores[0]
        det_class_id = det_class_id[0]
        mrcnn_mask = mrcnn_mask[0]

    # mrcnn_mask: (D, H, W, C); D=detection, H,W=height,width, C=detection class

    y = np.zeros(image_size, dtype=int)
    y_cls = np.zeros(image_size, dtype=int)
    order = np.argsort(det_scores)[::-1]

    label_i = 1
    for i in order:
        box = det_boxes[i]
        y1, x1, y2, x2 = box
        y1 = int(y1)
        x1 = int(x1)
        y2 = int(math.ceil(y2))
        x2 = int(math.ceil(x2))
        y1c = max(y1, 0)
        x1c = max(x1, 0)
        y2c = min(y2, image_size[0])
        x2c = min(x2, image_size[1])
        if y2c > y1c and x2c > x1c:
            mask = mrcnn_mask[i, :, :]
            img_mask = skimage.transform.resize(mask, (y2 - y1, x2 - x1), order=1, mode='constant')
            img_mask_bin = img_mask > 0.5
            img_mask_bin = img_mask_bin[y1c - y1:img_mask_bin.shape[0] - (y2 - y2c),
                           x1c - x1:img_mask_bin.shape[1] - (x2 - x2c)]

            if img_mask_bin.shape[0] > 0 and img_mask_bin.shape[1] > 0:
                # Empty pixels mask
                empty = y[y1c:y2c, x1c:x2c] == 0

                if float((img_mask_bin & empty).sum()) / float(img_mask_bin.sum() + 1e-8) > mask_nms_thresh:
                    y[y1c:y2c, x1c:x2c][empty & img_mask_bin] = label_i
                    y_cls[y1c:y2c, x1c:x2c][empty & img_mask_bin] = det_class_id[i]
                    label_i += 1
    return y, y_cls


def mrcnn_augmented_detections(net, x, xf, net_block_size):
    """
    Generate augmented detections with a Mask-RCNN network

    :param net: The Mask-RCNN network model
    :param x: an image as a NumPy array; (height, width, channels)
    :param xf: An affine transformation matrix used for augmentation, (1, 2, 3) aray
    :param net_block_size: the network block size; image will be rounded up to this before passing to
        network for inference
    :return: `(detections, misc_detect, xf_padded, inv_xf_padded, padded_shape)`
        detections: detections returned by `net.detect_forward`
        misc_detect: misc detection data returned by `net.detect_forward`
        xf_padded: augmentation transformation with padding; useful if you want to invert the augmentation present in
            e.g misc_detect
        inv_xf_padded: inverse of xf_padded
        padded_shape: the (height, width) shape of the image with padding
    """
    x_4 = x.transpose(2, 0, 1)[None, ...].astype(np.float32)

    # Get window, padded transformation and padded shape
    window, xf_padded, padded_shape = mrcnn_transformed_image_padding(x.shape[:2], xf, net_block_size)
    inv_xf_padded = augmentation.inv_nx2x3(xf_padded)

    # Apply scaling factors so that xf operates on [-1, 1] Torch grid
    grid_to_tgt_px, src_px_to_grid = augmentation.grid_to_px_nx2x3(1, x.shape[:2], padded_shape)
    final_xf_padded_torch = augmentation.cat_nx2x3(src_px_to_grid, inv_xf_padded, grid_to_tgt_px)

    # Variables for image and transformation
    X_var = torch.autograd.Variable(torch.from_numpy(x_4).float().cuda(), volatile=True)
    xf_padded_var = torch.autograd.Variable(torch.from_numpy(final_xf_padded_torch).float().cuda(), volatile=True)

    # Affine grid for applying augmentation
    grid_X_padded = F.affine_grid(xf_padded_var, torch.Size((1, 1, padded_shape[0], padded_shape[1])))

    # Apply augmentation
    X_aug = F.grid_sample(X_var, grid_X_padded)

    # Detect
    detections = net.detect_forward_np(X_aug, window)

    return detections, xf_padded, inv_xf_padded, padded_shape


def deaugment_mrcnn_detections(detections, inv_xf_padded):
    """
    De-augment detections from `mrcnn_augmented_detections`

    :param detections: detections from a single image returned by `mrcnn_augmented_detections`
    :param inv_xf_padded: the inverse padded transformation matrix, returned by `mrcnn_augmented_detections`
    :return: `(det_boxes, det_class_ids, det_scores, mrcnn_masks)`
        det_boxes: (N, 4) array of detection boxes, each box is [y1, x1, y2, x2]
        det_class_ids: (N,) array giving the class of each detection
        det_scores: (N,) array giving the score of each detection
        mrcnn_masks: list of (H,W) arrays giving each mask; they are transformed to the space of the original image so
            they have different sizes
    """
    # Get detections for image 0 in augmented space
    det_boxes_aug, det_class_ids, det_scores, mrcnn_mask_aug = detections[0]

    # Get corners of detection boxes
    det_aug_xy_00 = np.stack([det_boxes_aug[0, :, None, 1], det_boxes_aug[0, :, None, 0]], axis=2)
    det_aug_xy_01 = np.stack([det_boxes_aug[0, :, None, 3], det_boxes_aug[0, :, None, 0]], axis=2)
    det_aug_xy_10 = np.stack([det_boxes_aug[0, :, None, 1], det_boxes_aug[0, :, None, 2]], axis=2)
    det_aug_xy_11 = np.stack([det_boxes_aug[0, :, None, 3], det_boxes_aug[0, :, None, 2]], axis=2)
    det_corners_aug_xy = np.concatenate([det_aug_xy_00, det_aug_xy_01, det_aug_xy_10, det_aug_xy_11], axis=1)
    det_corners_aug_flat_xy = det_corners_aug_xy.reshape((-1, 2))
    # Transform detection boxes from augmented space to image space
    det_corners_flat_xy = augmentation.transform_points(inv_xf_padded[0], det_corners_aug_flat_xy)
    det_corners_xy = det_corners_flat_xy.reshape(det_corners_aug_xy.shape)

    # Compute detection boxes in image space
    det_boxes_int = np.concatenate([np.floor(det_corners_xy[:, :, ::-1].min(axis=1)),
                                    np.ceil(det_corners_xy[:, :, ::-1].max(axis=1))], axis=1).astype(int)
    # Top left corner of each box, rounded
    det_boxes_topleft_xy = np.floor(det_corners_xy.min(axis=1))

    # We now need to transform the masks.
    # We need to compute the transformation applied to each mask. To do this, we compute the transformation
    # required to transform the masks to their de-augmented co-ordinate frames

    # We do this using an affine triangle to triangle transformation matrix

    # Source triangle: top-left, top-right and bottom-left corners in mask space
    mask_size = mrcnn_mask_aug.shape[2:4]
    src_tris_xy = np.array([[
        [0.0, 0.0], [float(mask_size[1]), 0.0], [0.0, float(mask_size[1])]
    ]])

    # Target triangles; de-augmented detection box corners, relative to top left of de-augmented box
    tgt_tris_xy = det_corners_xy[:, :3, :] - np.floor(det_boxes_topleft_xy[:, None, :])

    # Compute mask matrices
    mask_matrices = augmentation.triangle_to_triangle_matrices(src_tris_xy, tgt_tris_xy)

    # De-augment the detection masks
    mrcnn_mask = []
    for det_i in range(len(det_boxes_int)):
        y1, x1, y2, x2 = det_boxes_int[det_i]
        mask_aug = mrcnn_mask_aug[0, det_i, :, :, det_class_ids[0, det_i]]
        img_mask = cv2.warpAffine(mask_aug, mask_matrices[det_i], (x2 - x1, y2 - y1))
        mrcnn_mask.append(img_mask)

    return det_boxes_int, det_class_ids[0], det_scores[0], mrcnn_mask


def mrcnn_augmented_detections_to_label_image(image_size, det_scores, det_class_id, det_boxes, mrcnn_mask,
                                              mask_nms_thresh=0.9):
    """
    Generate a label image from augmented Mask-RCNN detections
    Use this rather than `mrcnn_detections_to_label_image` if the detections were augmented using
    `mrcnn_augmented_detections`

    :param image_size: the size of the label image as `(height, width)`
    :param det_scores: Detection scores; (1, N) or (N,) array
    :param det_class_id: Detection class IDs; (1, N) or (N,) array
    :param det_boxes: Detection boxes; (1, N, 4) or (N, 4) array, each box is [y1, x1, y2, x2]
    :param mrcnn_mask: Detection masks; list of (H, W) detection masks whose size corresponds to that of the
        detection boxes
    :param mask_nms_thresh: Mask NMS threshold
    :return: (label_img, cls_img)
        label_img; (height, width) label image with different integer ID for each instance
        cls_img; (height, width) with pixels labelled by class
    """
    if det_scores.ndim == 2:
        # Remove batch dimension
        det_boxes = det_boxes[0]
        det_class_id = det_class_id[0]
        det_scores = det_scores[0]

    # mrcnn_mask: (D, H, W, C); D=detection, H,W=height,width, C=detection class

    y = np.zeros(image_size, dtype=int)
    y_cls = np.zeros(image_size, dtype=int)
    order = np.argsort(det_scores)[::-1]

    label_i = 1
    for i in order:
        box = det_boxes[i]
        y1, x1, y2, x2 = box
        y1 = int(y1)
        x1 = int(x1)
        y2 = int(math.ceil(y2))
        x2 = int(math.ceil(x2))
        y1c = max(y1, 0)
        x1c = max(x1, 0)
        y2c = min(y2, image_size[0])
        x2c = min(x2, image_size[1])
        if y2c > y1c and x2c > x1c:
            mask = mrcnn_mask[i]
            mask_bin = mask > 0.5
            mask_bin = mask_bin[y1c - y1:mask_bin.shape[0] - (y2 - y2c), x1c - x1:mask_bin.shape[1] - (x2 - x2c)]

            if mask_bin.shape[0] > 0 and mask_bin.shape[1] > 0:
                # Empty pixels mask
                empty = y[y1c:y2c, x1c:x2c] == 0

                if float((mask_bin & empty).sum()) / float(mask_bin.sum() + 1e-8) > mask_nms_thresh:
                    y[y1c:y2c, x1c:x2c][empty & mask_bin] = label_i
                    y_cls[y1c:y2c, x1c:x2c][empty & mask_bin] = det_class_id[i]
                    label_i += 1
    return y, y_cls



def mask_intersection(box_a, box_b, mask_a, mask_b):
    box_inter = np.append(np.maximum(box_a[:2], box_b[:2]), np.minimum(box_a[2:], box_b[2:]), axis=0)
    box_i_in_a = box_inter - np.repeat(box_a[:2], 2, axis=0)
    box_i_in_b = box_inter - np.repeat(box_b[:2], 2, axis=0)
    mask_a_in_i = mask_a[box_i_in_a[0]:box_i_in_a[2], box_i_in_a[1]:box_i_in_a[3]]
    mask_b_in_i = mask_b[box_i_in_b[0]:box_i_in_b[2], box_i_in_b[1]:box_i_in_b[3]]

    return box_inter, mask_a_in_i, mask_b_in_i


def mask_union(box_a, box_b, mask_a, mask_b):
    box_union = np.append(np.minimum(box_a[:2], box_b[:2]), np.maximum(box_a[2:], box_b[2:]), axis=0)
    box_u_sz = tuple(box_union[2:] - box_union[:2])
    box_a_in_u = box_a - np.tile(box_union[:2], [2])
    box_b_in_u = box_b - np.tile(box_union[:2], [2])
    mask_a_in_u = np.zeros(box_u_sz, dtype=mask_a.dtype)
    mask_b_in_u = np.zeros(box_u_sz, dtype=mask_b.dtype)
    mask_a_in_u[box_a_in_u[0]:box_a_in_u[2], box_a_in_u[1]:box_a_in_u[3]] = mask_a
    mask_b_in_u[box_b_in_u[0]:box_b_in_u[2], box_b_in_u[1]:box_b_in_u[3]] = mask_b

    return box_union, mask_a_in_u, mask_b_in_u


def mask_agreement(mask_a, mask_b):
    mask_a_agg = mask_a * 2 - 1
    mask_b_agg = mask_b * 2 - 1
    pos_mask = (mask_a_agg >= 0.0) | (mask_b_agg >= 0.0)
    return (mask_a_agg * mask_b_agg * pos_mask).sum() / max(pos_mask.sum(), 1.0)


def mask_real_iou(mask_a, mask_b):
    return np.minimum(mask_a, mask_b).sum() / (np.maximum(mask_a, mask_b).sum() + 1.0e-12)


def mask_bin_iou(mask_a, mask_b):
    mask_a = mask_a >= 0.5
    mask_b = mask_b >= 0.5
    return (mask_a & mask_b).sum() / ((mask_a | mask_b).sum() + 1.0e-12)


def greedily_merge_augmented_detections(aug_dets, detection_proportion_thresh=0.25, box_overlap_thresh=0.25,
                                        similarity_thresh=None, similarity_function='agreement', progress_iter_func=None):
    """
    Greedily merge augmented detections. The detections will be merged and combined to form a hopefully improved set
    of detections.

    Gather outputs from the `deaugment_mrcnn_detections` function and put in a list to pass as the `aug_dets`
    parameter.

    :param aug_dets: Augmented detections.
    :param detection_proportion_thresh: The proportion of images in which a detection must occur for it to pass.
        This way, if an object is detected in too few of the images, then it will be rejected
    :param box_overlap_thresh: Detection box IoU must be at least this value for the merger to consider merging
        the detections
    :param similarity_thresh: Detections whose similarity is at least this value will be merged.
        If `similarity_function` is 'agreement', the default for `similarity_thresh` is -0.5.
        If `similarity_function` is 'real_iou', the default for `similarity_thresh` is 0.25.
        If `similarity_function` is 'bin_iou', the default for `similarity_thresh` is 0.333.
    :param similarity_function: 'agreement', 'real_iou' or 'bin_iou' for agreement measurement,
        real-valued intersection-over-union or binary intersection-over-union
    :param progress_iter_func: pass tqdm to get a progress bar
    :return: `(merged_boxes, merged_class_ids, merged_scores, merged_masks)`
        merged_boxes: (N, 4) array of detection boxes, each box is [y1, x1, y2, x2]
        merged_class_ids: (N,) array giving the class of each detection
        merged_scores: (N,) array giving the score of each detection
        merged_masks: list of (H,W) arrays giving each mask; they are transformed to the space of the original image so
            they have different sizes
    """
    if progress_iter_func is None:
        progress_iter_func = lambda x: x

    if similarity_function == 'agreement':
        similarity_fn = mask_agreement
        default_similarity_thresh = -0.5
    elif similarity_function == 'real_iou':
        similarity_fn = mask_real_iou
        default_similarity_thresh = 0.25
    elif similarity_function == 'bin_iou':
        similarity_fn = mask_bin_iou
        default_similarity_thresh = 0.333
    else:
        raise ValueError('similarity_function should be agreement, real_iou or bin_iou, not {}'.format(
            similarity_function))

    if similarity_thresh is None:
        similarity_thresh = default_similarity_thresh

    det_boxes, det_class_ids, det_scores, mrcnn_masks = aug_dets[0]
    det_count = np.ones(det_class_ids.shape, dtype=float)

    for aug_i in progress_iter_func(range(1, len(aug_dets))):
        det_boxes_i, det_class_ids_i, det_scores_i, mrcnn_masks_i = aug_dets[aug_i]
        b_unused = np.ones(det_class_ids_i.shape, dtype=bool)

        if len(det_boxes) > 0 and len(det_boxes_i) > 0:
            box_overlaps = compute_overlaps(det_boxes, det_boxes_i)
            mask_similarity = -np.ones_like(box_overlaps)


            det_as, det_bs = np.where(box_overlaps >= box_overlap_thresh)

            for det_a in det_as:
                for det_b in det_bs:
                    _, mask_a, mask_b = mask_union(
                        det_boxes[det_a], det_boxes_i[det_b], mrcnn_masks[det_a], mrcnn_masks_i[det_b])
                    similarity = similarity_fn(mask_a, mask_b)
                    mask_similarity[det_a, det_b] = similarity

            while True:
                det_a, det_b = np.unravel_index(np.argmax(mask_similarity.flatten()), mask_similarity.shape)
                if mask_similarity[det_a, det_b] < similarity_thresh:
                    break

                b_unused[det_b] = False
                mask_similarity[det_a, :] = -1
                mask_similarity[:, det_b] = -1

                # Merge
                box_u, mask_a_u, mask_b_u = mask_union(
                    det_boxes[det_a], det_boxes_i[det_b], mrcnn_masks[det_a], mrcnn_masks_i[det_b])
                det_boxes[det_a] = box_u
                det_scores[det_a] = det_scores[det_a] + det_scores_i[det_b]
                mrcnn_masks[det_a] = mask_a_u + mask_b_u
                det_count[det_a] += 1

        det_boxes = np.append(det_boxes, det_boxes_i[b_unused], axis=0)
        det_class_ids = np.append(det_class_ids, det_class_ids_i[b_unused], axis=0)
        det_scores = np.append(det_scores, det_scores_i[b_unused], axis=0)
        mrcnn_masks.extend([mrcnn_masks_i[j] for j in np.where(b_unused)[0]])
        det_count = np.append(det_count, np.ones((int(b_unused.sum()),), dtype=float), axis=0)

    # Average over detections
    det_scores = det_scores / det_count
    mrcnn_masks = [mask / n_dets for mask, n_dets in zip(mrcnn_masks, det_count)]

    n_dets_thresh = round(len(aug_dets) * detection_proportion_thresh)
    pass_mask = det_count >= n_dets_thresh

    merged_boxes = det_boxes[pass_mask]
    merged_class_ids = det_class_ids[pass_mask]
    merged_scores = det_scores[pass_mask]
    merged_masks = [mask for mask, p in zip(mrcnn_masks, pass_mask) if p]

    return merged_boxes, merged_class_ids, merged_scores, merged_masks

