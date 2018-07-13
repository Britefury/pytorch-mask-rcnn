import numpy as np
import skimage.measure

from maskrcnn.model.utils import compute_overlaps


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