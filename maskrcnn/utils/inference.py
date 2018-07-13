import math

import numpy as np
import skimage.transform


def mrcnn_detections_to_label_image(image_size, det_scores, det_class_id, mask_boxes, mrcnn_mask, mask_nms_thresh=0.9):
    """
    Generate a label image from Mask-RCNN detections

    :param image_size: the size of the label image as `(height, width)`
    :param det_scores: Detection scores; (1, N) or (N,) array
    :param det_class_id: Detection class IDs; (1, N) or (N,) array
    :param mask_boxes: Detection boxes; (1, N, 4) or (N, 4) array, each box is [y1, x1, y2, x2]
    :param mrcnn_mask: Detection masks; (1, N, H, W) or (N, H, W) array, where H,W is the mask size
    :param mask_nms_thresh: Mask NMS threshold
    :return: (label_img, cls_img)
        label_img; (height, width) label image with different integer ID for each instance
        cls_img; (height, width) with pixels labelled by class
    """
    if det_scores.ndim == 2:
        # Remove batch dimension
        mask_boxes = mask_boxes[0]
        det_scores = det_scores[0]
        det_class_id = det_class_id[0]
        mrcnn_mask = mrcnn_mask[0]

    # mrcnn_mask: (D, H, W, C); D=detection, H,W=height,width, C=detection class

    y = np.zeros(image_size, dtype=int)
    y_cls = np.zeros(image_size, dtype=int)
    order = np.argsort(det_scores)[::-1]

    label_i = 1
    for i in order:
        box = mask_boxes[i]
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