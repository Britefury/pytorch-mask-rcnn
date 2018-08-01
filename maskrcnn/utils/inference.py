import math
import numpy as np
import skimage.transform


def mrcnn_detections_to_label_image(image_size, det_class_id, det_scores, mask_boxes, mrcnn_mask, mask_nms_thresh=0.9):
    """
    Generate a label image from Mask-RCNN detections

    :param image_size: the size of the label image as `(height, width)`
    :param det_class_id: Detection class IDs; (1, N) or (N,) array
    :param det_scores: Detection scores; (1, N) or (N,) array
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


def mrcnn_detections_to_image_space(image_size, det_boxes, det_class_id, det_scores, mask_boxes, mrcnn_mask):
    """
    Transform detections into image space.
    Box co-ordinates are rounded down (lower) and up (upper) to pixel co-ordinates.
    Masks are scaled to image space.
    Note that `det_boxes` is not used, but a potentially filtered version of it is returned

    :param image_size: the size of the label image as `(height, width)`
    :param det_boxes: Detection boxes; (1, N, 4) or (N, 4) array, each box is [y1, x1, y2, x2]
    :param det_class_id: Detection class IDs; (1, N) or (N,) array
    :param det_scores: Detection scores; (1, N) or (N,) array
    :param mask_boxes: Mask boxes; (1, N, 4) or (N, 4) array, each box is [y1, x1, y2, x2]
    :param mrcnn_mask: Detection masks; (1, N, H, W) or (N, H, W) array, where H,W is the mask size
    :return: (det_boxes_img, det_class_id, det_scores, mask_boxes_img, mrcnn_mask_img)
        det_boxes_img: image space detection boxes; (N, 4) array
        det_class_id_img: detection class IDs; (N,) array
        det_scores_img: scores; (N,) array
        mask_boxes_img: image space detection boxes; (N, 4) array
        mrcnn_mask_imgs: list of mask images, each of which is a (h, w) array with dtype=float, where h and w
            are likely differ for each mask
    """
    if det_scores.ndim == 2:
        # Remove batch dimension
        mask_boxes = mask_boxes[0]
        det_scores = det_scores[0]
        det_class_id = det_class_id[0]
        mrcnn_mask = mrcnn_mask[0]

    # mrcnn_mask: (D, H, W, C); D=detection, H,W=height,width, C=detection class

    det_boxes_img = []
    det_class_id_img = []
    det_scores_img = []
    mask_boxes_img = []
    mrcnn_mask_imgs = []

    label_i = 1
    for i in range(len(mask_boxes)):
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
            img_mask = img_mask[y1c - y1:img_mask.shape[0] - (y2 - y2c),
                                x1c - x1:img_mask.shape[1] - (x2 - x2c)]

            if img_mask.shape[0] > 0 and img_mask.shape[1] > 0:
                det_boxes_img.append(det_boxes[i])
                det_class_id_img.append(det_class_id[i])
                det_scores_img.append(det_scores[i])
                mask_boxes_img.append(np.array([y1c, x1c, y2c, x2c]))
                mrcnn_mask_imgs.append(img_mask)

    det_boxes_img = np.array(det_boxes_img)
    det_class_id_img = np.array(det_class_id_img, dtype=int)
    det_scores_img = np.array(det_scores_img)
    mask_boxes_img = np.array(mask_boxes_img, dtype=int)

    return det_boxes_img, det_class_id_img, det_scores_img, mask_boxes_img, mrcnn_mask_imgs

