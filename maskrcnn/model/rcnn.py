import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn.nms.nms_wrapper import nms
from maskrcnn.roialign.crop_and_resize.crop_and_resize import CropAndResizeAligned, CropAndResize
from maskrcnn.roialign.roi_align.roi_align import RoIAlign
from .utils import not_empty, is_empty, log2, intersect1d, unique1d, box_refinement, split_detections
from .utils import flatten_detections_with_sample_indices, flatten_detections
from .utils import unflatten_detections, concatenate_detections, torch_tensor_to_int_list
from .rpn import apply_box_deltas, compute_rpn_losses, compute_rpn_losses_per_sample
from .rpn import RPNBaseModel, alt_forward_method



class RCNNHead (nn.Module):
    def __init__(self, config, depth, pool_size, num_classes, roi_canonical_scale, roi_canonical_level,
                 min_pyramid_level, max_pyramid_level, roi_align_function, roi_align_sampling_ratio):
        super(RCNNHead, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.roi_canonical_scale = roi_canonical_scale
        self.roi_canonical_level = roi_canonical_level
        self.min_pyramid_level = min_pyramid_level
        self.max_pyramid_level = max_pyramid_level
        self.roi_align_function = roi_align_function
        self.roi_align_sampling_ratio = roi_align_sampling_ratio

        self.mlp = config.RCNN_MLP2

        if self.mlp:
            self.fc1 = nn.Linear(7 * 7 * self.depth, 1024)
            self.fc2 = nn.Linear(1024, 1024)
        else:
            self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
            self.bn1 = nn.BatchNorm2d(1024, eps=config.BN_EPS, momentum=0.01)
            self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
            self.bn2 = nn.BatchNorm2d(1024, eps=config.BN_EPS, momentum=0.01)

        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois, n_rois_per_sample, image_shape):
        x = pyramid_roi_align(x, rois, n_rois_per_sample,
                              self.pool_size, image_shape,
                              self.roi_canonical_scale, self.roi_canonical_level,
                              self.min_pyramid_level, self.max_pyramid_level,
                              self.roi_align_function, self.roi_align_sampling_ratio)
        if self.mlp:
            x = x.view(x.shape[0], -1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

        x = x.view(x.shape[0],-1)
        rcnn_class_logits_flat = self.linear_class(x)
        rcnn_probs_flat = self.softmax(rcnn_class_logits_flat)

        rcnn_bbox_flat = self.linear_bbox(x)
        rcnn_bbox_flat = rcnn_bbox_flat.view(rcnn_bbox_flat.size()[0], -1, 4)

        (rcnn_class_logits,) = unflatten_detections(n_rois_per_sample, rcnn_class_logits_flat)
        (rcnn_probs,) = unflatten_detections(n_rois_per_sample, rcnn_probs_flat)
        (rcnn_bbox,) = unflatten_detections(n_rois_per_sample, rcnn_bbox_flat)

        return [rcnn_class_logits, rcnn_probs, rcnn_bbox]

    def detectron_weight_mapping(self):
        def convert_bbox_pred_w(shape, src_blobs):
            val = src_blobs['bbox_pred_w']
            val2 = val.reshape((-1, 4) + val.shape[1:])
            val2 = val2[:, [1, 0, 3, 2], :]
            val = val2.reshape(val.shape)
            return val

        def convert_bbox_pred_b(shape, src_blobs):
            val = src_blobs['bbox_pred_b']
            val2 = val.reshape((-1, 4))
            val2 = val2[:, [1, 0, 3, 2]]
            val = val2.reshape(val.shape)
            return val

        det_map = {}
        orphans = []
        if self.mlp:
            det_map['fc1.weight'] = 'fc6_w'
            det_map['fc1.bias'] = 'fc6_b'
            det_map['fc2.weight'] = 'fc7_w'
            det_map['fc2.bias'] = 'fc7_b'
        else:
            raise RuntimeError('Detectron cannot import non-MLP RCNN layers')
        det_map['linear_class.weight'] = 'cls_score_w'
        det_map['linear_class.bias'] = 'cls_score_b'
        det_map['linear_bbox.weight'] = convert_bbox_pred_w
        det_map['linear_bbox.bias'] = convert_bbox_pred_b
        return det_map, orphans



############################################################
#  ROIAlign Layer
############################################################

def pyramid_roi_align(feature_maps, boxes, n_boxes_per_sample,
                      pool_size, image_shape, roi_canonical_scale,
                      roi_canonical_level, min_pyramid_level, max_pyramid_level,
                      roi_align_function, roi_align_sampling_ratio):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Inputs:
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - n_boxes_per_sample: [n] where n is the number of boxes in each sample; the remainder
            are assumed to be zero-padding
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels
    - roi_canonical_scale: canonical pyramid level scale
    - roi_canonical_level: canonical pyramid level

    Output:
    Pooled regions in the shape: [num_boxes_in_all_samples, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Assign each ROI to a level in the pyramid based on the ROI area.
    boxes_flat, box_sample_indices = flatten_detections_with_sample_indices(n_boxes_per_sample, boxes)
    y1, x1, y2, x2 = boxes_flat.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1
    area = h * w
    area_in_img_space = area * (image_shape[0] * image_shape[1])
    sz_in_img_space = torch.sqrt(area_in_img_space)
    sz_in_img_space = torch.clamp(sz_in_img_space, min=1e-6)[:, 0]

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a `roi_base_natural_size` x `roi_base_natural_size` ROI (in pixels) maps to the lowest level
    roi_level = torch.floor(roi_canonical_level + log2(sz_in_img_space/roi_canonical_scale))
    roi_level = roi_level.clamp(min_pyramid_level, max_pyramid_level)


    # Loop through levels and apply ROI pooling to each.
    pooled = []
    boxes_in_each_level = []
    for level_i, level in enumerate(range(min_pyramid_level, max_pyramid_level + 1)):
        ix  = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:,0]
        level_boxes = boxes_flat[ix, :]
        level_sample_indices = box_sample_indices[ix]

        # Keep track of which box is mapped to which level
        boxes_in_each_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        if roi_align_function == 'crop_and_resize':
            pooled_features = CropAndResize(pool_size, pool_size, 0)(
                feature_maps[level_i], level_boxes, level_sample_indices)
        elif roi_align_function == 'border_aware_crop_and_resize':
            pooled_features = CropAndResizeAligned(pool_size, pool_size, 0)(
                feature_maps[level_i], level_boxes, level_sample_indices)
        elif roi_align_function == 'roi_align':
            pooled_features = RoIAlign(pool_size, pool_size, roi_align_sampling_ratio)(
                feature_maps[level_i], level_boxes, level_sample_indices)
        else:
            raise ValueError('Unknown RoI Align function \'{}\''.format(roi_align_function))
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    boxes_in_each_level = torch.cat(boxes_in_each_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(boxes_in_each_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    return boxes

def refine_detections(rois, probs, deltas, window, config, image_size, min_confidence,
                      override_class=None):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.
        min_confidence: minimum confidence
        override_class: int or None; override class ID to always be this class

    Returns detections:
        boxes: [N, (y1, x1, y2, x2)]
        class_ids: [N]
        scores: [N]
    """
    device = rois.device

    # Class IDs per ROI
    if override_class:
        class_ids = (torch.ones(probs.size()[0], dtype=torch.long, device=device) * override_class)
    else:
        _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0], dtype=torch.long, device=device)
    class_scores = probs[idx, class_ids]
    deltas_specific = deltas[idx, class_ids]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    if config.RCNN_BBOX_USE_STD_DEV:
        std_dev = torch.tensor(np.reshape(config.BBOX_STD_DEV, [1, 4]), dtype=torch.float, device=device)
        deltas_specific = deltas_specific * std_dev
    refined_rois = apply_box_deltas(rois, deltas_specific)

    # Convert coordinates to image domain
    height, width = image_size
    scale = torch.tensor(np.array([height, width, height, width]), dtype=torch.float, device=device)
    refined_rois *= scale

    refined_sizes = refined_rois[:, 2:4] - refined_rois[:, 0:2]

    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    refined_sizes = refined_rois[:, 2:4] - refined_rois[:, 0:2]

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)
    refined_sizes = refined_rois[:, 2:4] - refined_rois[:, 0:2]

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    if override_class is not None:
        keep_bool = torch.ones(class_ids.size(), dtype=torch.uint8, device=device)
    else:
        keep_bool = class_ids>0

    # Filter out low confidence boxes
    if min_confidence is not None and min_confidence > 0.0:
        keep_bool = keep_bool & (class_scores >= min_confidence)
    keep = torch.nonzero(keep_bool)
    if keep.shape[0] == 0:
        # No detections
        return None, None, None
    else:
        keep = keep[:,0]

        # Apply per-class NMS
        pre_nms_class_ids = class_ids[keep]
        pre_nms_scores = class_scores[keep]
        pre_nms_rois = refined_rois[keep]

        for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
            # Pick detections of this class
            ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]

            # Sort
            ix_rois = pre_nms_rois[ixs]
            ix_scores = pre_nms_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order,:]

            class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).detach(), config.RCNN_DETECTION_NMS_THRESHOLD)

            # Map indicies
            class_keep = keep[ixs[order[class_keep]]]

            if i==0:
                nms_keep = class_keep
            else:
                nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
        keep = intersect1d(keep, nms_keep)

        # Keep top detections
        roi_count = config.RCNN_DETECTION_MAX_INSTANCES
        top_ids = class_scores[keep].sort(descending=True)[1][:roi_count]
        keep = keep[top_ids]

        # Coordinates are in image domain.
        return refined_rois[keep], class_ids[keep], class_scores[keep]


def detection_layer(config, rois, rcnn_class, rcnn_bbox, n_rois_per_sample, image_windows, image_size, min_confidence,
                    override_class=None):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    rois: [batch, n_rois, 4]
    rcnn_class: [batch, n_rois, n_classes]
    rcnn_bbox: [batch, n_rois, n_classes, 4]
    override_class: int or None; override class ID to always be this class

    Returns:
        (det_boxes, det_class_ids, det_scores, n_dets_per_sample) where
            det_boxes: [batch, num_detections, (y1, x1, y2, x2)] (pixel co-ordinates)
            det_class_ids: [batch, num_detections]
            det_scores: [batch, num_detections]
            n_dets_per_sample: [batch]
    """
    device = rcnn_bbox.device

    det_boxes = []
    det_class_ids = []
    det_scores = []
    n_detections_total = 0
    for sample_i, n_rois in enumerate(n_rois_per_sample):
        sample_det_boxes, sample_det_class_ids, sample_det_scores = refine_detections(
            rois[sample_i, :n_rois], rcnn_class[sample_i, :n_rois], rcnn_bbox[sample_i, :n_rois],
            image_windows[sample_i], config, image_size, min_confidence, override_class=override_class)
        if sample_det_boxes is None:
            sample_det_boxes = torch.zeros([0], dtype=torch.float, device=device)
            sample_det_class_ids = torch.zeros([0], dtype=torch.long, device=device)
            sample_det_scores = torch.zeros([0], dtype=torch.float, device=device)
        else:
            n_detections_total += sample_det_boxes.size()[0]
            sample_det_boxes = sample_det_boxes.unsqueeze(0)
            sample_det_class_ids = sample_det_class_ids.unsqueeze(0)
            sample_det_scores = sample_det_scores.unsqueeze(0)
        det_boxes.append(sample_det_boxes)
        det_class_ids.append(sample_det_class_ids)
        det_scores.append(sample_det_scores)

    if n_detections_total > 0:
        (det_boxes, det_class_ids, det_scores), n_dets_per_sample = concatenate_detections(
            det_boxes, det_class_ids, det_scores)
    else:
        det_boxes = torch.zeros([0], dtype=torch.float, device=device)
        det_class_ids = torch.zeros([0], dtype=torch.long, device=device)
        det_scores = torch.zeros([0], dtype=torch.float, device=device)
        n_dets_per_sample = [0] * len(n_rois_per_sample)

    return det_boxes, det_class_ids, det_scores, n_dets_per_sample


############################################################
#  Loss Functions
############################################################

def compute_rcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if not_empty(target_class_ids):
        loss = F.cross_entropy(pred_class_logits,target_class_ids.long())
    else:
        device = pred_class_logits.device
        loss = torch.tensor(0.0, dtype=torch.float, device=device)

    return loss


def compute_rcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    device = pred_bbox.device

    if not_empty(target_class_ids):
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0],:]
        pred_bbox = pred_bbox[indices[:,0],indices[:,1],:]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = torch.tensor([0], dtype=torch.float, device=device)

    return loss





def compute_faster_rcnn_losses(config, rpn_pred_class_logits, rpn_pred_bbox, rpn_target_match, rpn_target_bbox,
                               rpn_target_num_pos_per_sample, rcnn_pred_class_logits, rcnn_pred_bbox,
                               rcnn_target_class_ids, rcnn_target_deltas):
    rpn_target_num_pos_per_sample = torch_tensor_to_int_list(rpn_target_num_pos_per_sample)
    rpn_class_loss, rpn_bbox_loss = compute_rpn_losses(config, rpn_pred_class_logits, rpn_pred_bbox, rpn_target_match,
                                                       rpn_target_bbox, rpn_target_num_pos_per_sample)

    rcnn_class_loss = compute_rcnn_class_loss(rcnn_target_class_ids, rcnn_pred_class_logits)
    rcnn_bbox_loss = compute_rcnn_bbox_loss(rcnn_target_deltas, rcnn_target_class_ids, rcnn_pred_bbox)

    return [rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss]



############################################################
#  Detection Target Layer
############################################################

def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    device = boxes1.device

    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = torch.zeros(y1.size()[0], device=device)
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps



def rcnn_detection_target_one_sample(proposals, prop_class_logits, prop_class, prop_bbox_deltas, gt_class_ids, gt_boxes, config,
                                     hard_negative_mining=False):
    """Subsamples proposals and generates target box refinement and class_ids.

    Works on a single sample.

    If hard_negative_mining is True, values must be provided for prop_class_logits, prop_class and prop_bbox_deltas,
    otherwise they are optional.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates.
    prop_class_logits: (optional) [N, N_CLASSES] predicted class logits for each proposal.
    prop_class: (optional) [N, N_CLASSES] predicted class probabilities for each proposal.
    prop_bbox_deltas: (optional) [N, N_CLASSES, 4] predicted bbox deltas for each proposal.
    gt_class_ids: [N_GT] Integer class IDs.
    gt_boxes: [N_GT, (y1, x1, y2, x2)] in normalized
              coordinates.
    hard_negative_mining: bool; if True, use hard negative mining to choose target boxes

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [RCNN_TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    roi_class_logits: [RCNN_TRAIN_ROIS_PER_IMAGE, N_CLASSES] predicted class logits of selected proposals
    roi_class_probs: [RCNN_TRAIN_ROIS_PER_IMAGE, N_CLASSES] predicted class probabilities of selected proposals
    roi_bbox_deltas: [RCNN_TRAIN_ROIS_PER_IMAGE, N_CLASSES, 4] predicted bbox deltas of selected proposals.
    target_class_ids: [RCNN_TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [RCNN_TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    """
    device = proposals.device

    if hard_negative_mining:
        if prop_class_logits is None:
            raise ValueError('prop_class_logits cannot be None when hard_negative_mining is True')
        if prop_class is None:
            raise ValueError('prop_class cannot be None when hard_negative_mining is True')
        if prop_bbox_deltas is None:
            raise ValueError('prop_bbox_deltas cannot be None when hard_negative_mining is True')

    if prop_class_logits is not None and prop_class is not None and prop_bbox_deltas is not None:
        has_rcnn_predictions = True
    elif prop_class_logits is None and prop_class is None and prop_bbox_deltas is None:
        has_rcnn_predictions = False
    else:
        raise ValueError('prop_class_logits, prop_class and prop_bbox_deltas should either all have '
                         'values or all be None')



    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    if not_empty(torch.nonzero(gt_class_ids < 0)):
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
        crowd_boxes = gt_boxes[crowd_ix, :]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix, :]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = torch.tensor([True] * proposals.size()[0], dtype=torch.uint8, device=device)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    # Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    if hard_negative_mining:
        # Get the probability of the negative (empty) class for each proposal; needed for hard negative mining
        negative_cls_prob = prop_class[:,0]
    else:
        negative_cls_prob = None

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    if not_empty(torch.nonzero(positive_roi_bool)):
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.RCNN_TRAIN_ROIS_PER_IMAGE *
                             config.RCNN_ROI_POSITIVE_RATIO)

        if hard_negative_mining:
            # Hard negative mining
            # Choose samples with the highest negative class (class 0) probability (incorrect)
            if positive_count < positive_indices.size()[0]:
                _, hard_neg_idx = negative_cls_prob[positive_indices].topk(positive_count)
                positive_indices = positive_indices[hard_neg_idx]
        else:
            rand_idx = torch.randperm(positive_indices.size()[0])
            rand_idx = rand_idx[:positive_count]
            rand_idx = rand_idx.to(device)
            positive_indices = positive_indices[rand_idx]

        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices,:]
        if has_rcnn_predictions:
            positive_class_logits = prop_class_logits[positive_indices,:]
            positive_class_probs = prop_class[positive_indices,:]
            positive_bbox_deltas = prop_bbox_deltas[positive_indices,:,:]
        else:
            positive_class_logits = positive_class_probs = positive_bbox_deltas = None

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices,:]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment,:]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment]

        # Compute bbox refinement for positive ROIs
        deltas = box_refinement(positive_rois.detach(), roi_gt_boxes.detach())
        if config.RCNN_BBOX_USE_STD_DEV:
            std_dev = torch.tensor(config.BBOX_STD_DEV, dtype=torch.float, device=device)
            deltas /= std_dev
    else:
        positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if not_empty(torch.nonzero(negative_roi_bool)) and positive_count>0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.RCNN_ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)

        if hard_negative_mining:
            # Hard negative mining
            # Choose samples with the lowest negative class (class 0) probability (incorrect)
            if negative_count< negative_indices.size()[0]:
                _, hard_neg_idx = negative_cls_prob[negative_indices].topk(negative_count, largest=False)
                negative_indices = negative_indices[hard_neg_idx]
        else:
            rand_idx = torch.randperm(negative_indices.size()[0])
            rand_idx = rand_idx[:negative_count]
            rand_idx = rand_idx.to(device)
            negative_indices = negative_indices[rand_idx]

        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices, :]
        if has_rcnn_predictions:
            negative_class_logits = prop_class_logits[negative_indices,:]
            negative_class_probs = prop_class[negative_indices,:]
            negative_bbox_deltas = prop_bbox_deltas[negative_indices,:,:]
        else:
            negative_class_logits = negative_class_probs = negative_bbox_deltas = None
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        if has_rcnn_predictions:
            roi_class_logits = torch.cat((positive_class_logits, negative_class_logits), dim=0)
            roi_class_probs = torch.cat((positive_class_probs, negative_class_probs), dim=0)
            roi_bbox_deltas = torch.cat((positive_bbox_deltas, negative_bbox_deltas), dim=0)
        else:
            roi_class_logits = roi_class_probs = roi_bbox_deltas = None

        zeros = torch.zeros(negative_count, dtype=torch.long, device=device)
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = torch.zeros((negative_count,4), device=device)
        deltas = torch.cat([deltas, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
        roi_class_logits = positive_class_logits
        roi_class_probs = positive_class_probs
        roi_bbox_deltas = positive_bbox_deltas
    elif negative_count > 0:
        rois = negative_rois
        roi_class_logits = negative_class_logits
        roi_class_probs = negative_class_probs
        roi_bbox_deltas = negative_bbox_deltas
        roi_gt_class_ids = torch.zeros(negative_count, dtype=torch.long, device=device)
        deltas = torch.zeros((negative_count,4), device=device)
    else:
        rois = torch.zeros([0], dtype=torch.float, device=device)
        if has_rcnn_predictions:
            roi_class_logits = torch.zeros([0], dtype=torch.float, device=device)
            roi_class_probs = torch.zeros([0], dtype=torch.float, device=device)
            roi_bbox_deltas = torch.zeros([0], dtype=torch.float, device=device)
        else:
            roi_class_logits = roi_class_probs = roi_bbox_deltas = None

        roi_gt_class_ids = torch.zeros([0], dtype=torch.int, device=device)
        deltas = torch.zeros([0], dtype=torch.float, device=device)

    return rois, roi_class_logits, roi_class_probs, roi_bbox_deltas, roi_gt_class_ids, deltas


def rcnn_detection_target_layer(proposals, prop_class_logits, prop_class, prop_bbox_deltas, n_proposals_per_sample,
                                gt_class_ids, gt_boxes, n_gts_per_sample, config, hard_negative_mining):
    """Subsamples proposals and generates target box refinement and class_ids for each.

    Works on a mini-batch of samples.

    If hard_negative_mining is True, values must be provided for prop_class_logits, prop_class and prop_bbox_deltas,
    otherwise they are optional.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    prop_class_logits: [batch, N, N_CLASSES] predicted class logits for each proposal. Might
               be zero padded if there are not enough proposals.
    prop_class: [batch, N, N_CLASSES] predicted class probabilities for each proposal. Might
               be zero padded if there are not enough proposals.
    prop_bbox_deltas: [batch, N, N_CLASSES, 4] predicted bbox deltas for each proposal. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, N_GT] Integer class IDs.
    gt_boxes: [batch, N_GT, (y1, x1, y2, x2)] in normalized
              coordinates.
    hard_negative_mining: bool; if True, use hard negative mining to choose target boxes

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, RCNN_TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    roi_class_logits: [batch, RCNN_TRAIN_ROIS_PER_IMAGE, N_CLASSES] predicted class logits of selected proposals
    roi_class_probs: [batch, RCNN_TRAIN_ROIS_PER_IMAGE, N_CLASSES] predicted class probabilities of selected proposals
    roi_bbox_deltas: [batch, RCNN_TRAIN_ROIS_PER_IMAGE, N_CLASSES, 4] predicted bbox deltas of selected proposals.
    target_class_ids: [batch, RCNN_TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, RCNN_TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    """
    if hard_negative_mining:
        if prop_class_logits is None:
            raise ValueError('prop_class_logits cannot be None when hard_negative_mining is True')
        if prop_class is None:
            raise ValueError('prop_class cannot be None when hard_negative_mining is True')
        if prop_bbox_deltas is None:
            raise ValueError('prop_bbox_deltas cannot be None when hard_negative_mining is True')

    if prop_class_logits is not None and prop_class is not None and prop_bbox_deltas is not None:
        has_rcnn_predictions = True
    elif prop_class_logits is None and prop_class is None and prop_bbox_deltas is None:
        has_rcnn_predictions = False
    else:
        raise ValueError('prop_class_logits, prop_class and prop_bbox_deltas should either all have '
                         'values or all be None')

    rois = []
    if has_rcnn_predictions:
        roi_class_logits = []
        roi_class_probs = []
        roi_bbox_deltas = []
    else:
        roi_class_logits = roi_class_probs = roi_bbox_deltas = None
    target_class_ids = []
    target_deltas = []
    for sample_i, (n_props, n_gts) in enumerate(zip(n_proposals_per_sample, n_gts_per_sample)):
        sample_roi_class_logits = sample_roi_class_probs = sample_roi_bbox_deltas = None
        if n_props > 0 and n_gts > 0:
            if has_rcnn_predictions:
                sample_prop_class_logits = prop_class_logits[sample_i, :n_props]
                sample_prop_class = prop_class[sample_i, :n_props]
                sample_prop_bbox_deltas = prop_bbox_deltas[sample_i, :n_props]
            else:
                sample_prop_class_logits = sample_prop_class = sample_prop_bbox_deltas = None
            sample_rois, sample_roi_class_logits, sample_roi_class_probs, sample_roi_bbox_deltas, \
                    sample_roi_gt_class_ids, sample_deltas = rcnn_detection_target_one_sample(
                proposals[sample_i, :n_props], sample_prop_class_logits,
                sample_prop_class, sample_prop_bbox_deltas,
                gt_class_ids[sample_i, :n_gts], gt_boxes[sample_i, :n_gts], config, hard_negative_mining)
            if not_empty(sample_rois):
                sample_rois = sample_rois.unsqueeze(0)
                if has_rcnn_predictions:
                    sample_roi_class_logits = sample_roi_class_logits.unsqueeze(0)
                    sample_roi_class_probs = sample_roi_class_probs.unsqueeze(0)
                    sample_roi_bbox_deltas = sample_roi_bbox_deltas.unsqueeze(0)
                sample_roi_gt_class_ids = sample_roi_gt_class_ids.unsqueeze(0)
                sample_deltas = sample_deltas.unsqueeze(0)
        else:
            sample_rois = proposals.new()
            if has_rcnn_predictions:
                sample_roi_class_logits = proposals.new()
                sample_roi_class_probs = proposals.new()
                sample_roi_bbox_deltas = proposals.new()
            sample_roi_gt_class_ids = gt_class_ids.new()
            sample_deltas = proposals.new()
        rois.append(sample_rois)
        if has_rcnn_predictions:
            roi_class_logits.append(sample_roi_class_logits)
            roi_class_probs.append(sample_roi_class_probs)
            roi_bbox_deltas.append(sample_roi_bbox_deltas)
        target_class_ids.append(sample_roi_gt_class_ids)
        target_deltas.append(sample_deltas)

    if has_rcnn_predictions:
        (rois, roi_class_logits, roi_class_probs, roi_bbox_deltas, roi_gt_class_ids, deltas), n_dets_per_sample = concatenate_detections(
            rois, roi_class_logits, roi_class_probs, roi_bbox_deltas, target_class_ids, target_deltas)
    else:
        (rois, roi_gt_class_ids, deltas), n_dets_per_sample = concatenate_detections(
            rois, target_class_ids, target_deltas)

    return rois, roi_class_logits, roi_class_probs, roi_bbox_deltas, roi_gt_class_ids, deltas, n_dets_per_sample





############################################################
#  Faster R-CNN Model
############################################################


class FasterRCNNBaseModel (RPNBaseModel):
    """
    Faster R-CNN Base model

    Network:
    - RCNN head
    - inherits from RPNBaseModel:
        - feature pyramid network for feature extraction
        - RPN head for proposal generation

    To build a Faster R-CNN based model, inherit from AbstractFasterRCNNModel as it provides methods for training and detection.
    """


    def __init__(self, config):
        """
        config: A Sub-class of the Config class
        """
        super(FasterRCNNBaseModel, self).__init__(config)

        # FPN Classifier
        self.classifier = RCNNHead(config, 256, config.RCNN_POOL_SIZE, config.NUM_CLASSES,
                                   config.ROI_CANONICAL_SCALE, config.ROI_CANONICAL_LEVEL,
                                   config.ROI_MIN_PYRAMID_LEVEL, config.ROI_MAX_PYRAMID_LEVEL,
                                   config.ROI_ALIGN_FUNCTION, config.ROI_ALIGN_SAMPLING_RATIO)


    def rcnn_detect_forward(self, images, image_windows, rcnn_feature_maps, rpn_rois, n_rois_per_sample,
                            override_class=None):
        """
        Runs the RCNN
        :param images:
        :param image_windows:
        :param rcnn_feature_maps:
        :param rpn_rois:
        :param n_rois_per_sample:
        :param override_class:
        :return:
        """
        image_size = images.shape[2:]

        # Network Heads
        # Proposal classifier and BBox regressor heads
        if rpn_rois.size()[1] > self.config.DETECTION_BLOCK_SIZE_INFERENCE:
            rcnn_class = []
            rcnn_bbox = []
            for block_i in range(0, rpn_rois.size()[1], self.config.DETECTION_BLOCK_SIZE_INFERENCE):
                block_j = min(block_i + self.config.DETECTION_BLOCK_SIZE_INFERENCE, rpn_rois.size()[1])

                # Get the number of ROIs per sample for this block
                n_rois_per_sample_block = [
                    min(block_j,n_rois) - min(block_i,n_rois) for n_rois in n_rois_per_sample
                ]

                _, rcnn_class_block, rcnn_bbox_block = self.classifier(
                    rcnn_feature_maps, rpn_rois[:, block_i:block_j, ...], n_rois_per_sample_block, image_size)

                rcnn_class.append(rcnn_class_block)
                rcnn_bbox.append(rcnn_bbox_block)

            rcnn_class = torch.cat(rcnn_class, dim=0)
            rcnn_bbox = torch.cat(rcnn_bbox, dim=0)

        else:
            _, rcnn_class, rcnn_bbox = self.classifier(
                rcnn_feature_maps, rpn_rois, n_rois_per_sample, image_size)

        # Detections
        # det_boxes: [batch, num_detections, (y1, x1, y2, x2)] in image coordinates
        # det_class_ids: [batch, num_detections]
        # det_scores: [batch, num_detections]
        det_boxes, det_class_ids, det_scores, n_dets_per_sample = detection_layer(
            self.config, rpn_rois, rcnn_class, rcnn_bbox, n_rois_per_sample, image_windows, image_size,
            min_confidence=self.config.RCNN_DETECTION_MIN_CONFIDENCE, override_class=override_class)

        return det_boxes, det_class_ids, det_scores, n_dets_per_sample




class AbstractFasterRCNNModel (FasterRCNNBaseModel):
    """
    Abstract Faster R-CNN model

    Adds training and detection forward passes to FasterRCNNBaseModel
    """
    def _train_forward(self, molded_images, gt_class_ids, gt_boxes, n_gts_per_sample, hard_negative_mining=False):
        device = molded_images.device

        # Get image size
        image_size = molded_images.size()[2:]

        # Compute scale factor for converting normalized co-ordinates to pixel co-ordinates
        h, w = image_size
        scale = torch.tensor(np.array([h, w, h, w]), dtype=torch.float, device=device)

        # Get RPN proposals
        pre_nms_limit =  self.config.RPN_PRE_NMS_LIMIT_TRAIN
        nms_threshold =  self.config.RPN_NMS_THRESHOLD
        proposal_count = self.config.RPN_POST_NMS_ROIS_TRAINING
        rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, rpn_bbox, rpn_rois, _, n_rois_per_sample = \
            self._feature_maps_proposals_and_roi(molded_images, pre_nms_limit, nms_threshold, proposal_count)

        # Normalize coordinates
        gt_boxes_nrm = gt_boxes / scale

        if hard_negative_mining:
            # Apply RCNN head so that we can do hard negative mining in the detection target layer
            # Network Heads
            # Proposal classifier and BBox regressor heads
            roi_class_logits, roi_class, roi_bbox = self.classifier(
                rcnn_feature_maps, rpn_rois, n_rois_per_sample, image_size)


            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, rcnn_class_logits, rcnn_class, rcnn_bbox, target_class_ids, target_deltas, n_targets_per_sample = \
                rcnn_detection_target_layer(
                    rpn_rois, roi_class_logits, roi_class, roi_bbox, n_rois_per_sample,
                    gt_class_ids, gt_boxes_nrm, n_gts_per_sample, self.config, hard_negative_mining)
        else:
            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, _, _, _, target_class_ids, target_deltas, n_targets_per_sample = \
                rcnn_detection_target_layer(
                    rpn_rois, None, None, None, n_rois_per_sample,
                    gt_class_ids, gt_boxes_nrm, n_gts_per_sample, self.config, hard_negative_mining)

            if max(n_targets_per_sample) > 0:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                rcnn_class_logits, rcnn_class, rcnn_bbox = self.classifier(
                    rcnn_feature_maps, rois, n_targets_per_sample, image_size)
            else:
                rcnn_class_logits = torch.zeros([0], dtype=torch.float, device=device)
                rcnn_class = torch.zeros([0], dtype=torch.int, device=device)
                rcnn_bbox = torch.zeros([0], dtype=torch.float, device=device)

        return (rpn_class_logits, rpn_bbox, target_class_ids, rcnn_class_logits,
                target_deltas, rcnn_bbox, n_targets_per_sample)


    @alt_forward_method
    def train_forward(self, molded_images, gt_class_ids, gt_boxes, n_gts_per_sample, hard_negative_mining=False):
        (rpn_class_logits, rpn_bbox, target_class_ids, rcnn_class_logits,
         target_deltas, rcnn_bbox, n_targets_per_sample) = self._train_forward(
            molded_images, gt_class_ids, gt_boxes, n_gts_per_sample, hard_negative_mining=hard_negative_mining)

        target_class_ids, rcnn_class_logits, target_deltas, rcnn_bbox = \
            flatten_detections(n_targets_per_sample, target_class_ids, rcnn_class_logits, target_deltas, rcnn_bbox)

        return (rpn_class_logits, rpn_bbox, target_class_ids, rcnn_class_logits,
                target_deltas, rcnn_bbox, n_targets_per_sample)


    @alt_forward_method
    def train_loss_forward(self, molded_images, rpn_target_match, rpn_target_bbox, rpn_num_pos,
                           gt_class_ids, gt_boxes, n_gts_per_sample, hard_negative_mining=False):
        rpn_class_logits, rpn_pred_bbox, target_class_ids, rcnn_class_logits, target_deltas, rcnn_bbox, \
            n_targets_per_sample = self._train_forward(molded_images, gt_class_ids, gt_boxes, n_gts_per_sample,
                                   hard_negative_mining=hard_negative_mining)

        rpn_class_losses, rpn_bbox_losses = compute_rpn_losses_per_sample(
            self.config, rpn_class_logits, rpn_pred_bbox, rpn_target_match, rpn_target_bbox, rpn_num_pos)

        rcnn_class_losses = []
        rcnn_bbox_losses = []
        for sample_i, n_targets in enumerate(n_targets_per_sample):
            if n_targets > 0:
                rcnn_class_loss = compute_rcnn_class_loss(
                    target_class_ids[sample_i, :n_targets], rcnn_class_logits[sample_i, :n_targets, :])
                rcnn_bbox_loss = compute_rcnn_bbox_loss(
                    target_deltas[sample_i, :n_targets, :], target_class_ids[sample_i, :n_targets],
                    rcnn_bbox[sample_i, :n_targets, :])
                rcnn_class_losses.append(rcnn_class_loss[None])
                rcnn_bbox_losses.append(rcnn_bbox_loss[None])
            else:
                rcnn_class_losses.append(torch.tensor([0.0], dtype=torch.float, device=molded_images.device))
                rcnn_bbox_losses.append(torch.tensor([0.0], dtype=torch.float, device=molded_images.device))
        rcnn_class_losses = torch.cat(rcnn_class_losses, dim=0)
        rcnn_bbox_losses = torch.cat(rcnn_bbox_losses, dim=0)

        return (rpn_class_losses, rpn_bbox_losses, rcnn_class_losses, rcnn_bbox_losses)


    @alt_forward_method
    def detect_forward(self, images, image_windows, override_class=None):
        """Runs the detection pipeline.

        images: Tensor of images
        image_windows: tensor of image windows where each row is [y1, x1, y2, x2]
        override_class: int or None; override class ID to always be this class

        Returns: [detection0, detection1, ... detectionN]
        List of detections, one per sample, where each detection is a tuple of:
        (det_boxes, det_class_ids, det_scores, mrcnn_mask) where:
            det_boxes: [1, detections, [y1, x1, y2, x2]]
            det_class_ids: [1, detections]
            det_scores: [1, detections]
            mrcnn_mask: [1, detections, height, width, obj_class]
        """
        image_size = images.shape[2:]

        # rpn_feature_maps: [batch, channels, height, width]
        # mrcnn_feature_maps: [batch, channels, height, width]
        # rpn_bbox: [batch, anchors, 4]
        # rpn_rois: [batch, n_rois_after_nms, 4]
        # roi_scores: [batch, n_rois_after_nms]
        # n_rois_per_sample: [batch]
        rpn_feature_maps, rcnn_feature_maps, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample = self.rpn_detect_forward(
            images)

        # det_boxes: [batch, num_detections, (y1, x1, y2, x2)] in image coordinates
        # det_class_ids: [batch, num_detections]
        # det_scores: [batch, num_detections]
        det_boxes, det_class_ids, det_scores, n_dets_per_sample = self.rcnn_detect_forward(
            images, image_windows, rcnn_feature_maps, rpn_rois, n_rois_per_sample, override_class=override_class)

        if is_empty(det_boxes) or is_empty(det_class_ids) or is_empty(det_scores):
            # No detections
            n_images = images.shape[0]
            return [(np.zeros((n_images, 0, 4), dtype=np.float32),
                     np.zeros((n_images, 0), dtype=int),
                     np.zeros((n_images, 0), dtype=np.float32))
                    for i in range(n_images)]

        # Convert boxes to normalized coordinates
        # TODO: let DetectionLayer return normalized coordinates to avoid
        #       unnecessary conversions

        #
        # Detections done
        #

        # Convert to numpy
        det_boxes_np = det_boxes.cpu().numpy()
        det_class_ids_np = det_class_ids.cpu().numpy()
        det_scores_np = det_scores.cpu().numpy()

        return split_detections(n_dets_per_sample, det_boxes_np, det_class_ids_np, det_scores_np)
