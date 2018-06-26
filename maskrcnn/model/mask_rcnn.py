import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn.roialign.crop_and_resize.crop_and_resize import CropAndResizeAligned
from .utils import not_empty, is_empty, box_refinement, SamePad2d, concatenate_detections, flatten_detections,\
    unflatten_detections, split_detections, torch_tensor_to_int_list
from .rpn import RPNHead, compute_rpn_losses, compute_rpn_losses_per_sample, alt_forward_method
from .rcnn import RCNNHead, FasterRCNNBaseModel, refine_detections_batch, pyramid_roi_align, compute_rcnn_bbox_loss,\
    compute_rcnn_class_loss, bbox_overlaps



############################################################
#  Feature Pyramid Network Heads
############################################################

class MaskHead (nn.Module):
    def __init__(self, config, depth, pool_size, num_classes, roi_canonical_scale, roi_canonical_level,
                 min_pyramid_level, max_pyramid_level, roi_align_function, roi_align_sampling_ratio):
        super(MaskHead, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.roi_canonical_scale = roi_canonical_scale
        self.roi_canonical_level = roi_canonical_level
        self.min_pyramid_level = min_pyramid_level
        self.max_pyramid_level = max_pyramid_level
        self.roi_align_function = roi_align_function
        self.roi_align_sampling_ratio = roi_align_sampling_ratio

        if config.TORCH_PADDING:
            self.padding = None
            pad = 1 * config.MASK_CONV_DILATION
            dilation = config.MASK_CONV_DILATION
        else:
            self.padding = SamePad2d(kernel_size=3, stride=1)
            pad = 0
            dilation = 1

        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=pad, dilation=dilation)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=pad, dilation=dilation)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=pad, dilation=dilation)

        if config.MASK_BATCH_NORM:
            self.bn1 = nn.BatchNorm2d(256, eps=config.BN_EPS)
            self.bn2 = nn.BatchNorm2d(256, eps=config.BN_EPS)
            self.bn3 = nn.BatchNorm2d(256, eps=config.BN_EPS)
            self.bn4 = nn.BatchNorm2d(256, eps=config.BN_EPS)
        else:
            self.bn1 = self.bn2 = self.bn3 = self.bn4 = None

        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois, n_rois_per_sample, image_shape):
        x = pyramid_roi_align(x, rois, n_rois_per_sample,
                              self.pool_size, image_shape,
                              self.roi_canonical_scale, self.roi_canonical_level,
                              self.min_pyramid_level, self.max_pyramid_level,
                              self.roi_align_function, self.roi_align_sampling_ratio)

        if self.padding is not None:
            x = self.padding(x)
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)

        if self.padding is not None:
            x = self.padding(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.relu(x)

        if self.padding is not None:
            x = self.padding(x)
        x = self.conv3(x)
        if self.bn3 is not None:
            x = self.bn3(x)
        x = self.relu(x)

        if self.padding is not None:
            x = self.padding(x)
        x = self.conv4(x)
        if self.bn4 is not None:
            x = self.bn4(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        (x,) = unflatten_detections(n_rois_per_sample, x)

        return x

    def detectron_weight_mapping(self):
        det_map = {}
        orphans = []
        det_map['conv1.weight'] = '_[mask]_fcn1_w'
        det_map['conv1.bias'] = '_[mask]_fcn1_b'
        det_map['conv2.weight'] = '_[mask]_fcn2_w'
        det_map['conv2.bias'] = '_[mask]_fcn2_b'
        det_map['conv3.weight'] = '_[mask]_fcn3_w'
        det_map['conv3.bias'] = '_[mask]_fcn3_b'
        det_map['conv4.weight'] = '_[mask]_fcn4_w'
        det_map['conv4.bias'] = '_[mask]_fcn4_b'

        det_map['deconv.weight'] = 'conv5_mask_w'
        det_map['deconv.bias'] = 'conv5_mask_b'

        det_map['conv5.weight'] = 'mask_fcn_logits_w'
        det_map['conv5.bias'] = 'mask_fcn_logits_b'
        return det_map, orphans


############################################################
#  Detection Target Layer
############################################################

def maskrcnn_detection_target_one_sample(proposals, prop_class_logits, prop_class, prop_bbox_deltas,
                                         gt_class_ids, gt_boxes, gt_masks, config,
                                         hard_negative_mining=False):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Works on a single sample.

    If hard_negative_mining is True, values must be provided for prop_class_logits, prop_class and prop_bbox_deltas,
    otherwise they are optional.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates.
    prop_class_logits: [N, N_CLASSES] predicted class logits for each proposal.
    prop_class: [N, N_CLASSES] predicted class probabilities for each proposal.
    prop_bbox_deltas: [N, N_CLASSES, 4] predicted bbox deltas for each proposal.
    gt_class_ids: [N_GT] Integer class IDs.
    gt_boxes: [N_GT, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [height, width, N_GT] of boolean type
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
    target_mask: [RCNN_TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
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
        crowd_boxes = gt_boxes[crowd_ix.data, :]
        crowd_masks = gt_masks[crowd_ix.data, :, :]
        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[non_crowd_ix.data, :]
        gt_masks = gt_masks[non_crowd_ix.data, :]

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
                _, hard_neg_idx = negative_cls_prob[positive_indices.data].topk(positive_count)
                positive_indices = positive_indices[hard_neg_idx]
        else:
            rand_idx = torch.randperm(positive_indices.size()[0])
            rand_idx = rand_idx[:positive_count]
            rand_idx = rand_idx.to(device)
            positive_indices = positive_indices[rand_idx]

        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data,:]
        if has_rcnn_predictions:
            positive_class_logits = prop_class_logits[positive_indices.data,:]
            positive_class_probs = prop_class[positive_indices.data,:]
            positive_bbox_deltas = prop_bbox_deltas[positive_indices.data,:,:]
        else:
            positive_class_logits = positive_class_probs = positive_bbox_deltas = None

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data,:]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data,:]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        # Compute bbox refinement for positive ROIs
        deltas = box_refinement(positive_rois.data, roi_gt_boxes.data)
        if config.RCNN_BBOX_USE_STD_DEV:
            std_dev = torch.tensor(config.BBOX_STD_DEV, dtype=torch.float, device=device)
            deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data,:,:]

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = torch.arange(roi_masks.size()[0], dtype=torch.int, device=device)
        masks = CropAndResizeAligned(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(
            roi_masks.detach().unsqueeze(1), boxes.detach(), box_ids.detach())
        masks = masks.squeeze(1)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)
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
                _, hard_neg_idx = negative_cls_prob[negative_indices.data].topk(negative_count, largest=False)
                negative_indices = negative_indices[hard_neg_idx]
        else:
            rand_idx = torch.randperm(negative_indices.size()[0])
            rand_idx = rand_idx[:negative_count]
            rand_idx = rand_idx.to(device)
            negative_indices = negative_indices[rand_idx]

        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
        if has_rcnn_predictions:
            negative_class_logits = prop_class_logits[negative_indices.data,:]
            negative_class_probs = prop_class[negative_indices.data,:]
            negative_bbox_deltas = prop_bbox_deltas[negative_indices.data,:,:]
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

        zeros = torch.zeros((negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), device=device)
        masks = torch.cat([masks, zeros], dim=0)
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

        roi_gt_class_ids = torch.zeros(negative_count, device=device, dtype=torch.long)
        deltas = torch.zeros((negative_count, 4), device=device)
        masks = torch.zeros((negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), device=device)
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
        masks = torch.zeros([0], dtype=torch.float, device=device)

    return rois, roi_class_logits, roi_class_probs, roi_bbox_deltas, roi_gt_class_ids, deltas, masks


def maskrcnn_detection_target_layer(proposals, prop_class_logits, prop_class, prop_bbox_deltas, n_proposals_per_sample,
                                    gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample, config,
                                    hard_negative_mining):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

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
    gt_masks: [batch, height, width, N_GT] of boolean type
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
    target_mask: [batch, RCNN_TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
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
    target_mask = []
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
                    sample_roi_gt_class_ids, sample_deltas, sample_masks = maskrcnn_detection_target_one_sample(
                proposals[sample_i, :n_props], sample_prop_class_logits,
                sample_prop_class, sample_prop_bbox_deltas,
                gt_class_ids[sample_i, :n_gts], gt_boxes[sample_i, :n_gts],
                gt_masks[sample_i, :n_gts], config
            )
            if not_empty(sample_rois):
                sample_rois = sample_rois.unsqueeze(0)
                if has_rcnn_predictions:
                    sample_roi_class_logits = sample_roi_class_logits.unsqueeze(0)
                    sample_roi_class_probs = sample_roi_class_probs.unsqueeze(0)
                    sample_roi_bbox_deltas = sample_roi_bbox_deltas.unsqueeze(0)
                sample_roi_gt_class_ids = sample_roi_gt_class_ids.unsqueeze(0)
                sample_deltas = sample_deltas.unsqueeze(0)
                sample_masks = sample_masks.unsqueeze(0)
        else:
            sample_rois = proposals.data.new()
            if has_rcnn_predictions:
                sample_roi_class_logits = proposals.data.new()
                sample_roi_class_probs = proposals.data.new()
                sample_roi_bbox_deltas = proposals.data.new()
            sample_roi_gt_class_ids = gt_class_ids.data.new()
            sample_deltas = proposals.data.new()
            sample_masks = gt_masks.data.new()
        rois.append(sample_rois)
        if has_rcnn_predictions:
            roi_class_logits.append(sample_roi_class_logits)
            roi_class_probs.append(sample_roi_class_probs)
            roi_bbox_deltas.append(sample_roi_bbox_deltas)
        target_class_ids.append(sample_roi_gt_class_ids)
        target_deltas.append(sample_deltas)
        target_mask.append(sample_masks)


    if has_rcnn_predictions:
        (rois, roi_class_logits, roi_class_probs, roi_bbox_deltas, roi_gt_class_ids, deltas, masks), n_dets_per_sample = concatenate_detections(
            rois, roi_class_logits, roi_class_probs, roi_bbox_deltas, target_class_ids, target_deltas, target_mask)
    else:
        (rois, roi_gt_class_ids, deltas, masks), n_dets_per_sample = concatenate_detections(
            rois, target_class_ids, target_deltas, target_mask)

    return rois, roi_class_logits, roi_class_probs, roi_bbox_deltas, roi_gt_class_ids, deltas, masks, n_dets_per_sample




############################################################
#  Loss Functions
############################################################

def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    device = pred_masks.device

    if not_empty(target_class_ids):
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:,0].data,:,:]
        y_pred = pred_masks[indices[:,0].data,indices[:,1].data,:,:]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.tensor([0], dtype=torch.float, device=device)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_maskrcnn_losses(config, rpn_pred_class_logits, rpn_pred_bbox, rpn_target_match, rpn_target_bbox,
                            rpn_target_num_pos_per_sample, rcnn_pred_class_logits, rcnn_pred_bbox,
                            rcnn_target_class_ids, rcnn_target_deltas, mrcnn_pred_mask, mrcnn_target_mask):
    rpn_target_num_pos_per_sample = torch_tensor_to_int_list(rpn_target_num_pos_per_sample)

    rpn_class_loss, rpn_bbox_loss = compute_rpn_losses(config, rpn_pred_class_logits, rpn_pred_bbox, rpn_target_match, rpn_target_bbox,
                                                       rpn_target_num_pos_per_sample)

    mrcnn_class_loss = compute_rcnn_class_loss(rcnn_target_class_ids, rcnn_pred_class_logits)
    mrcnn_bbox_loss = compute_rcnn_bbox_loss(rcnn_target_deltas, rcnn_target_class_ids, rcnn_pred_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(mrcnn_target_mask, rcnn_target_class_ids, mrcnn_pred_mask)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]


############################################################
#  Mask R-CNN Model
############################################################

class AbstractMaskRCNNModel (FasterRCNNBaseModel):
    """
    Mask R-CNN abstract model

    Network:
    - Mask head
    - inherits from FasterRCNNBaseModel:
        - feature pyramid network for feature extraction
        - RPN head for proposal generation
        - RCNN head
    """

    def __init__(self, config):
        """
        config: A Sub-class of the Config class
        """
        super(AbstractMaskRCNNModel, self).__init__(config)

        # FPN Mask
        self.mask = MaskHead(config, 256, config.MASK_POOL_SIZE, config.NUM_CLASSES,
                             config.ROI_CANONICAL_SCALE, config.ROI_CANONICAL_LEVEL,
                             config.ROI_MIN_PYRAMID_LEVEL, config.ROI_MAX_PYRAMID_LEVEL,
                             config.ROI_ALIGN_FUNCTION, config.ROI_ALIGN_SAMPLING_RATIO)


    def _train_forward(self, molded_images, gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample, hard_negative_mining=False):
        """Supervised forward training pass helper

        molded_images: Tensor of images
        gt_class_ids: ground truth detection classes [batch, detection]
        gt_boxes: ground truth detection boxes [batch, detection, [y1, x1, y2, x2]
        n_gts_per_sample: number of ground truth detections per sample [batch]
        hard_negative_mining: if True, use hard negative mining to choose samples for training R-CNN head

        Returns:
            (rpn_class_logits, rpn_bbox, target_class_ids, rcnn_class_logits,
                    target_deltas, rcnn_bbox, target_mask, mrcnn_mask, n_targets_per_sample) where
                rpn_class_logits: [batch, anchor]; predicted class logits from RPN
                rpn_bbox: [batch, anchor, 4]; predicted bounding box deltas
                target_class_ids: [batch, ROI]; RCNN target class IDs
                rcnn_class_logits: [batch, ROI, cls]; RCNN predicted class logits
                target_deltas: [batch, ROI, 4]; RCNN target box deltas
                rcnn_bbox: [batch, ROI, cls, 4]; RCNN predicted box deltas
                target_mask: [batch, ROI, mask_height, mask_width]; target masks
                mrcnn_mask: [batch, ROI, mask_height, mask_width, cls]; predicted masks
                n_targets_per_sample: [batch] the number of target ROIs in each sample
        """
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
        rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits, rpn_bbox, rpn_rois, _, n_rois_per_sample = \
            self._feature_maps_rpn_preds_and_roi(molded_images, pre_nms_limit, nms_threshold, proposal_count)

        # Normalize coordinates
        gt_boxes_nrm = gt_boxes / scale

        if hard_negative_mining:
            # Apply RCNN head so that we can do hard negative mining in the detection target layer
            # Network Heads
            # Proposal classifier and BBox regressor heads
            roi_class_logits, roi_class, roi_bbox = self.classifier(
                mrcnn_feature_maps, rpn_rois, n_rois_per_sample, image_size)


            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, target_class_ids, target_deltas, target_mask, n_targets_per_sample = \
                maskrcnn_detection_target_layer(
                    rpn_rois, roi_class_logits, roi_class, roi_bbox, n_rois_per_sample,
                    gt_class_ids, gt_boxes_nrm, gt_masks, n_gts_per_sample, self.config, hard_negative_mining)

            if is_empty(rois):
                mrcnn_mask = torch.zeros([0], dtype=torch.float, device=device)
            else:
                # Create masks for detections
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois, n_targets_per_sample, image_size)

        else:
            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, _, _, _, target_class_ids, target_deltas, target_mask, n_targets_per_sample = \
                maskrcnn_detection_target_layer(
                    rpn_rois, None, None, None, n_rois_per_sample,
                    gt_class_ids, gt_boxes_nrm, gt_masks, n_gts_per_sample, self.config,
                    hard_negative_mining)


            if max(n_targets_per_sample) == 0:
                mrcnn_class_logits = torch.zeros([0], dtype=torch.float, device=device)
                mrcnn_class = torch.zeros([0], dtype=torch.int, device=device)
                mrcnn_bbox = torch.zeros([0], dtype=torch.float, device=device)
                mrcnn_mask = torch.zeros([0], dtype=torch.float, device=device)
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(
                    mrcnn_feature_maps, rois, n_targets_per_sample, image_size)

                # Create masks for detections
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois, n_targets_per_sample, image_size)

        return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits,
                target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, n_targets_per_sample]


    @alt_forward_method
    def train_forward(self, molded_images, gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample, hard_negative_mining=False):
        """Supervised forward training pass helper

        molded_images: Tensor of images
        gt_class_ids: ground truth detection classes [batch, detection]
        gt_boxes: ground truth detection boxes [batch, detection, [y1, x1, y2, x2]
        n_gts_per_sample: number of ground truth detections per sample [batch]
        hard_negative_mining: if True, use hard negative mining to choose samples for training R-CNN head

        Returns:
            (rpn_class_logits, rpn_bbox, target_class_ids, rcnn_class_logits,
                    target_deltas, rcnn_bbox, target_mask, mrcnn_mask, n_targets_per_sample) where
                rpn_class_logits: [batch, anchor]; predicted class logits from RPN
                rpn_bbox: [batch, anchor, 4]; predicted bounding box deltas
                target_class_ids: [batch, ROI]; RCNN target class IDs
                rcnn_class_logits: [batch, ROI, cls]; RCNN predicted class logits
                target_deltas: [batch, ROI, 4]; RCNN target box deltas
                rcnn_bbox: [batch, ROI, cls, 4]; RCNN predicted box deltas
                target_mask: [batch, ROI, mask_height, mask_width]; target masks
                mrcnn_mask: [batch, ROI, mask_height, mask_width, cls]; predicted masks
                n_targets_per_sample: [batch] the number of target ROIs in each sample
        """
        (rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits,
            target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, n_targets_per_sample) = self._train_forward(
                    molded_images, gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample,
                    hard_negative_mining=hard_negative_mining)

        target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
            flatten_detections(n_targets_per_sample, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)

        return (rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits,
                target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, n_targets_per_sample)


    @alt_forward_method
    def train_loss_forward(self, molded_images, rpn_target_match, rpn_target_bbox, rpn_num_pos,
                           gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample, hard_negative_mining=False):
        (rpn_class_logits, rpn_pred_bbox, target_class_ids, rcnn_class_logits,
            target_deltas, rcnn_bbox, target_mask, mrcnn_mask, n_targets_per_sample) = self._train_forward(
                    molded_images, gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample,
                    hard_negative_mining=hard_negative_mining)

        rpn_class_losses, rpn_bbox_losses = compute_rpn_losses_per_sample(
            self.config, rpn_class_logits, rpn_pred_bbox, rpn_target_match, rpn_target_bbox, rpn_num_pos)

        rcnn_class_losses = []
        rcnn_bbox_losses = []
        mrcnn_mask_losses = []
        for sample_i, n_targets in enumerate(n_targets_per_sample):
            if n_targets > 0:
                rcnn_class_loss = compute_rcnn_class_loss(
                    target_class_ids[sample_i, :n_targets], rcnn_class_logits[sample_i, :n_targets])
                rcnn_bbox_loss = compute_rcnn_bbox_loss(
                    target_deltas[sample_i, :n_targets], target_class_ids[sample_i, :n_targets],
                    rcnn_bbox[sample_i, :n_targets])
                mrcnn_mask_loss = compute_mrcnn_mask_loss(
                    target_mask[sample_i, :n_targets], target_class_ids[sample_i, :n_targets],
                    mrcnn_mask[sample_i, :n_targets])
                rcnn_class_losses.append(rcnn_class_loss[None])
                rcnn_bbox_losses.append(rcnn_bbox_loss[None])
                mrcnn_mask_losses.append(mrcnn_mask_loss[None])
            else:
                rcnn_class_losses.append(torch.tensor([0.0], dtype=torch.float, device=molded_images.device))
                rcnn_bbox_losses.append(torch.tensor([0.0], dtype=torch.float, device=molded_images.device))
                mrcnn_mask_losses.append(torch.tensor([0.0], dtype=torch.float, device=molded_images.device))
        rcnn_class_losses = torch.cat(rcnn_class_losses, dim=0)
        rcnn_bbox_losses = torch.cat(rcnn_bbox_losses, dim=0)
        mrcnn_mask_losses = torch.cat(mrcnn_mask_losses, dim=0)

        return (rpn_class_losses, rpn_bbox_losses, rcnn_class_losses, rcnn_bbox_losses, mrcnn_mask_losses)


    def mask_detect_forward(self, images, mrcnn_feature_maps, det_boxes, n_dets_per_sample):
        """Runs the mask stage of the detection pipeline.

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
        device = det_boxes.device

        image_size = images.shape[2:]

        # Convert boxes to normalized coordinates
        # TODO: let DetectionLayer return normalized coordinates to avoid
        #       unnecessary conversions
        h, w = image_size
        scale = torch.tensor(np.array([h, w, h, w]), dtype=torch.float, device=device)

        # Normalized boxes for mask generation
        det_boxes_nrm = det_boxes / scale[None, None, :]


        # Generate masks
        mrcnn_mask = []
        for mask_i in range(0, det_boxes_nrm.size()[1], self.config.DETECTION_BLOCK_SIZE_INFERENCE):
            mask_j = min(mask_i + self.config.DETECTION_BLOCK_SIZE_INFERENCE, det_boxes_nrm.size()[1])

            n_dets_per_sample_block = [
                min(mask_j, n_dets) - min(mask_i, n_dets) for n_dets in n_dets_per_sample
            ]

            mrcnn_mask_block = self.mask(mrcnn_feature_maps, det_boxes_nrm[:, mask_i:mask_j, ...],
                                         n_dets_per_sample_block, image_size)
            # mrcnn_mask: [batch, detection_index, object_class, height, width)

            mrcnn_mask_block = mrcnn_mask_block.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            mrcnn_mask.append(mrcnn_mask_block)

        mrcnn_mask = np.concatenate(mrcnn_mask, axis=1)

        return mrcnn_mask


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
        # rpn_feature_maps: [batch, channels, height, width]
        # mrcnn_feature_maps: [batch, channels, height, width]
        # rpn_bbox: [batch, anchors, 4]
        # rpn_rois: [batch, n_rois_after_nms, 4]
        # roi_scores: [batch, n_rois_after_nms]
        # n_rois_per_sample: [batch]
        image_size = images.shape[2:]

        rpn_feature_maps, mrcnn_feature_maps, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample = self.rpn_detect_forward(
            images)

        # det_boxes: [batch, num_detections, (y1, x1, y2, x2)] in image coordinates
        # det_class_ids: [batch, num_detections]
        # det_scores: [batch, num_detections]
        det_boxes, det_class_ids, det_scores, n_dets_per_sample = self.rcnn_detect_forward(
            image_size, image_windows, mrcnn_feature_maps, rpn_rois, n_rois_per_sample, override_class=override_class)

        if is_empty(det_boxes) or is_empty(det_class_ids) or is_empty(det_scores):
            # No detections
            n_images = images.shape[0]
            return [(np.zeros((n_images, 0, 4), dtype=np.float32),
                     np.zeros((n_images, 0), dtype=int),
                     np.zeros((n_images, 0), dtype=np.float32),
                     np.zeros((n_images, 0) + tuple(self.config.MASK_SHAPE) + (self.config.NUM_CLASSES,), dtype=np.float32))
                    for i in range(n_images)]


        #
        # Detections done
        #

        # Convert to numpy
        det_boxes_np = det_boxes.data.cpu().numpy()
        det_class_ids_np = det_class_ids.data.cpu().numpy()
        det_scores_np = det_scores.data.cpu().numpy()

        mrcnn_mask = self.mask_detect_forward(images, mrcnn_feature_maps, det_boxes, n_dets_per_sample)


        return split_detections(n_dets_per_sample, det_boxes_np, det_class_ids_np, det_scores_np, mrcnn_mask)
