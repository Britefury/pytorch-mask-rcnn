import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn.nms.nms_wrapper import nms
from .detections import RPNDetections
from .utils import SamePad2d, compute_overlaps, concatenate_detections, split_detections, torch_tensor_to_int_list

############################################################
#  Region Proposal Network
############################################################

class RPNHead (nn.Module):
    """Region Proposal Network head model.

    config: configuration object
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: number of channels per feature map pixel incoming from FPN

    Invoking this model returns (rpn_logits, rpn_probs, rpn_bbox):
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, config, anchors_per_location, anchor_stride, depth):
        super(RPNHead, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        if config.TORCH_PADDING:
            self.padding = None
            self.conv_shared = nn.Conv2d(self.depth, config.RPN_HIDDEN_CHANNELS, kernel_size=3, stride=self.anchor_stride, padding=1)
        else:
            self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
            self.conv_shared = nn.Conv2d(self.depth, config.RPN_HIDDEN_CHANNELS, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(config.RPN_HIDDEN_CHANNELS,
                                    config.n_rpn_logits_per_anchor * anchors_per_location,
                                    kernel_size=1, stride=1)
        if config.RPN_OBJECTNESS_FUNCTION in {'softmax', 'focal'}:
            self.softmax = nn.Softmax(dim=2)
        else:
            self.softmax = None
        self.conv_bbox = nn.Conv2d(config.RPN_HIDDEN_CHANNELS, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        if self.padding is not None:
            x = self.padding(x)
        x = self.relu(self.conv_shared(x))

        # Anchor Score. [batch, anchors per location * 1/2, height, width].
        rpn_class_logits = self.conv_class(x)

        # [batch, anchors per location * 1/2, height, width] -> [batch, height, width, anchors per location * 1/2]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
        rpn_class_logits = rpn_class_logits.contiguous()

        if self.softmax is not None:
            rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)
            # Softmax on last dimension of BG/FG.
            rpn_probs = self.softmax(rpn_class_logits)
        else:
            rpn_class_logits = rpn_class_logits.view(x.size()[0], -1)
            # Sigmoid to get FG probability
            rpn_probs = F.sigmoid(rpn_class_logits)


        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0,2,3,1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]

    def detectron_weight_mapping(self):
        def convert_conv_bbox_w(shape, src_blobs):
            val = src_blobs['rpn_bbox_pred_fpn2_w']
            val2 = val.reshape((3, 4) + val.shape[1:])
            val2 = val2[:, [1, 0, 3, 2], :, :, :]
            val = val2.reshape(val.shape)
            return val

        def convert_conv_bbox_b(shape, src_blobs):
            val = src_blobs['rpn_bbox_pred_fpn2_b']
            val2 = val.reshape((3, 4))
            val2 = val2[:, [1, 0, 3, 2]]
            val = val2.reshape(val.shape)
            return val

        det_map = {}
        orphans = []
        det_map['conv_shared.weight'] = 'conv_rpn_fpn2_w'
        det_map['conv_shared.bias'] = 'conv_rpn_fpn2_b'
        det_map['conv_class.weight'] = 'rpn_cls_logits_fpn2_w'
        det_map['conv_class.bias'] = 'rpn_cls_logits_fpn2_b'
        det_map['conv_bbox.weight'] = convert_conv_bbox_w
        det_map['conv_bbox.bias'] = convert_conv_bbox_b
        return det_map, orphans



############################################################
#  RPN prediction to proposal conversion
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.

    :param boxes: [..., 4] where last dimension is y1, x1, y2, x2
    :param deltas: [..., 4] where last dimension is [dy, dx, log(dh), log(dw)]
    :return: boxes as a [..., 4] tensor
    """
    # Convert to y, x, h, w
    height = boxes[..., 2] - boxes[..., 0]
    width = boxes[..., 3] - boxes[..., 1]
    center_y = boxes[..., 0] + 0.5 * height
    center_x = boxes[..., 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[..., 0] * height
    center_x += deltas[..., 1] * width
    height *= torch.exp(deltas[..., 2])
    width *= torch.exp(deltas[..., 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=-1)
    return result

def clip_boxes(boxes, window):
    """
    Clip boxes to lie within window

    :param boxes: [N, 4] each col is y1, x1, y2, x2
    :param window: [4] in the form y1, x1, y2, x2
    :return: clipped boxes as a [N, 4] tensor

    """
    boxes = torch.stack( \
        [boxes[..., 0].clamp(float(window[0]), float(window[2])),
         boxes[..., 1].clamp(float(window[1]), float(window[3])),
         boxes[..., 2].clamp(float(window[0]), float(window[2])),
         boxes[..., 3].clamp(float(window[1]), float(window[3]))], dim=-1)
    return boxes

def rpn_preds_to_proposals(rpn_pred_probs, rpn_box_deltas, proposal_count, nms_threshold, anchors,
                           image_size, pre_nms_limit, config):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    :param rpn_pred_probs: predicted background/foreground probabilities from region proposal network RPN
        Either:
            (sample, anchors,) array  where the last dimension is [fg_prob] if
                `config.RPN_OBJECTNESS_FUNCTION == 'sigmoid'`
        Or:
            (sample, anchors, 2) array  where the last dimension is [bg_prob, fg_prob] otherwise
    :param rpn_box_deltas: predicted bounding box deltas from region proposal network RPN
        (sample, anchors, 4) array  where the last dimension is [dy, dx, log(dh), log(dw)]
    :param proposal_count: maximum number of proposals to be generated
    :param nms_threshold: Non-maximum suppression threshold
    :param anchors: Anchors as a (A,4) Torch tensor
    :param image_size: image size as a (height, width) tuple
    :param pre_nms_limit: number of pre-NMS boxes to consider
    :param config:
    :return: (normalized_boxes, scores, rois_per_sample):
        normalized_boxes: proposals in normalized coordinates [sample, rois, (y1, x1, y2, x2)] (Torch tensor)
        scores: objectness scores [sample, rois] (Torch tensor)
        rois_per_sample: list giving the number of ROIs in each sample
    """
    device = rpn_pred_probs.device
    n_samples = rpn_pred_probs.shape[0]

    # Select scores and deltas corresponding to valid anchors
    if config.n_rpn_logits_per_anchor == 2:
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = rpn_pred_probs[:, :, 1]
    else:
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = rpn_pred_probs

    # Box deltas [batch, num_rois, 4]
    if config.RPN_BBOX_USE_STD_DEV:
        std_dev = torch.tensor(np.reshape(config.BBOX_STD_DEV, [1, 1, 4]), requires_grad=False, device=device, dtype=torch.float)
        rpn_box_deltas = rpn_box_deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(pre_nms_limit, anchors.size()[0])
    scores, order = scores.sort(dim=1, descending=True)
    order = order[:, :pre_nms_limit]
    scores = scores[:, :pre_nms_limit]
    rpn_box_deltas = rpn_box_deltas[torch.arange(n_samples, dtype=torch.long)[:, None], order, :]
    anchors = anchors[order, :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, rpn_box_deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = image_size
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    retained = []
    rois_per_sample = []
    max_n_rois = 0
    for sample_i in range(n_samples):
        keep = nms(torch.cat((boxes[sample_i], scores[sample_i].unsqueeze(1)), 1).detach(), nms_threshold)
        keep = keep[:proposal_count]
        n_rois = keep.shape[0]
        max_n_rois = max(max_n_rois, n_rois)
        retained.append(keep)
        rois_per_sample.append(n_rois)

    # Normalize dimensions to range of 0 to 1.
    norm = torch.tensor(np.array([height, width, height, width]), dtype=torch.float, device=device)

    retained_norm_boxes = torch.zeros(n_samples, max_n_rois, 4, dtype=torch.float, device=device)
    retained_scores = torch.zeros(n_samples, max_n_rois, dtype=torch.float, device=device)
    for sample_i, (n_rois, keep) in enumerate(zip(rois_per_sample, retained)):
        retained_norm_boxes[sample_i, :n_rois, :] = boxes[sample_i, keep, :] / norm
        retained_scores[sample_i, :n_rois] = scores[sample_i, keep]

    return retained_norm_boxes, retained_scores, rois_per_sample


def rpn_preds_to_proposals_by_level(rpn_pred_probs_by_lvl, rpn_box_deltas_by_lvl, proposal_count, nms_threshold, anchors_by_lvl,
                                    image_size, pre_nms_limit, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Processes FPN levels separately

    :param rpn_pred_probs_by_lvl: predicted background/foreground probabilities from region proposal network RPN
        list of (batch, anchors, 2) arrays where the last dimension is [bg_prob, fg_prob]; one array for each
        FPN level
    :param rpn_box_deltas_by_lvl: predicted bounding box deltas from region proposal network RPN
        list (batch, anchors, 4) arrays where the last dimension is [dy, dx, log(dh), log(dw)]; one for each
        FPN level
    :param proposal_count: maximum number of proposals to be generated
    :param nms_threshold: Non-maximum suppression threshold
    :param anchors_by_lvl: Anchors as a list of (A,4) Torch tensors; one for each FPN level
    :param image_size: image size as a (height, width) tuple
    :param pre_nms_limit: number of pre-NMS boxes to consider
    :param config:
    :return: (normalized_boxes, scores, n_boxes_per_sample) where
        normalized_boxes: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
        scores: proposal objectness scores [batch, rois] (Torch tensor)
        n_boxes_per_sample: list giving number of ROIs in each sample in the batch
    """
    n_samples = rpn_pred_probs_by_lvl[0].shape[0]
    device = rpn_pred_probs_by_lvl[0].device
    boxes_by_lvl = []
    scores_by_lvl = []
    n_rois_per_sample_by_level = []
    total_n_rois_per_sample = None
    for lvl_i, (rpn_pred_probs, rpn_box_deltas, anchors) in enumerate(zip(
            rpn_pred_probs_by_lvl, rpn_box_deltas_by_lvl, anchors_by_lvl)):
        lvl_boxes, lvl_scores, lvl_rois_per_sample = rpn_preds_to_proposals(
            rpn_pred_probs, rpn_box_deltas, proposal_count, nms_threshold, anchors, image_size, pre_nms_limit,
            config=config)
        boxes_by_lvl.append(lvl_boxes)
        scores_by_lvl.append(lvl_scores)
        n_rois_per_sample_by_level.append(lvl_rois_per_sample)
        if total_n_rois_per_sample is None:
            total_n_rois_per_sample = lvl_rois_per_sample
        else:
            total_n_rois_per_sample = [a + b for a, b in zip(total_n_rois_per_sample, lvl_rois_per_sample)]
    max_rois_per_sample = max(total_n_rois_per_sample)
    boxes = torch.zeros(n_samples, max_rois_per_sample, 4, dtype=torch.float, device=device)
    scores = torch.zeros(n_samples, max_rois_per_sample, dtype=torch.float, device=device)

    pos = [0 for _ in range(n_samples)]
    for lvl_i, n_rois_per_sample in enumerate(n_rois_per_sample_by_level):
        for sample_i in range(n_samples):
            start = pos[sample_i]
            end = start + n_rois_per_sample[sample_i]
            boxes[sample_i, start:end, :] = boxes_by_lvl[lvl_i][sample_i, :n_rois_per_sample[sample_i], :]
            scores[sample_i, start:end] = scores_by_lvl[lvl_i][sample_i, :n_rois_per_sample[sample_i]]
            pos[sample_i] = end


    return boxes, scores, total_n_rois_per_sample


############################################################
#  RPN Loss Functions
############################################################

RPN_CLS_POSITIVE = 1
RPN_CLS_NEGATIVE = -1
RPN_CLS_NEUTRAL = 0

def focal_loss_with_logits_cat_cross_ent(class_logits, targets, num_pos_samples, weight=None, alpha=0.25, gamma=2.0):
    """
    Compute focal loss with predicted logits, using categorical cross-entropy

    :param class_logits: predicted class logits as a [sample, class] tensor
    :param targets: per-sample class index as a [sample] tensor
    :param num_pos_samples: number of positive samples, used for normalization, as a torch scalar
    :param weight: per-sample weight as a [sample] tensor
    :param alpha: alpha value for alpha balancing
    :param gamma: focal loss exponent
    :return: focal loss as a torch scalar
    """
    # Probabilities, clamp to prevent log(0)
    class_prob = F.softmax(class_logits, dim=1)
    # p_t = p if target == 1
    # p_t = (1 - p) if target == 0
    class_prob_t = torch.gather(class_prob, 1, targets[:, None])[:, 0]

    # alpha if target == 1
    # (1 - alpha) if target == 0
    sample_weight = (alpha * targets.float() + (1.0 - alpha) * (1.0 - targets.float()))
    if weight is not None:
        sample_weight = sample_weight * weight
    # weight = w * (1 - p_t)**gamma
    sample_weight = sample_weight * (1.0 - class_prob_t).pow(gamma).detach()

    l = F.cross_entropy(class_logits, targets, reduce=False) * sample_weight

    num_pos_samples = num_pos_samples.clamp(min=1.0)
    return l.sum() / num_pos_samples


def focal_loss_with_logits(class_logits, targets, num_pos_samples, weight=None, alpha=0.25, gamma=2.0):
    """
    Compute focal loss with predicted logits

    :param class_logits: predicted class logits as a [sample, class] tensor
    :param targets: per-sample class index as a [sample] tensor
    :param num_pos_samples: number of positive samples, used for normalization, as a torch scalar
    :param weight: per-sample weight as a [sample] tensor
    :param alpha: alpha value for alpha balancing
    :param gamma: focal loss exponent
    :return: focal loss as a torch scalar
    """
    # One hot representation of targets
    targets_one_hot = torch.zeros_like(class_logits)
    targets_one_hot.scatter_(1, targets[:, None], 1)

    # Probabilities, clamp to prevent log(0)
    class_prob = F.softmax(class_logits, dim=1)
    # p_t = p if target == 1
    # p_t = (1 - p) if target == 0
    class_prob_t = class_prob * targets_one_hot + (1.0 - class_prob) * (1.0 - targets_one_hot)

    # alpha if target == 1
    # (1 - alpha) if target == 0
    sample_weight = (alpha * targets_one_hot + (1.0 - alpha) * (1.0 - targets_one_hot))
    if weight is not None:
        sample_weight = sample_weight * weight[:, None]
    # weight = w * (1 - p_t)**gamma
    sample_weight = sample_weight * (1.0 - class_prob_t).pow(gamma).detach()
    num_pos_samples = num_pos_samples.clamp(min=1.0)
    return F.binary_cross_entropy_with_logits(class_logits, targets_one_hot, weight=sample_weight,
                                              size_average=False) / num_pos_samples


def compute_rpn_losses(config, rpn_pred_class_logits, rpn_pred_bbox, rpn_target_match, rpn_target_bbox,
                       rpn_target_num_pos_per_sample):
    """RPN anchor classifier loss.

    :param config: configuration object
    :param rpn_pred_class_logits: [batch, anchors, 2] RPN classifier logits for FG/BG if using softmax or focal loss for
        RPN class (see config.RPN_OBJECTNESS_FUNCTION),
        [batch, anchors] RPN classifier FG logits if using sigmoid.
    :param rpn_pred_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    :param rpn_target_match: [batch, anchors]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    :param rpn_target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    :param rpn_target_num_pos_per_sample: [batch] number of positives per sample

    :return: (cls_loss, box_loss) where cls_loss and box_loss are the RPN objectness and box losses as torch scalars
    """
    device = rpn_pred_class_logits.device

    rpn_target_num_pos_per_sample = torch_tensor_to_int_list(rpn_target_num_pos_per_sample)

    if len(rpn_target_match.size()) == 3:
        # Squeeze last dim to simplify
        rpn_target_match = rpn_target_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_target_match == RPN_CLS_POSITIVE).long()

    if config.RPN_TRAIN_ANCHORS_PER_IMAGE is None:
        # No sample balancing; use all samples, so don't select
        rpn_pred_class_logits = rpn_pred_class_logits.view(-1, rpn_pred_class_logits.shape[-1])
        anchor_class = anchor_class.view(-1)
        cls_weight = (rpn_target_match != RPN_CLS_NEUTRAL).view(-1).float()

        tgt_box_for_loss = rpn_target_bbox.view(-1, 4)
        rpn_box_for_loss = rpn_pred_bbox.view(-1, 4)
        box_weight = anchor_class.view(-1).float()
    else:
        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors don't.
        indices = torch.nonzero(rpn_target_match != RPN_CLS_NEUTRAL)
        pos_indices = torch.nonzero(anchor_class)

        # Pick rows that contribute to the loss and filter out the rest.
        rpn_pred_class_logits = rpn_pred_class_logits[indices[:, 0], indices[:, 1], ...]
        anchor_class = anchor_class[indices[:, 0], indices[:, 1]]
        cls_weight = torch.ones(rpn_pred_class_logits.shape[0], dtype=torch.float, device=device)

        tgt_box_for_loss = []
        for sample_i, n_pos in enumerate(rpn_target_num_pos_per_sample):
            if n_pos > 0:
                tgt_box_for_loss.append(rpn_target_bbox[sample_i, :n_pos, :])
        tgt_box_for_loss = torch.cat(tgt_box_for_loss, dim=0)
        rpn_box_for_loss = rpn_pred_bbox[pos_indices[:, 0], pos_indices[:, 1]]
        box_weight = torch.ones(rpn_box_for_loss.shape[0], dtype=torch.float, device=device)

    if config.RPN_OBJECTNESS_FUNCTION == 'sigmoid':
        # Binary cross-entropy loss
        cls_loss = F.binary_cross_entropy_with_logits(
            rpn_pred_class_logits, anchor_class.float(), weight=cls_weight)
        box_loss = (F.smooth_l1_loss(rpn_box_for_loss, tgt_box_for_loss, reduce=False) *
                    box_weight[:, None]).mean()
    elif config.RPN_OBJECTNESS_FUNCTION == 'softmax':
        # Cross-entropy loss
        cls_loss = (F.cross_entropy(rpn_pred_class_logits, anchor_class, reduce=False) * cls_weight).mean()
        box_loss = (F.smooth_l1_loss(rpn_box_for_loss, tgt_box_for_loss, reduce=False) *
                    box_weight[:, None]).mean()
    elif config.RPN_OBJECTNESS_FUNCTION == 'focal':
        # Focal loss
        num_pos = anchor_class.sum().float()
        cls_loss = focal_loss_with_logits(rpn_pred_class_logits, anchor_class, num_pos,
                                          weight=cls_weight, alpha=config.RPN_FOCAL_LOSS_POS_CLS_WEIGHT,
                                          gamma=config.RPN_FOCAL_LOSS_EXPONENT)
        box_loss = (F.smooth_l1_loss(rpn_box_for_loss, tgt_box_for_loss, reduce=False) *
                    box_weight[:, None]).sum() / num_pos
    else:
        raise ValueError('Invalid value {} for config.RPN_OBJECTNESS_FUNCTION'.format(
            config.RPN_OBJECTNESS_FUNCTION))


    return cls_loss, box_loss

def compute_rpn_losses_per_sample(config, rpn_pred_class_logits, rpn_pred_bbox, rpn_target_match, rpn_target_bbox,
                                  rpn_target_num_pos_per_sample):
    """RPN anchor classifier loss.

    :param config: configuration object
    :param rpn_pred_class_logits: [batch, anchors, 2] RPN classifier logits for FG/BG if using softmax or focal loss for
        RPN class (see config.RPN_OBJECTNESS_FUNCTION),
        [batch, anchors] RPN classifier FG logits if using sigmoid.
    :param rpn_pred_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    :param rpn_target_match: [batch, anchors]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    :param rpn_target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    :param rpn_target_num_pos_per_sample: [batch] number of positives per sample

    :return: (cls_loss, box_loss) where cls_loss and box_loss are the RPN objectness and box losses as torch scalars
    """

    device = rpn_pred_class_logits.device

    if len(rpn_target_match.size()) == 3:
        # Squeeze last dim to simplify
        rpn_target_match = rpn_target_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_target_match == RPN_CLS_POSITIVE).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors don't.
    non_neutral = rpn_target_match != RPN_CLS_NEUTRAL
    cls_losses = []
    box_losses = []
    for sample_i, n_pos in enumerate(rpn_target_num_pos_per_sample):
        if config.RPN_TRAIN_ANCHORS_PER_IMAGE is None:
            # No sample balancing; use all samples, so don't select
            sample_class_logits = rpn_pred_class_logits[sample_i]
            sample_anchor_class = anchor_class[sample_i]
            cls_weight = non_neutral[sample_i].float()

            if n_pos > 0:
                sample_rpn_bbox = rpn_pred_bbox[sample_i, :]
                sample_pos_target_box = rpn_target_bbox[sample_i, :, :]
            else:
                sample_rpn_bbox = sample_pos_target_box = None
            box_weight = sample_anchor_class.float()
        else:
            indices = torch.nonzero(non_neutral[sample_i])
            pos_indices = torch.nonzero(anchor_class[sample_i])

            # Pick rows that contribute to the loss and filter out the rest.
            sample_class_logits = rpn_pred_class_logits[sample_i, indices[:, 0], ...]
            sample_anchor_class = anchor_class[sample_i, indices[:, 0]]
            cls_weight = torch.ones(sample_class_logits.shape[0], dtype=torch.float, device=device)

            if n_pos > 0:
                sample_rpn_bbox = rpn_pred_bbox[sample_i, pos_indices[:, 0]]
                sample_pos_target_box = rpn_target_bbox[sample_i, :n_pos, :]
                box_weight = torch.ones(sample_rpn_bbox.shape[0], dtype=torch.float, device=device)
            else:
                sample_rpn_bbox = sample_pos_target_box = None
                box_weight = None

        if config.RPN_OBJECTNESS_FUNCTION == 'sigmoid':
            # Binary cross-entropy loss
            cls_loss = F.binary_cross_entropy_with_logits(
                sample_class_logits, sample_anchor_class.float(), weight=cls_weight)
            if n_pos > 0:
                box_loss = (F.smooth_l1_loss(sample_rpn_bbox, sample_pos_target_box, reduce=False) *
                            box_weight[:, None]).mean()
            else:
                box_loss = torch.tensor(0.0, dtype=torch.float, device=rpn_pred_bbox.device)
        elif config.RPN_OBJECTNESS_FUNCTION == 'softmax':
            # Cross-entropy loss
            cls_loss = (F.cross_entropy(sample_class_logits, sample_anchor_class.long(), reduce=False) * cls_weight).mean()
            if n_pos > 0:
                box_loss = (F.smooth_l1_loss(sample_rpn_bbox, sample_pos_target_box, reduce=False) *
                            box_weight[:, None]).mean()
            else:
                box_loss = torch.tensor(0.0, dtype=torch.float, device=rpn_pred_bbox.device)
        elif config.RPN_OBJECTNESS_FUNCTION == 'focal':
            # Cross-entropy loss
            num_pos = sample_anchor_class.sum().float()
            cls_loss = focal_loss_with_logits(sample_class_logits, sample_anchor_class, num_pos,
                                              weight=cls_weight, alpha=config.RPN_FOCAL_LOSS_POS_CLS_WEIGHT,
                                              gamma=config.RPN_FOCAL_LOSS_EXPONENT)
            if n_pos > 0:
                box_loss = (F.smooth_l1_loss(sample_rpn_bbox, sample_pos_target_box, reduce=False) *
                            box_weight[:, None]).sum() / num_pos
            else:
                box_loss = torch.tensor(0.0, dtype=torch.float, device=rpn_pred_bbox.device)
        else:
            raise ValueError('Invalid value {} for config.RPN_OBJECTNESS_FUNCTION'.format(
                config.RPN_OBJECTNESS_FUNCTION))

        cls_losses.append(cls_loss[None])
        box_losses.append(box_loss[None])

    return torch.cat(cls_losses, dim=0), torch.cat(box_losses, dim=0)


############################################################
#  Target generation
############################################################

def build_rpn_targets_balanced(config, anchors, valid_anchors_mask, gt_class_ids, gt_boxes):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    Only `config.RPN_TRAIN_ANCHORS_PER_IMAGE` anchors will have non-neutral values and at most
    `config.RPN_TRAIN_ANCHORS_PER_IMAGE` box deltas with be generated

    :param config: configuration object
    :param anchors: [num_anchors, (y1, x1, y2, x2)]
    :param valid_anchors_mask: [num_anchors]
    :param gt_class_ids: [num_gt_boxes] Integer class IDs.
    :param gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    :return: (rpn_target_match, rpn_bbox_deltas, num_positives) where
        rpn_target_match: NumPy array [num_anchors] (int32) matches between anchors and GT boxes.
                      1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox_deltas: NumPy array [M, (dy, dx, log(dh), log(dw))] Anchor bbox deltas (M = number of boxes)
        num_positives: int; number of positives
    """

    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    valid_anchors = anchors[valid_anchors_mask]
    rpn_match = np.full([len(valid_anchors)], RPN_CLS_NEUTRAL, dtype=np.int32)
    # RPN bounding box deltas: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_deltas = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    if len(gt_boxes) == 0:
        # Special case
        rpn_match_all = np.zeros([len(valid_anchors_mask)], dtype=np.int32)
        rpn_match_all.fill(RPN_CLS_NEGATIVE)
        return rpn_match_all, rpn_deltas, 0

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = compute_overlaps(valid_anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All valid_anchors don't intersect a crowd
        no_crowd_bool = np.ones([len(valid_anchors)], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(valid_anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = RPN_CLS_NEGATIVE
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = RPN_CLS_POSITIVE
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = RPN_CLS_POSITIVE

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == RPN_CLS_POSITIVE)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = RPN_CLS_NEUTRAL
    # Same for negative proposals
    ids = np.where(rpn_match == RPN_CLS_NEGATIVE)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == RPN_CLS_POSITIVE))
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = RPN_CLS_NEUTRAL

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    positive_anchor_ids = np.where(rpn_match == RPN_CLS_POSITIVE)[0]
    positive_gt = gt_boxes[anchor_iou_argmax[positive_anchor_ids]]
    positive_anc = valid_anchors[positive_anchor_ids]

    # Convert coordinates to center plus width/height.
    # GT Box
    positive_gt_size = positive_gt[:, 2:4] - positive_gt[:, 0:2]
    positive_gt_centre = (positive_gt[:, 0:2] + positive_gt[:, 2:4]) * 0.5
    # Anchor
    positive_anc_size = positive_anc[:, 2:4] - positive_anc[:, 0:2]
    positive_anc_centre = (positive_anc[:, 0:2] + positive_anc[:, 2:4]) * 0.5

    rpn_deltas[:len(positive_anchor_ids), 0:2] = (positive_gt_centre - positive_anc_centre) / positive_anc_size
    rpn_deltas[:len(positive_anchor_ids), 2:4] = np.log(positive_gt_size / positive_anc_size)

    if config.RPN_BBOX_USE_STD_DEV:
        rpn_deltas /= config.BBOX_STD_DEV[None, :]

    # Reverse valid anchor selection
    rpn_match_all = np.zeros([len(valid_anchors_mask)], dtype=np.int32)

    rpn_match_all[valid_anchors_mask] = rpn_match

    return rpn_match_all, rpn_deltas, np.count_nonzero(rpn_match == RPN_CLS_POSITIVE)


def build_rpn_targets_all(config, anchors, valid_anchors_mask, gt_class_ids, gt_boxes):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    All targets that can be assigned are generated

    :param config: configuration object
    :param anchors: [num_anchors, (y1, x1, y2, x2)]
    :param valid_anchors_mask: [num_anchors]
    :param gt_class_ids: [num_gt_boxes] Integer class IDs.
    :param gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    :return: (rpn_target_match, rpn_bbox_deltas, num_positives) where
        rpn_target_match: NumPy array [num_anchors] (int32) matches between anchors and GT boxes.
                      1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox_deltas: NumPy array [num_anchors, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        num_positives: int; number of positives
    """

    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    valid_anchors = anchors[valid_anchors_mask]
    rpn_match = np.full([len(valid_anchors)], RPN_CLS_NEUTRAL, dtype=np.int32)
    # RPN bounding box deltas: [n_anchors, (dy, dx, log(dh), log(dw))]
    rpn_deltas = np.zeros([len(valid_anchors), 4])

    if len(gt_boxes) == 0:
        # Special case
        rpn_match_all = np.zeros([len(valid_anchors_mask)], dtype=np.int32)
        rpn_match_all.fill(-1)
        rpn_bbox_all =  np.zeros([len(valid_anchors_mask), 4], dtype=np.float32)
        return rpn_match_all, rpn_bbox_all, 0

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = compute_overlaps(valid_anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All valid_anchors don't intersect a crowd
        no_crowd_bool = np.ones([len(valid_anchors)], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(valid_anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = RPN_CLS_NEGATIVE
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = RPN_CLS_POSITIVE
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = RPN_CLS_POSITIVE

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == RPN_CLS_POSITIVE)[0]

    # TODO: use box_refinment() rather than duplicating the code here
    gt_boxes_per_pos_anchor = gt_boxes[anchor_iou_argmax[ids]]
    pos_anchors = valid_anchors[ids]

    # Convert coordinates to center plus width/height.
    # GT Box
    gt_size = gt_boxes_per_pos_anchor[:, 2:4] - - gt_boxes_per_pos_anchor[:, 0:2]
    gt_centre = (gt_boxes_per_pos_anchor[:, 0:2] + gt_boxes_per_pos_anchor[:, 2:4]) * 0.5

    # Anchor
    anc_size = pos_anchors[:, 2:4] - - pos_anchors[:, 0:2]
    anc_centre = (pos_anchors[:, 0:2] + pos_anchors[:, 2:4]) * 0.5

    # Compute the bbox refinement that the RPN should predict.
    rpn_deltas[ids, 0:2] = (gt_centre - anc_centre) / anc_size
    rpn_deltas[ids, 2:4] =  np.log(gt_size / anc_size)

    # Normalize
    if config.RPN_BBOX_USE_STD_DEV:
        rpn_deltas[ids] /= config.BBOX_STD_DEV

    # Reverse valid anchor selection
    rpn_match_all = np.zeros([len(valid_anchors_mask)], dtype=np.int32)
    rpn_match_all[valid_anchors_mask] = rpn_match

    rpn_bbox_all = np.zeros([len(valid_anchors_mask), 4], dtype=np.float32)
    rpn_bbox_all[valid_anchors_mask] = rpn_deltas

    return rpn_match_all, rpn_bbox_all, np.count_nonzero(rpn_match == RPN_CLS_POSITIVE)




############################################################
#  RPN Model
############################################################

class RPNBaseModel (nn.Module):
    """
    RPN Base model

    Network consists of:
    - feature pyramid network for feature extraction
    - RPN head for proposal generation

    Provides methods for:
    - converting ground truth boxes to targets for RPN
    - getting feature maps and RPN proposals for input images

    To build an RPN based model, inherit from AbstractRPNModel as it provides methods for training and detection.
    """

    def __init__(self, config):
        """
        config: A Sub-class of the Config class
        """
        super(RPNBaseModel, self).__init__()
        self.config = config

        # Build the backbone FPN
        self.fpn = self.build_backbone_fpn(config, 256)

        # RPN
        self.rpn = RPNHead(config, len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)




    def build_backbone_fpn(self, config, out_channels):
        raise NotImplementedError('Abstract for {}'.format(type(self)))



    def ground_truth_to_rpn_targets(self, image_shape, gt_class_ids, gt_boxes):
        """
        Generate RPN training targets given image shape and ground truth
        :param image_shape: image size as a tuple (height, width)
        :param gt_class_ids: ground truth box class IDs as a [N] array
        :param gt_boxes: ground truth boxes as a [N, 4] array
        :return: (rpn_target_match, rpn_bbox_deltas, num_positives)
            rpn_target_match: NumPy array [num_anchors] (int32) matches between anchors and GT boxes.
                              1 = positive anchor, -1 = negative anchor, 0 = neutral
            rpn_bbox_deltas: NumPy array [M, (dy, dx, log(dh), log(dw))] Anchor bbox deltas (M = number of boxes)
            num_positives: int; number of positives
        """
        anchors, valid_mask = self.config.ANCHOR_CACHE.get_anchors_and_valid_masks_for_image_shape(image_shape)
        if self.config.RPN_TRAIN_ANCHORS_PER_IMAGE is not None:
            rpn_match, rpn_bbox, num_positives = build_rpn_targets_balanced(self.config, anchors, valid_mask,
                                                                            gt_class_ids, gt_boxes)
        else:
            rpn_match, rpn_bbox, num_positives = build_rpn_targets_all(self.config, anchors, valid_mask, gt_class_ids,
                                                                       gt_boxes)
        return rpn_match, rpn_bbox, num_positives


    def _feature_maps_and_rpn_preds_all(self, images):
        """
        Generate feature maps and RPN predictions for given images.
        Joins output from FPN levels.

        :param images: images as a [batch, channel, height, width] tensor
        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits, rpn_class_probs, rpn_bbox_deltas):
            rpn_feature_maps: per-FPN level feature maps for RPN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            mrcnn_feature_maps: per-FPN level feature maps for RCNN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            rpn_class_logits: [batch, num_anchors, 2] or [batch, num_anchors] (depending on
                `self.config.RPN_OBJECTNESS_FUNCTION`) RPN class logit predictions
            rpn_class_probs: [batch, num_anchors, 2] or [batch, num_anchors] (depending on
                `self.config.RPN_OBJECTNESS_FUNCTION`) RPN class probability predictions
            rpn_bbox_deltas: [batch, num_anchors, 4] RPN box delta predictions
        """
        # Feature extraction
        rpn_feature_maps, mrcnn_feature_maps = self.fpn(images)

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class_probs, rpn_bbox_deltas = outputs

        return rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits, rpn_class_probs, rpn_bbox_deltas


    def _feature_maps_roi_preds_and_roi_all(self, images, pre_nms_limit, nms_threshold, proposal_count):
        """
        Generate feature maps, RPN predictions and proposed boxes for given images
        Joins output from FPN levels.

        :param images: images to process
        :param pre_nms_limit: number of proposals to pass prior to NMS
        :param nms_threshold: the NMS threshold to use
        :param proposal_count: number of proposals to pass subsequent to NMS
        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, n_rois_per_sample) where
            rpn_feature_maps: per-FPN level feature maps for RPN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            mrcnn_feature_maps: per-FPN level feature maps for RCNN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            rpn_class_logits: [batch, num_anchors, 2] or [batch, num_anchors] (depending on 
                `self.config.RPN_OBJECTNESS_FUNCTION`) RPN class logit predictions
            rpn_bbox_deltas: [batch, num_anchors, 4] RPN box delta predictions
            rpn_rois: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
            roi_scores: proposal objectness scores [batch, rois] (Torch tensor)
            n_rois_per_sample: list giving number of ROIs in each sample in the batch
        """
        device = images.device

        rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, rpn_class_probs, rpn_bbox_deltas = \
            self._feature_maps_and_rpn_preds_all(images)

        image_size = tuple(images.size()[2:])

        # Get anchors
        anchs_var = self.config.ANCHOR_CACHE.get_anchors_var_for_image_shape(image_size, device)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        rpn_rois, roi_scores, n_rois_per_sample = rpn_preds_to_proposals(
            rpn_class_probs, rpn_bbox_deltas, proposal_count=proposal_count,
            nms_threshold=nms_threshold,
            anchors=anchs_var, image_size=image_size,
            pre_nms_limit=pre_nms_limit, config=self.config)

        return rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample


    def _feature_maps_and_rpn_preds_by_level(self, images):
        """
        Generate feature maps and RPN predictions per FPN pyramid level for given images

        :param images: images as a [batch, channel, height, width] tensor
        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits_by_lvl, rpn_class_probs_by_lvl,
                  rpn_bbox_deltas_by_lvl):
            rpn_feature_maps: per-FPN level feature maps for RPN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            mrcnn_feature_maps: per-FPN level feature maps for RCNN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            rpn_class_logits_by_lvl: list of [batch, num_anchors, 2] or [batch, num_anchors] (depending on
                `self.config.RPN_OBJECTNESS_FUNCTION`) RPN class logit predictions; one per level
            rpn_class_probs_by_lvl: list of [batch, num_anchors, 2] or [batch, num_anchors] (depending on
                `self.config.RPN_OBJECTNESS_FUNCTION`) RPN class probability predictions; one per level
            rpn_bbox_deltas_by_lvl: list of [batch, num_anchors, 4] RPN box delta predictions; one per level
        """
        # Feature extraction
        rpn_feature_maps, mrcnn_feature_maps = self.fpn(images)

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        rpn_class_logits_by_lvl, rpn_class_probs_by_lvl, rpn_bbox_deltas_by_lvl = zip(*layer_outputs)

        return rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits_by_lvl, rpn_class_probs_by_lvl, rpn_bbox_deltas_by_lvl


    def _feature_maps_rpn_preds_and_roi_by_level(self, images, pre_nms_limit, nms_threshold, proposal_count):
        """
        Generate feature maps, RPN predictions per FPN pyramid level and proposed boxes for given images.
        The box proposals are generated separately for each FPN level then joined

        :param images: images to process
        :param pre_nms_limit: number of proposals to pass prior to NMS
        :param nms_threshold: the NMS threshold to use
        :param proposal_count: number of proposals to pass subsequent to NMS
        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits_by_lvl, rpn_bbox_deltas_by_lvl, rpn_rois,
                  n_rois_per_sample) where
            rpn_feature_maps: per-FPN level feature maps for RPN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            mrcnn_feature_maps: per-FPN level feature maps for RCNN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            rpn_class_logits_by_lvl: list of [batch, num_anchors, 2] or [batch, num_anchors] (depending on
                `self.config.RPN_OBJECTNESS_FUNCTION`) RPN class logit predictions; one per level
            rpn_bbox_deltas_by_lvl: list of [batch, num_anchors, 4] RPN box delta predictions; one per level
            rpn_rois: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
            roi_scores: proposal objectness scores [batch, rois] (Torch tensor)
            n_rois_per_sample: list giving number of ROIs in each sample in the batch
        """
        device = images.device

        rpn_feature_maps, rcnn_feature_maps, rpn_class_logits_by_lvl, rpn_class_probs_by_lvl, rpn_bbox_deltas_by_lvl = \
            self._feature_maps_and_rpn_preds_by_level(images)

        image_size = tuple(images.size()[2:])

        # Get anchors
        anchs_vars = self.config.ANCHOR_CACHE.get_anchors_var_for_image_shape_by_level(image_size, device)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        rpn_rois, roi_scores, n_rois_per_sample = rpn_preds_to_proposals_by_level(
            rpn_class_probs_by_lvl, rpn_bbox_deltas_by_lvl, proposal_count=proposal_count,
            nms_threshold=nms_threshold,
            anchors_by_lvl=anchs_vars, image_size=image_size,
            pre_nms_limit=pre_nms_limit, config=self.config)

        return rpn_feature_maps, rcnn_feature_maps, rpn_class_logits_by_lvl, rpn_bbox_deltas_by_lvl, \
               rpn_rois, roi_scores, n_rois_per_sample


    def _feature_maps_rpn_preds_and_roi(self, images, pre_nms_limit, nms_threshold, proposal_count):
        """
        Generate feature maps, RPN predictions and proposed boxes for given images.
        Will generate proposals in one go or separately per FPN pyramid level then join depending on the value
        of `self.config.RPN_FILTER_PROPOSALS_BY_LEVEL`.

        :param images: images to process
        :param pre_nms_limit: number of proposals to pass prior to NMS
        :param nms_threshold: the NMS threshold to use
        :param proposal_count: number of proposals to pass subsequent to NMS
        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, n_rois_per_sample) where
            rpn_feature_maps: per-FPN level feature maps for RPN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            mrcnn_feature_maps: per-FPN level feature maps for RCNN;
                list of [batch, feat_chn, lvl_height, lvl_width] tensors
            rpn_class_logits: [batch, num_anchors, 2] or [batch, num_anchors] (depending on
                `self.config.RPN_OBJECTNESS_FUNCTION`) RPN class logit predictions
            rpn_bbox_deltas: [batch, num_anchors, 4] RPN box delta predictions
            rpn_rois: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
            roi_scores: proposal objectness scores [batch, rois] (Torch tensor)
            n_rois_per_sample: list giving number of ROIs in each sample in the batch
        """
        if self.config.RPN_FILTER_PROPOSALS_BY_LEVEL:
            rpn_feature_maps, rcnn_feature_maps, rpn_class_logits_by_lvl, rpn_bbox_deltas_by_lvl, \
                rpn_rois, roi_scores, n_rois_per_sample = self._feature_maps_rpn_preds_and_roi_by_level(
                    images, pre_nms_limit, nms_threshold, proposal_count)
            rpn_class_logits = torch.cat(rpn_class_logits_by_lvl, 1)
            rpn_bbox_deltas = torch.cat(rpn_bbox_deltas_by_lvl, 1)
            return rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample
        else:
            return self._feature_maps_roi_preds_and_roi_all(images, pre_nms_limit, nms_threshold, proposal_count)


    def rpn_detect_forward(self, images):
        """Runs the RPN stage of the detection pipeline

        :param images: Tensor of images

        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample)
            rpn_feature_maps: list of [batch, channels, height, width] feature maps for RPN; one per FPN pyramid level
            mrcnn_feature_maps: list of [batch, channels, height, width] feature maps for RCNN; one per FPN pyramid level
            rpn_bbox_deltas: [batch, anchors, 4] predicted box deltas from RPN
            rpn_rois: [batch, n_rois_after_nms, 4] RPN proposed boxes in normalized co-ordinates
            roi_scores: [batch, n_rois_after_nms] RPN box objecness scores
            n_rois_per_sample: [batch] number of rois per sample in the batch
        """
        pre_nms_limit =  self.config.RPN_PRE_NMS_LIMIT_TEST
        nms_threshold =  self.config.RPN_NMS_THRESHOLD
        proposal_count = self.config.RPN_POST_NMS_ROIS_INFERENCE
        # rpn_feature_maps: [batch, channels, height, width]
        # mrcnn_feature_maps: [batch, channels, height, width]
        # rpn_bbox: [batch, anchors, 4]
        # rpn_rois: [batch, n_rois_after_nms, 4]
        # n_rois_per_sample: [batch]
        #
        # n_rois_after_nms is the maximum number of ROIs. Zero padding used for samples with
        # less ROIs passing NMS
        # n_rois_per_sample gives the number of valid ROIs in each sample
        rpn_feature_maps, mrcnn_feature_maps, _, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample = \
            self._feature_maps_rpn_preds_and_roi(images, pre_nms_limit, nms_threshold, proposal_count)

        return rpn_feature_maps, mrcnn_feature_maps, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample




class AltForwardWrapper (nn.Module):
    def __init__(self, instance, method):
        super(AltForwardWrapper, self).__init__()

        self._instance = instance
        self._method = method

    def forward(self, *args, **kwargs):
        return self._method(self._instance, *args, **kwargs)


class alt_forward_method (object):
    def __init__(self, method):
        self._method = method

    def __get__(self, instance, owner):
        return AltForwardWrapper(instance, self._method)



class AbstractRPNNModel (RPNBaseModel):
    """
    Abstract RPN model

    Adds training and detection forward passes to RPNBaseModel
    """
    def train_forward(self, images):
        """
        Training forward pass; generates RPN predictions. Compare to targets to compute losses

        :param images: training images

        :return: (rpn_class_logits, rpn_bbox_deltas) where
            rpn_class_logits: [batch, num_anchors, 2] or [batch, num_anchors] (depending on
                `self.config.RPN_OBJECTNESS_FUNCTION`) RPN class logit predictions
            rpn_bbox_deltas: [batch, num_anchors, 4] RPN box delta predictions
        """
        # Get RPN proposals
        rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, _, rpn_bbox_deltas = self._feature_maps_and_rpn_preds_all(
            images)

        return (rpn_class_logits, rpn_bbox_deltas)


    @alt_forward_method
    def train_loss_forward(self, images, rpn_target_match, rpn_target_bbox, rpn_num_pos):
        """
        Training forward pass returning per-sample losses.

        :param images: training images
        :param rpn_target_match: [batch, anchors]. Anchor match type. 1=positive,
                   -1=negative, 0=neutral anchor.
        :param rpn_target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
        :param rpn_num_pos: [batch] number of positives per sample

        :return: (rpn_class_losses, rpn_bbox_losses) where
            rpn_class_losses: [batch] RPN objectness per-sample loss
            rpn_bbox_losses: [batch] RPN box delta per-sample loss
        """

        # Convert rpn_num_pos to a list
        rpn_num_pos = torch_tensor_to_int_list(rpn_num_pos)

        # Get RPN proposals
        rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, _, rpn_bbox = self._feature_maps_and_rpn_preds_all(
            images)

        rpn_class_losses, rpn_bbox_losses = compute_rpn_losses_per_sample(self.config, rpn_class_logits, rpn_bbox,
                                                                          rpn_target_match, rpn_target_bbox,
                                                                          rpn_num_pos)

        return (rpn_class_losses, rpn_bbox_losses)


    def detect_forward(self, images):
        """Runs the detection pipeline and returns the results as torch tensors

        :param images: Tensor of images

        :return: (detections, n_rois_per_sample) where
            detections: `RPNDetections` named tuple that has the following attributes:
                boxes: [batch, n_rois_after_nms, 4] detection boxes; dim 1 may be zero-padded
                scores: [batch, n_rois_after_nms] detection confidence scores; dim 1 may be zero-padded
            n_dets_per_sample: [batch] number of rois per sample in the batch
        """
        device = images.device

        image_size = images.shape[2:]

        h, w = image_size
        scale = torch.tensor([h, w, h, w], dtype=torch.float, device=device)

        _, _, _, det_boxes_nrm, det_scores, n_dets_per_sample = self.rpn_detect_forward(images)

        det_boxes = det_boxes_nrm * scale[None, None, :]

        return RPNDetections(boxes=det_boxes, scores=det_scores), n_dets_per_sample


    def detect_forward_np(self, images):
        """Runs the detection pipeline and returns the results as a list of detection tuples consisting of NumPy arrays

        :param images: Tensor of images

        :return: [detection0, detection1, ... detectionN] List of detections, one per sample, where each
            detection is an `RPNDetections` named tuple that has the following attributes:
                boxes: [1, detections, [y1, x1, y2, x2]] NumPy array
                scores: [1, detections] NumPy array
        """
        t_dets, n_dets_per_sample = self.detect_forward(images)

        det_boxes_np = t_dets.boxes.cpu().numpy()
        det_scores_np = t_dets.scores.cpu().numpy()

        dets = split_detections(n_dets_per_sample, det_boxes_np, det_scores_np)
        return [RPNDetections(boxes=d[0], scores=d[1]) for d in dets]
