import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn.nms.nms_wrapper import nms
from .utils import SamePad2d, compute_overlaps, concatenate_detections, split_detections

############################################################
#  Region Proposal Network
############################################################

class RPNHead (nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
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
        if config.RPN_OBJECTNESS_SOFTMAX:
            self.conv_class = nn.Conv2d(config.RPN_HIDDEN_CHANNELS, 2 * anchors_per_location, kernel_size=1, stride=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.conv_class = nn.Conv2d(config.RPN_HIDDEN_CHANNELS, anchors_per_location, kernel_size=1, stride=1)
            self.softmax = None
        self.conv_bbox = nn.Conv2d(config.RPN_HIDDEN_CHANNELS, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        if self.padding is not None:
            x = self.padding(x)
        x = self.relu(self.conv_shared(x))

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
        rpn_class_logits = rpn_class_logits.contiguous()

        if self.softmax is not None:
            rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)
            # Softmax on last dimension of BG/FG.
            rpn_probs = self.softmax(rpn_class_logits)
        else:
            rpn_class_logits = rpn_class_logits.view(x.size()[0], -1)
            # Softmax on last dimension of BG/FG.
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
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def proposal_layer_one_sample(rpn_pred_probs, rpn_box_deltas, proposal_count, nms_threshold, anchors,
                              image_size, pre_nms_limit, config):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    :param rpn_pred_probs: predicted background/foreground probabilities from region proposal network RPN
        Either:
            (anchors, 2) array  where the last dimension is [bg_prob, fg_prob] if
            `config.RPN_OBJECTNESS_SOFTMAX`
        Or:
            (anchors,) array  where the last dimension is [fg_prob]
    :param rpn_box_deltas: predicted bounding box deltas from region proposal network RPN
        (anchors, 4) array  where the last dimension is [dy, dx, log(dh), log(dw)]
    :param proposal_count: maximum number of proposals to be generated
    :param nms_threshold: Non-maximum suppression threshold
    :param anchors: Anchors as a (A,4) Torch tenspr
    :param image_size: image size as a (height, width) tuple
    :param pre_nms_limit: number of pre-NMS boxes to consider
    :param config:
    :return: (normalized_boxes, scores):
        normalized_boxes: proposals in normalized coordinates [rois, (y1, x1, y2, x2)] (Torch tensor)
        scores: objectness scores [rois] (Torch tensor)
    """
    device = rpn_pred_probs.device

    # Select scores and deltas corresponding to valid anchors
    if config.RPN_OBJECTNESS_SOFTMAX:
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = rpn_pred_probs[:, 1]
    else:
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = rpn_pred_probs

    # Box deltas [batch, num_rois, 4]
    if config.RPN_BBOX_USE_STD_DEV:
        std_dev = torch.tensor(np.reshape(config.BBOX_STD_DEV, [1, 4]), requires_grad=False, device=device, dtype=torch.float)
        rpn_box_deltas = rpn_box_deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(pre_nms_limit, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    rpn_box_deltas = rpn_box_deltas[order, :] # TODO: Support batch size > 1 ff.
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
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).detach(), nms_threshold)
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]
    scores = scores[keep]

    # Normalize dimensions to range of 0 to 1.
    norm = torch.tensor(np.array([height, width, height, width]), dtype=torch.float, device=device)
    normalized_boxes = boxes / norm

    return normalized_boxes, scores


def proposal_layer(rpn_pred_probs, rpn_box_deltas, proposal_count, nms_threshold, anchors,
                   image_size, pre_nms_limit, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    :param rpn_probs: predicted background/foreground probabilities from region proposal network RPN
        (batch, anchors, 2) array  where the last dimension is [bg_prob, fg_prob]
    :param rpn_box_deltas: predicted bounding box deltas from region proposal network RPN
        (batch, anchors, 4) array  where the last dimension is [dy, dx, log(dh), log(dw)]
    :param proposal_count: maximum number of proposals to be generated
    :param nms_threshold: Non-maximum suppression threshold
    :param anchors: Anchors as a (A,4) Torch tenspr
    :param image_size: image size as a (height, width) tuple
    :param pre_nms_limit: number of pre-NMS boxes to consider
    :param config:
    :return: (normalized_boxes, scores, n_boxes_per_sample) where
        normalized_boxes: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
        scores: proposal objectness scores [batch, rois] (Torch tensor)
        n_boxes_per_sample: list giving number of ROIs in each sample in the batch
    """
    normalized_boxes = []
    scores = []
    for sample_i in range(rpn_pred_probs.size()[0]):
        norm_boxes, sample_scores = proposal_layer_one_sample(rpn_pred_probs[sample_i], rpn_box_deltas[sample_i],
                                                              proposal_count, nms_threshold,
                                                              anchors, image_size, pre_nms_limit, config)
        normalized_boxes.append(norm_boxes.unsqueeze(0))
        scores.append(sample_scores.unsqueeze(0))

    (normalized_boxes, scores), n_boxes_per_sample = concatenate_detections(normalized_boxes, scores)

    return normalized_boxes, scores, n_boxes_per_sample


def proposal_layer_by_level(rpn_pred_probs_by_lvl, rpn_box_deltas_by_lvl, proposal_count, nms_threshold, anchors_by_lvl,
                            image_size, pre_nms_limit, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Processes FPN levels separately

    :param rpn_probs: predicted background/foreground probabilities from region proposal network RPN
        (batch, anchors, 2) array  where the last dimension is [bg_prob, fg_prob]
    :param rpn_box_deltas: predicted bounding box deltas from region proposal network RPN
        (batch, anchors, 4) array  where the last dimension is [dy, dx, log(dh), log(dw)]
    :param proposal_count: maximum number of proposals to be generated
    :param nms_threshold: Non-maximum suppression threshold
    :param anchors: Anchors as a (A,4) Torch tenspr
    :param image_size: image size as a (height, width) tuple
    :param pre_nms_limit: number of pre-NMS boxes to consider
    :param config:
    :return: (normalized_boxes, scores, n_boxes_per_sample) where
        normalized_boxes: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
        scores: proposal objectness scores [batch, rois] (Torch tensor)
        n_boxes_per_sample: list giving number of ROIs in each sample in the batch
    """
    normalized_boxes = []
    scores = []
    for sample_i in range(rpn_pred_probs_by_lvl[0].size()[0]):
        sample_nrm_boxes = []
        sample_scores = []
        for lvl_i, (rpn_pred_probs, rpn_box_deltas, anchors) in enumerate(zip(
                rpn_pred_probs_by_lvl, rpn_box_deltas_by_lvl, anchors_by_lvl)):
            lvl_boxes, lvl_scores = proposal_layer_one_sample(rpn_pred_probs[sample_i], rpn_box_deltas[sample_i],
                                                              proposal_count, nms_threshold,
                                                              anchors, image_size, pre_nms_limit, config)
            sample_nrm_boxes.append(lvl_boxes)
            sample_scores.append(lvl_scores)
        sample_nrm_boxes = torch.cat(sample_nrm_boxes, 0)
        sample_scores = torch.cat(sample_scores, 0)

        sample_scores, order = sample_scores.sort(descending=True)
        order = order[:proposal_count]
        sample_scores = sample_scores[:proposal_count]
        sample_nrm_boxes = sample_nrm_boxes[order, :]

        normalized_boxes.append(sample_nrm_boxes.unsqueeze(0))
        scores.append(sample_scores.unsqueeze(0))

    (normalized_boxes, scores), n_boxes_per_sample = concatenate_detections(normalized_boxes, scores)

    return normalized_boxes, scores, n_boxes_per_sample


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(config, rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    if len(rpn_match.size()) == 3:
        # Squeeze last dim to simplify
        rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    if config.RPN_OBJECTNESS_SOFTMAX:
        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = rpn_class_logits[indices[:, 0], indices[:, 1], :]
        anchor_class = anchor_class[indices[:, 0], indices[:, 1]]

        # Crossentropy loss
        loss = F.cross_entropy(rpn_class_logits, anchor_class)
    else:
        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = rpn_class_logits[indices[:, 0], indices[:, 1]]
        anchor_class = anchor_class[indices[:, 0], indices[:, 1]]

        # loss = F.binary_cross_entropy_with_logits(
        #     rpn_class_logits, rpn_class_logits.float(), weight, size_average=False)
        loss = F.binary_cross_entropy_with_logits(
            rpn_class_logits, anchor_class.float())

    return loss

def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox, rpn_num_pos_per_sample):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    rpn_num_pos_per_sample: [batch] number of positives per sample
    """

    if len(rpn_match.size()) == 3:
        # Squeeze last dim to simplify
        rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices[:,0],indices[:,1]]

    pos_target_box = []
    for sample_i, n_pos in enumerate(rpn_num_pos_per_sample):
        if n_pos > 0:
            pos_target_box.append(target_bbox[sample_i, :n_pos, :])
    pos_target_box = torch.cat(pos_target_box, dim=0)

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, pos_target_box)

    return loss



def compute_rpn_losses(config, rpn_match, rpn_bbox, rpn_num_pos_per_sample, rpn_class_logits, rpn_pred_bbox):

    rpn_class_loss = compute_rpn_class_loss(config, rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox, rpn_num_pos_per_sample)

    return [rpn_class_loss, rpn_bbox_loss]


############################################################
#  Target generation
############################################################

def build_rpn_targets(image_shape, anchors, valid_anchors_mask, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    valid_anchors_mask: [num_anchors]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    num_positives: int; number of positives
    """

    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    valid_anchors = anchors[valid_anchors_mask]
    rpn_match = np.zeros([len(valid_anchors)], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    if len(gt_boxes) == 0:
        # Special case
        rpn_match_all = np.zeros([len(valid_anchors_mask)], dtype=np.int32)
        rpn_match_all.fill(-1)
        return rpn_match_all, rpn_bbox, 0

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
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, valid_anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        if config.RPN_BBOX_USE_STD_DEV:
            rpn_bbox[ix] /= config.BBOX_STD_DEV
        ix += 1

    # Reverse valid anchor selection
    rpn_match_all = np.zeros([len(valid_anchors_mask)], dtype=np.int32)

    rpn_match_all[valid_anchors_mask] = rpn_match

    return rpn_match_all, rpn_bbox, np.count_nonzero(rpn_match == 1)




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
        anchors, valid_mask = self.config.ANCHOR_CACHE.get_anchors_and_valid_masks_for_image_shape(image_shape)
        rpn_match, rpn_bbox, num_positives = build_rpn_targets(image_shape, anchors, valid_mask,
                                                               gt_class_ids, gt_boxes, self.config)
        return rpn_match, rpn_bbox, num_positives


    def _feature_maps_and_proposals_all(self, molded_images):
        # Feature extraction
        rpn_feature_maps, mrcnn_feature_maps = self.fpn(molded_images)

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


    def _feature_maps_proposals_and_roi_all(self, molded_images, pre_nms_limit, nms_threshold, proposal_count):
        """
        :param molded_images: images to process
        :param pre_nms_limit: number of proposals to pass prior to NMS
        :param nms_threshold: the NMS threshold to use
        :param proposal_count: number of proposals to pass subsequent to NMS
        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, n_rois_per_sample) where
            rpn_feature_maps: Feature maps for RPN
            rcnn_feature_maps: Feature maps for RCNN heads
            rpn_rois: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
            roi_scores: proposal objectness scores [batch, rois] (Torch tensor)
            n_rois_per_sample: list giving number of ROIs in each sample in the batch
        """
        device = molded_images.device

        rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, rpn_class_probs, rpn_bbox_deltas = \
            self._feature_maps_and_proposals_all(molded_images)

        image_size = tuple(molded_images.size()[2:])

        # Get anchors
        anchs_var = self.config.ANCHOR_CACHE.get_anchors_var_for_image_shape(image_size, device)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        rpn_rois, roi_scores, n_rois_per_sample = proposal_layer(
            rpn_class_probs, rpn_bbox_deltas, proposal_count=proposal_count,
            nms_threshold=nms_threshold,
            anchors=anchs_var, image_size=image_size,
            pre_nms_limit=pre_nms_limit, config=self.config)

        return rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample


    def _feature_maps_and_proposals_by_level(self, molded_images):
        # Feature extraction
        rpn_feature_maps, mrcnn_feature_maps = self.fpn(molded_images)

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


    def _feature_maps_proposals_and_roi_by_level(self, molded_images, pre_nms_limit, nms_threshold, proposal_count):
        """
        :param molded_images: images to process
        :param pre_nms_limit: number of proposals to pass prior to NMS
        :param nms_threshold: the NMS threshold to use
        :param proposal_count: number of proposals to pass subsequent to NMS
        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, n_rois_per_sample) where
            rpn_feature_maps: Feature maps for RPN
            rcnn_feature_maps: Feature maps for RCNN heads
            rpn_rois: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
            roi_scores: proposal objectness scores [batch, rois] (Torch tensor)
            n_rois_per_sample: list giving number of ROIs in each sample in the batch
        """
        device = molded_images.device

        rpn_feature_maps, rcnn_feature_maps, rpn_class_logits_by_lvl, rpn_class_probs_by_lvl, rpn_bbox_deltas_by_lvl = \
            self._feature_maps_and_proposals_by_level(molded_images)

        image_size = tuple(molded_images.size()[2:])

        # Get anchors
        anchs_vars = self.config.ANCHOR_CACHE.get_anchors_var_for_image_shape_by_level(image_size, device)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        rpn_rois, roi_scores, n_rois_per_sample = proposal_layer_by_level(
            rpn_class_probs_by_lvl, rpn_bbox_deltas_by_lvl, proposal_count=proposal_count,
            nms_threshold=nms_threshold,
            anchors_by_lvl=anchs_vars, image_size=image_size,
            pre_nms_limit=pre_nms_limit, config=self.config)

        return rpn_feature_maps, rcnn_feature_maps, rpn_class_logits_by_lvl, rpn_bbox_deltas_by_lvl, \
               rpn_rois, roi_scores, n_rois_per_sample


    def _feature_maps_proposals_and_roi(self, molded_images, pre_nms_limit, nms_threshold, proposal_count):
        """
        :param molded_images:
        :param proposal_count:
        :return: (rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, n_rois_per_sample) where
            rpn_feature_maps: Feature maps for RPN
            rcnn_feature_maps: Feature maps for RCNN heads
            rpn_rois: proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (Torch tensor)
            roi_scores: proposal objectness scores [batch, rois] (Torch tensor)
            n_rois_per_sample: list giving number of ROIs in each sample in the batch
        """
        if self.config.RPN_FILTER_PROPOSALS_BY_LEVEL:
            rpn_feature_maps, rcnn_feature_maps, rpn_class_logits_by_lvl, rpn_bbox_deltas_by_lvl, \
                rpn_rois, roi_scores, n_rois_per_sample = self._feature_maps_proposals_and_roi_by_level(
                    molded_images, pre_nms_limit, nms_threshold, proposal_count)
            rpn_class_logits = torch.cat(rpn_class_logits_by_lvl, 1)
            rpn_bbox_deltas = torch.cat(rpn_bbox_deltas_by_lvl, 1)
            return rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample
        else:
            return self._feature_maps_proposals_and_roi_all(molded_images, pre_nms_limit, nms_threshold, proposal_count)


    def rpn_detect_forward(self, images):
        """Runs the RPN stage of the detection pipeline

        images: Tensor of images

        Returns:
        # rpn_feature_maps: [batch, channels, height, width]
        # mrcnn_feature_maps: [batch, channels, height, width]
        # rpn_bbox: [batch, anchors, 4]
        # rpn_rois: [batch, n_rois_after_nms, 4]
        # roi_scores: [batch, n_rois_after_nms]
        # n_rois_per_sample: [batch]
        """
        #
        # Run object detection
        #

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
            self._feature_maps_proposals_and_roi(images, pre_nms_limit, nms_threshold, proposal_count)

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
    @alt_forward_method
    def train_forward(self, molded_images):
        # Get RPN proposals
        rpn_feature_maps, rcnn_feature_maps, rpn_class_logits, _, rpn_bbox = self._feature_maps_and_proposals_all(
            molded_images)

        return [rpn_class_logits, rpn_bbox]


    @alt_forward_method
    def detect_forward(self, images):
        """Runs the detection pipeline.

        images: Tensor of images
        image_windows: tensor of image windows where each row is [y1, x1, y2, x2]
        override_class: int or None; override class ID to always be this class

        Returns: [detection0, detection1, ... detectionN]
        List of detections, one per sample, where each detection is a tuple of:
        (rois, roi_scores) where:
            rois: [1, detections, [y1, x1, y2, x2]]
            roi_scores: [1, detections]
        """
        device = images.device

        image_size = images.shape[2:]

        h, w = image_size
        scale = torch.tensor(np.array([h, w, h, w]), device=device).float()

        rpn_feature_maps, mrcnn_feature_maps, rpn_bbox_deltas, rpn_rois, roi_scores, n_rois_per_sample = self.rpn_detect_forward(
            images)

        rpn_rois = rpn_rois * scale[None, None, :]

        rpn_rois_np = rpn_rois.cpu().numpy()
        roi_scores_np = roi_scores.cpu().numpy()

        return split_detections(n_rois_per_sample, rpn_rois_np, roi_scores_np)
