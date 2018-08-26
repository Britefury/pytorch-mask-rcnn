"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np
import os
from maskrcnn.model import anchors


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    #
    #
    # OVERALL ARCHITECTURE
    #
    #

    # If True, use the padding built into PyTorch convolutional layers
    # else use SamePad2D
    TORCH_PADDING = True

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Epsilon used in Batch-Norm layers
    BN_EPS = 1.0e-5

    #
    #
    # ANCHORS
    #
    #

    # Align anchor box centres with feature map pixel centres
    CENTRE_ANCHORS = True

    # If True, round *up* the number of feature cells
    ANCHORS_ROUND_UP_FEATURE_SHAPE = True

    # If True, generate Detectron compatible anchor boxes
    ANCHORS_DETECTRON = True

    #
    #
    # RPN
    #
    #

    # Number of hidden channels in RPN
    RPN_HIDDEN_CHANNELS = 256

    # Use batch-norm in RPN
    RPN_BATCH_NORM = False

    # RPN Objectness function
    # 'sigmoid' - single logit, sigmoid non-linearity, binary cross-entropy loss
    # 'softmax' - 2 logits, softmax non-linearity, categorical cross-entropy loss
    # 'focal' - 2 logits, softmax non-linearity, focal loss
    RPN_OBJECTNESS_FUNCTION = 'sigmoid'

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Pre-NMS propsal count limit
    RPN_PRE_NMS_LIMIT_TRAIN = 6000
    RPN_PRE_NMS_LIMIT_TEST = 6000

    # Non-max suppression threshold to filter RPN proposals.
    RPN_NMS_THRESHOLD = 0.7

    # ROIs kept after non-maximum supression (training and inference)
    RPN_POST_NMS_ROIS_TRAINING = 2048
    RPN_POST_NMS_ROIS_INFERENCE = 1024

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Focal loss hyper-parameters
    RPN_FOCAL_LOSS_EXPONENT = 2.0
    RPN_FOCAL_LOSS_POS_CLS_WEIGHT = 0.25

    # If True, filter RPN proposals (NMS, etc.) separately by level
    RPN_FILTER_PROPOSALS_BY_LEVEL = False

    # Bounding box refinement standard deviation for RPN
    RPN_BBOX_USE_STD_DEV = True
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    #
    #
    # RoI pooling - controls how pyramid levels are chosen for
    # RCNN and Mask
    #
    #

    # Select a specific level of the pyramid and identify its scale and level
    ROI_CANONICAL_SCALE = 224
    ROI_CANONICAL_LEVEL = 4
    ROI_MIN_PYRAMID_LEVEL = 2
    ROI_MAX_PYRAMID_LEVEL = 5

    # 'crop_and_resize', 'border_aware_crop_and_resize' or 'roi_align'
    ROI_ALIGN_FUNCTION = 'border_aware_crop_and_resize'

    # If ROI_ALIGN_FUNCTION == 'roi_align', this is the sampling ratio
    ROI_ALIGN_SAMPLING_RATIO = 0

    #
    #
    # RCNN
    #
    #

    # If True, use an 2-layer MLP for the RCNN body
    RCNN_MLP2 = False

    # Bounding box refinement standard deviation for RCNN
    # Uses same value as RPN
    RCNN_BBOX_USE_STD_DEV = True

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    RCNN_TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    RCNN_ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    RCNN_POOL_SIZE = 7

    # Max number of final detections
    RCNN_DETECTION_MAX_INSTANCES = 100

    # Maximum number of detections to process at once during inference, to keep memory consumption down
    DETECTION_BLOCK_SIZE_INFERENCE = 1024

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    RCNN_DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    RCNN_DETECTION_NMS_THRESHOLD = 0.3

    #
    #
    # Mask
    #
    #

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # RoI pool size for mask
    MASK_POOL_SIZE = 14

    # Target mask shape
    MASK_SHAPE = [28, 28]

    # Mask box border: Adds a border to the boxes used during mask training and inference
    # The size of the box used is the largest of MASK_BOX_ENLARGE or MASK_BOX_BORDER_MIN

    # MASK_BOX_ENLARGE is the fraction by which the box is enlarged, e.g. 1.2 will add
    # 10% to each edge of the box
    MASK_BOX_ENLARGE = 1.0

    # MASK_BOX_BORDER_MIN will cause a border of at least `MASK_BOX_BORDER_MIN` pixels to be
    # added to each edge
    MASK_BOX_BORDER_MIN = 0.0

    # Dilation used in convolutional layers in mask head
    MASK_CONV_DILATION = 1

    # If True, use batch-norm in mask head
    MASK_BATCH_NORM = True

    def __init__(self):
        # Anchor cache
        self.ANCHOR_CACHE = anchors.AnchorCache(self, max_cached_anchors=32, max_cached_vars=8)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


    @property
    def n_rpn_logits_per_anchor(self):
        if self.RPN_OBJECTNESS_FUNCTION == 'sigmoid':
            return 1
        else:
            return 2