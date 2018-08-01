from collections import namedtuple

RPNDetections = namedtuple('RPNDetections', ['boxes', 'scores'])
RCNNDetections = namedtuple('RCNNDetections', ['boxes', 'class_ids', 'scores'])
MaskRCNNDetections = namedtuple('MaskRCNNDetections', ['boxes', 'class_ids', 'scores', 'mask_boxes', 'masks'])
