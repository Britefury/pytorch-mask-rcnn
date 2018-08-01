import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../Detectron.pytorch/lib')

import pickle

import coco_utils
import coco_model
import coco
import torch
import visualize

import cv2

from core.test import _get_blobs, im_detect_bbox, box_results_with_nms_and_limit, _add_multilevel_rois_for_test, \
    _get_rois_blob, segm_results
from core.config import cfg, cfg_from_file, assert_and_infer_cfg
from modeling.model_builder import Generalized_RCNN
from utils import detectron_weight_helper, boxes as box_utils
from maskrcnn.model.utils import compute_overlaps, plot_image_with_boxes


CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    MEAN_PIXEL = np.array([[[102.9801, 115.9465, 122.7717]]])
    BN_EPS = 0.0
    RPN_PRE_NMS_LIMIT_TRAIN = 2000
    RPN_PRE_NMS_LIMIT_TEST = 1000
    RPN_POST_NMS_ROIS_INFERENCE = 1000
    CENTRE_ANCHORS = True
    ANCHORS_DETECTRON = True

    RPN_BBOX_USE_STD_DEV = False
    RCNN_BBOX_USE_STD_DEV = True

    ROI_ALIGN_FUNCTION = 'roi_align'
    ROI_ALIGN_SAMPLING_RATIO = 2

    RCNN_DETECTION_MIN_CONFIDENCE = 0.5
    RCNN_DETECTION_NMS_THRESHOLD = 0.5

    RPN_FILTER_PROPOSALS_BY_LEVEL = True

    RCNN_MLP2 = True

    MASK_BATCH_NORM = False

    TORCH_PADDING = True

config = InferenceConfig()


def detectron_full_masks(det_boxes, masks, scores, image_shape, threshold=0.5):
    """
    :param det_boxes: list of boxes per class
    :param masks: [detection, class, H, W]
    :param image_shape: (H, W)
    :return: (det_vis_boxes, det_vis_classes, det_vis_scores, det_vis_full_masks)
    """
    det_vis_boxes = []
    det_vis_classes = []
    det_vis_scores = []
    det_vis_full_masks = []
    det_i = 0
    for cls_i, cls_boxes in enumerate(det_boxes):
        if cls_i > 0:
            for i in range(len(cls_boxes)):
                if scores[det_i] > threshold:
                    # Convert neural network mask to full size mask
                    full_mask = coco_utils.unmold_mask(masks[det_i, cls_i], cls_boxes[i], image_shape)
                    det_vis_boxes.append(cls_boxes[i])
                    det_vis_classes.append(cls_i)
                    det_vis_scores.append(scores[det_i])
                    det_vis_full_masks.append(full_mask)

                det_i += 1
        else:
            det_i += len(cls_boxes)
    det_vis_boxes = np.stack(det_vis_boxes, axis=0)
    det_vis_classes = np.array(det_vis_classes)
    det_vis_scores = np.array(det_vis_scores)
    det_vis_full_masks = np.stack(det_vis_full_masks, axis=-1)
    return (det_vis_boxes, det_vis_classes, det_vis_scores, det_vis_full_masks)


def build_full_masks(det_boxes, masks, image_shape):
    full_masks = []
    for i in range(len(det_boxes)):
        # Convert neural network mask to full size mask
        full_mask = coco_utils.unmold_mask(masks[i], det_boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)
    return full_masks


def load_detectron_weight(net, detectron_weight_file):
    name_mapping, orphan_in_detectron = net.detectron_weight_mapping()

    # print(name_mapping)

    with open(detectron_weight_file, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')
    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']

    params = net.state_dict()
    for p_name, p_tensor in params.items():
        d_name = name_mapping[p_name]
        if isinstance(d_name, str):  # maybe str, None or True
            val = src_blobs[d_name]
        elif isinstance(d_name, np.ndarray):
            val = d_name
        elif callable(d_name):
            val = d_name(p_tensor.size(), src_blobs)
        assert val.shape == p_tensor.size()[:]
        p_tensor.copy_(torch.from_numpy(val))


DETECTRON_CONFIG_PATH = '../Detectron.pytorch/configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml'
DETECTRON_WEIGHTS_PATH = '../Detectron.pytorch/detectron_weights.pkl'
# IMAGE_PATH = 'images/12283150_12d37e6389_z.jpg'
IMAGE_PATH = 'images/3132016470_c27baa00e8_z.jpg'

print('Building baseline detectron model...')
# Detectron.pytorch model
cfg_from_file(DETECTRON_CONFIG_PATH)
cfg.MODEL.NUM_CLASSES = 81
assert_and_infer_cfg()
det_maskRCNN = Generalized_RCNN()
detectron_weight_helper.load_detectron_weight(det_maskRCNN, DETECTRON_WEIGHTS_PATH)
det_maskRCNN.cuda()
det_maskRCNN.eval()

print('Building pytorch-mask-rcnn model...')
net = coco_model.CocoMaskRCNN(config, model_dir='logs', bias=False, torchpad=True, architecture='resnet50')
load_detectron_weight(net, DETECTRON_WEIGHTS_PATH)
net.cuda()
net.eval()


print('Loading image...')
im = cv2.imread(IMAGE_PATH)
inputs, im_scale = _get_blobs(im, rois=None, target_scale=600, target_max_size=1000)

print('im.shape={}, im_scale={}'.format(im.shape, im_scale))
print(inputs['im_info'])
print(inputs['data'].shape)
im_h, im_w = inputs['data'].shape[2:]

def compare_torch_values(a, b):
    delta = a.cpu().numpy() - b.cpu().numpy()
    return 'shape={}, diff range={}:{}'.format(delta.shape, abs(delta).min(), abs(delta).max())

def compare_np_values(a, b):
    delta = a - b
    return 'shape={}, diff range={}:{}'.format(delta.shape, abs(delta).min(), abs(delta).max())

with torch.no_grad():
    x_var = torch.from_numpy(inputs['data']).cuda()
    info_var = torch.from_numpy(inputs['im_info']).cuda()
    x_windows_var = torch.tensor([[0.0, 0.0, im_h, im_w]]).float().cuda()

    # det_out = det_maskRCNN(x_var, info_var)
    #
    # print(det_out.keys())
    #
    # blob_conv = det_out['blob_conv']
    # for blob_i, blob in enumerate(blob_conv):
    #     print('blob_conv {}: {}'.format(blob_i, blob.size()))
    print('======================================')
    print('Using Detectron model for inference...')
    print('======================================')

    det_fpn_out = det_maskRCNN.Conv_Body(x_var)
    rpn_ret = det_maskRCNN.RPN(det_fpn_out, info_var, None)

    # Convert Detectron RPN ROIs to consistent format
    det_rpn_rois = rpn_ret['rois']
    det_rpn_boxes_yx = det_rpn_rois[:, [2, 1, 4, 3]] + np.array([[0.0, 0.0, 1.0, 1.0]])

    for key in rpn_ret.keys():
        print(key, rpn_ret[key].shape)

    # RCNN pass
    det_rcnn_out = det_maskRCNN(x_var, info_var)

    det_rcnn_box_delta = det_rcnn_out['bbox_pred'].cpu().numpy()
    det_rcnn_cls_prob = det_rcnn_out['cls_score'].cpu().numpy()

    # det_rcnn_box_delta = det_rcnn_box_delta.reshape((-1, det_rcnn_cls.shape[-1], 4))

    # Transform Detectron RCNN predictions to boxes
    det_rcnn_box_pred = box_utils.bbox_transform(det_rpn_rois[:, 1:5] / im_scale, det_rcnn_box_delta, cfg.MODEL.BBOX_REG_WEIGHTS)
    det_rcnn_box_pred = box_utils.clip_tiled_boxes(det_rcnn_box_pred, im.shape)
    det_rcnn_box_pred_yx = det_rcnn_box_pred[:, [1, 0, 3, 2]] + np.array([[0.0, 0.0, 1.0, 1.0]])

    print('det_rcnn_box_delta.shape={}, det_rcnn_cls.shape={}, det_rcnn_box_pred.shape={}'.format(
        det_rcnn_box_delta.shape, det_rcnn_cls_prob.shape, det_rcnn_box_pred.shape))

    det_rcnn_scores, det_rcnn_boxes, det_rcnn_cls_boxes = box_results_with_nms_and_limit(
        det_rcnn_cls_prob, det_rcnn_box_pred)
    # detrcnn_cls_boxes[0] = np.zeros((0, 5))
    # detrcnn_cls_scores = [boxes[:, 4] for boxes in detrcnn_cls_boxes]
    # detrcnn_cls_boxes = [boxes[:, :4] for boxes in detrcnn_cls_boxes]


    # Mask prediction
    mask_inputs = {'mask_rois': _get_rois_blob(det_rcnn_boxes, im_scale)}
    _add_multilevel_rois_for_test(mask_inputs, 'mask_rois')
    det_mask = det_maskRCNN.mask_net(det_rcnn_out['blob_conv'], mask_inputs)
    det_mask_np = det_mask.cpu().numpy()

    print('det_mask.shape={}'.format(det_mask.shape))

    det_rcnn_boxes_im = det_rcnn_boxes * im_scale

    det_rcnn_boxes_im_yx = det_rcnn_boxes_im[:, [1, 0, 3, 2]] + np.array([[0.0, 0.0, 1.0, 1.0]])
    det_rcnn_cls_boxes_yx = [
        (boxes[:, [1, 0, 3, 2]] + np.array([[0.0, 0.0, 1.0, 1.0]])) if len(boxes) > 0 else np.zeros((0,4))
        for boxes in det_rcnn_cls_boxes
    ]
    det_vis_rcnn_cls_boxes_yx = [
        np.round(boxes).astype(int) for boxes in det_rcnn_cls_boxes_yx
    ]

    print('det_rcnn_scores.shape={}, det_rcnn_boxes.shape={}'.format(
        det_rcnn_scores.shape, det_rcnn_boxes.shape))

    det_vis_boxes, det_vis_classes, det_vis_scores, det_vis_full_masks = detectron_full_masks(
        det_vis_rcnn_cls_boxes_yx, det_mask_np, det_rcnn_scores, im.shape[:2])

    plot_image_with_boxes(im[:,:,::-1], boxes=det_rpn_boxes_yx/im_scale, alpha=0.25)
    plot_image_with_boxes(im[:,:,::-1], boxes=det_rcnn_boxes_im_yx / im_scale, alpha=0.67)
    visualize.display_instances(im[:, :, ::-1], det_vis_boxes,
                                det_vis_full_masks, det_vis_classes, CLASS_NAMES, scores=det_vis_scores)



    print('========================================')
    print('Using pytorch-mask-rcnn for inference...')
    print('========================================')
    # Scale factor for converting to and from normalized box co-ordinates
    scale = np.array([inputs['data'].shape[2], inputs['data'].shape[3], inputs['data'].shape[2], inputs['data'].shape[3]])

    rpn_feature_maps, mrcnn_feature_maps, rpn_class_logits_by_lvl, rpn_bbox_by_lvl, rpn_rois, roi_scores, n_rois_per_sample = \
        net._feature_maps_rpn_preds_and_roi_by_level(x_var, config.RPN_PRE_NMS_LIMIT_TEST, net.config.RPN_POST_NMS_ROIS_INFERENCE)

    for lvl_i, (rpn_class_logits, rpn_bbox) in enumerate(zip(rpn_class_logits_by_lvl, rpn_bbox_by_lvl)):
        print('lvl {} rpn_class_logits.shape={}'.format(lvl_i, rpn_class_logits.shape))
        print('lvl {} rpn_bbox.shape={}'.format(lvl_i, rpn_bbox.shape))
    print('n_rois_per_sample={}'.format(n_rois_per_sample))
    print('rpn_rois.shape={}'.format(rpn_rois.shape))
    print('roi_scores.shape={}'.format(roi_scores.shape))

    # rpn_rois are in normalized co-ordinates
    rpn_boxes = rpn_rois[0].cpu().numpy() * scale

    [det] = net.detect_forward_np(
        x_var, torch.tensor([[0.0, 0.0, im_h, im_w]]).float().cuda()
    )
    rcnn_boxes_np = det.boxes[0]
    rcnn_class_ids_np = det.class_ids[0]
    rcnn_scores_np = det.scores[0]
    mrcnn_mask = det.masks[0]
    print('rcnn_boxes_np.shape={}, rcnn_class_ids_np.shape={}, rcnn_scores_np.shape={}, mrcnn_mask.shape={}'.format(
        rcnn_boxes_np.shape, rcnn_class_ids_np.shape, rcnn_scores_np.shape, mrcnn_mask.shape
    ))

    # # # rcnn_boxes, rcnn_class_ids, rcnn_scores, _ = \
    # # #     net.rcnn_detect_forward(x_var, x_windows_var, mrcnn_feature_maps,
    # # #                             torch.from_numpy(det_rpn_boxes_yx / scale).float().cuda()[None, :],
    # # #                             [len(det_rpn_boxes_yx)])
    # #
    # # rcnn_boxes_np = rcnn_boxes.cpu().numpy()[0]
    # # rcnn_class_ids_np = rcnn_class_ids.cpu().numpy()[0]
    # # rcnn_scores_np = rcnn_scores.cpu().numpy()[0]
    # print('rcnn_boxes_np.shape={}, rcnn_class_ids_np.shape={}, rcnn_scores_np.shape={}'.format(
    #     rcnn_boxes_np.shape, rcnn_class_ids_np.shape, rcnn_scores_np.shape
    # ))

    vis_boxes = np.round(rcnn_boxes_np / im_scale).astype(int)
    full_masks = build_full_masks(vis_boxes, mrcnn_mask, im.shape[:2])

    plot_image_with_boxes(im[:,:,::-1], boxes=rpn_boxes/im_scale, alpha=0.25)
    plot_image_with_boxes(im[:,:,::-1], boxes=rcnn_boxes_np/im_scale, alpha=0.67)
    visualize.display_instances(im[:, :, ::-1], vis_boxes,
                                full_masks, rcnn_class_ids_np, CLASS_NAMES, scores=rcnn_scores_np)


    print('============')
    print('Comparing...')
    print('============')

    print('---- FPN')
    for i, (det_fpn_i, rpn_f_i)  in enumerate(zip(det_fpn_out[::-1], rpn_feature_maps)):
        print('FPN Level {}: {}'.format(i, compare_torch_values(det_fpn_i, rpn_f_i)))

    print('---- RPN')
    det_rpn_class_logits = [
        rpn_ret['rpn_cls_logits_fpn2'].cpu().numpy().transpose(0, 2, 3, 1).reshape((1, -1)),
        rpn_ret['rpn_cls_logits_fpn3'].cpu().numpy().transpose(0, 2, 3, 1).reshape((1, -1)),
        rpn_ret['rpn_cls_logits_fpn4'].cpu().numpy().transpose(0, 2, 3, 1).reshape((1, -1)),
        rpn_ret['rpn_cls_logits_fpn5'].cpu().numpy().transpose(0, 2, 3, 1).reshape((1, -1)),
        rpn_ret['rpn_cls_logits_fpn6'].cpu().numpy().transpose(0, 2, 3, 1).reshape((1, -1)),
    ]
    # det_rpn_class_logits = np.concatenate(det_rpn_class_logits, axis=1)

    for i, (det_fpn_i, rpn_class_logits)  in enumerate(zip(det_rpn_class_logits, rpn_class_logits_by_lvl)):
        print('RPN class logits lvl {}: {}'.format(i, compare_np_values(det_fpn_i, rpn_class_logits.cpu().numpy())))

    print('det_rpn_rois.shape={}, rpn_rois.shape={}'.format(det_rpn_rois.shape, rpn_rois.shape))

    print('RPN boxes: {}'.format(compare_np_values(det_rpn_boxes_yx[:32], rpn_boxes[:32])))

    rpn_iou = compute_overlaps(det_rpn_boxes_yx, rpn_boxes)
    for thresh in np.arange(0.75, 1.0, 0.05):
        print('Overlaps > {} between RPN boxes from Detectron and pytorch-mask-rcnn: {}/{}'.format(
            thresh, (rpn_iou > thresh).sum(),min(rpn_iou.shape)))


    print('---- RCNN')

    rcnn_iou = compute_overlaps(det_rcnn_boxes_im_yx, rcnn_boxes_np)
    for thresh in np.arange(0.75, 1.0, 0.05):
        print('Overlaps > {} between RCNN boxes from Detectron and pytorch-mask-rcnn: {}/{}'.format(
            thresh, (rcnn_iou > thresh).sum(),min(rcnn_iou.shape)))


