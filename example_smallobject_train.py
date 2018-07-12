import click


@click.command()
@click.option('--dataset', type=click.Choice(['s1_train_test', 's1_train_val', 'stage2', 'ellipses']),
              default='s1_train_val')
@click.option('--backbone', type=click.Choice(['unet5_wn', 'unet5_bn', 'resnet50']),
              default='resnet50', help='backbone architecture')
@click.option('--head', type=click.Choice(['mask_rcnn', 'faster_rcnn', 'rpn']),
              default='mask_rcnn', help='head architecture')
@click.option('--learning_rate', type=float, default=1e-4, help='learning rate (Adam)')
@click.option('--pretrained_lr_factor', type=float, default=0.1, help='Learning rate scale for pretrained params')
@click.option('--fixed_size', type=int, default=256)
@click.option('--mask_size', type=int, default=28, help='Mask size')
@click.option('--rpn_train_anchors_per_image', type=int, default=64, help='RPN: train anchors per image')
@click.option('--detection_max_instances', type=int, default=1024, help='Max detected objects')
@click.option('--pre_nms_limit', type=int, default=6000, help='Pre-NMS detection limit')
@click.option('--rpn_nms_threshold', type=float, default=0.7, help='RPN NMS threshold')
@click.option('--post_nms_rois_training', type=int, default=2048, help='POST NMS # rois in training')
@click.option('--post_nms_rois_inference', type=int, default=1024, help='POST NMS # rois in inference')
@click.option('--train_rois_per_image', type=int, default=100, help='# rios per image during training')
@click.option('--detection_min_confidence', type=float, default=0.7, help='detection minimum confidence')
@click.option('--detection_nms_threshold', type=float, default=0.3, help='detection NMS threshold')
@click.option('--mask_box_enlarge', type=float, default=1.0)
@click.option('--mask_box_border_min', type=float, default=0.0)
@click.option('--mask_nms_thresh', type=float, default=0.9)
@click.option('--rcnn_hard_neg', is_flag=True, default=False, help='Use hard negative mining when training RCNN head')
@click.option('--per_sample_loss', is_flag=True, default=True, help='Use per-sample loss')
@click.option('--focal_loss', is_flag=True, default=False, help='Use focal loss')
@click.option('--gaussian_noise', type=float, default=0.1, help='Gaussian noise quantity')
@click.option('--scale_u_range', type=str, default='0.8:1.25',
              help='aug xform: scale uniform range; lower:upper')
@click.option('--scale_x_range', type=str, default='',
              help='aug xform: scale x range; lower:upper')
@click.option('--scale_y_range', type=str, default='',
              help='aug xform: scale y range; lower:upper')
@click.option('--crop_border', type=int, default=16, help='aug xform: crop border')
@click.option('--affine_std', type=float, default=0.1, help='aug xform: affine std-dev')
@click.option('--rot_range_mag', type=float, default=45.0, help='aug xform: rotation range magnitude')
@click.option('--hflip', default=True, is_flag=True, help='aug xform: enable random horizontal flips')
@click.option('--vflip', default=True, is_flag=True, help='aug xform: enable random horizontal flips')
@click.option('--hvflip', default=True, is_flag=True, help='aug xform: enable random horizontal-vertical flips')
@click.option('--light_scl_std', type=float, default=0.1,
              help='aug; lighting scale standard deviation')
@click.option('--light_off_std', type=float, default=0.1,
              help='aug; lighting offset standard deviation')
@click.option('--img_pad_mode', type=click.Choice(['reflect', 'replicate', 'constant']), default='constant')
@click.option('--standardisation', type=click.Choice(['none', 'zerocentre', 'zeromean', 'std']), default='std')
@click.option('--invert', type=click.Choice(['none', 'random']), default='none',
              help='aug; invert colours')
@click.option('--batch_size', type=int, default=4, help='mini-batch size')
@click.option('--num_epochs', type=int, default=100, help='number of epochs')
@click.option('--seed', type=int, default=0, help='random seed (0 for time-based)')
@click.option('--log_file', type=str, default='', help='log file path (none to disable)')
@click.option('--model_file', type=str, default='', help='path to file to save model to')
@click.option('--plot_dir', type=str, default='', help='plot detections directory name')
@click.option('--predictions_dir', type=str, default='', help='predictions directory name')
@click.option('--prediction_every_n_epochs', type=int, default=-1, help='generate predictions every n epochs')
@click.option('--eval_every_n_epochs', type=int, default=10, help='evaluate predictions every n epochs')
@click.option('--save_model_every_n_epochs', type=int, default=-1, help='save model every n epochs')
@click.option('--hide_progress_bar', is_flag=True, default=True, help='Hide training progress bar')
@click.option('--subsetseed', type=int, default=12345, help='test set random seed (0 for time based)')
@click.option('--exp_classes', type=str, default='9', help='Limit to use only samples in experiments in these classes (comma separated)')
@click.option('--device', type=int, default=0, help='Device')
@click.option('--num_threads', type=int, default=4, help='Number of worker threads')
def experiment(dataset, backbone, head, learning_rate, pretrained_lr_factor,
               fixed_size, mask_size, rpn_train_anchors_per_image, detection_max_instances,
               pre_nms_limit, rpn_nms_threshold, post_nms_rois_training,
               post_nms_rois_inference, train_rois_per_image, detection_min_confidence,
               detection_nms_threshold,
               mask_box_enlarge, mask_box_border_min, mask_nms_thresh, rcnn_hard_neg, per_sample_loss, focal_loss,
               gaussian_noise, scale_u_range, scale_x_range, scale_y_range,
               crop_border, affine_std, rot_range_mag, hflip, vflip, hvflip,
               light_scl_std, light_off_std,
               img_pad_mode, standardisation, invert,
               batch_size, num_epochs, seed,
               log_file, model_file, plot_dir, predictions_dir,
               prediction_every_n_epochs, eval_every_n_epochs, save_model_every_n_epochs,
               hide_progress_bar,
               subsetseed, exp_classes, device, num_threads):
    # Take a copy of the locals dict so that we can log the settings
    settings = locals().copy()

    import os
    import sys
    import pickle
    from examples import cmdline_helpers, logging

    # Convert command line options
    scale_u_range = cmdline_helpers.colon_separated_range(scale_u_range)
    scale_x_range = cmdline_helpers.colon_separated_range(scale_x_range)
    scale_y_range = cmdline_helpers.colon_separated_range(scale_y_range)
    exp_classes = cmdline_helpers.comma_separated_values(exp_classes, int)

    _PAD_MODE_TO_NP = {
        'reflect': 'reflect',
        'replicate': 'edge',
        'constant': 'constant',
    }
    img_pad_mode_np = _PAD_MODE_TO_NP[img_pad_mode]

    # Set up logging
    try:
        log = logging.Logger(log_file, os.path.join('logs', 'log_smallobject_{}_{}.txt'.format(dataset, head)))
    except logging.LogFileAlreadyExists:
        print('Output log file {} already exists'.format(log_file))
        return

    # Set up predictions directory
    if predictions_dir == '':
        predictions_dir = None
        prediction_every_n_epochs = -1

    if predictions_dir is not None:
        if os.path.exists(predictions_dir) and len(exp_classes) == 0:
            print('Output predictions dir {} already exists'.format(predictions_dir))
            return

    import time
    import tqdm
    import math
    import numpy as np
    from batchup import data_source, work_pool, sampling
    from examples.ellipses import ellipses_dataset
    from examples.kaggle_dsbowl2018 import cellnucleus_dataset
    from examples import augmentation, affine_transforms, affine_torch, inference
    from examples import smallobj_network_architectures
    from PIL import Image
    from matplotlib import pyplot as plt
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
    import torch, torch.cuda
    from torch import nn
    from torch.nn import functional as F
    from torch.optim import lr_scheduler
    from maskrcnn.model import mask_rcnn, rcnn, rpn
    from maskrcnn.model.utils import concatenate_detection_arrays, plot_boxes, visualise_labels
    from examples.ground_truths import label_image_to_gt
    import cv2

    if hide_progress_bar:
        progress_bar = None
    else:
        progress_bar = tqdm.tqdm

    if dataset == 's1_train_test':
        d_train = cellnucleus_dataset.Stage1TrainSegDataset()
        d_test = cellnucleus_dataset.Stage1TestSegDataset()

        train_indices = np.arange(len(d_train.X))
        test_indices = np.arange(len(d_test.X))

        eval_every_n_epochs = -1
    elif dataset == 'stage2':
        d_train = cellnucleus_dataset.Stage1TrainTestSegDataset()
        d_test = cellnucleus_dataset.Stage2TestSegDataset()

        train_indices = np.arange(len(d_train.X))
        test_indices = np.arange(len(d_test.X))

        eval_every_n_epochs = -1
    elif dataset == 's1_train_val':
        d_train = d_test = cellnucleus_dataset.Stage1TrainSegDataset()

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=subsetseed)

        if len(exp_classes) > 0:
            exp_classes = set(exp_classes)
            sample_indices = np.array([i for i in range(len(d_train.cls)) if int(d_train.cls[i]) in exp_classes])
            train_indices, test_indices = next(splitter.split(d_train.cls[sample_indices], d_train.cls[sample_indices]))
            train_indices = sample_indices[train_indices]
            test_indices = sample_indices[test_indices]
        else:
            train_indices, test_indices = next(splitter.split(d_train.cls, d_train.cls))
    elif dataset == 'ellipses':
        d_train = ellipses_dataset.EllipsesTrainDataset()
        d_test = ellipses_dataset.EllipsesTestDataset()

        train_indices = np.arange(len(d_train.X))
        test_indices = np.arange(len(d_test.X))


    num_classes = 2


    torch_device = torch.device('cuda', device)

    if num_threads > 0:
        pool = work_pool.WorkerThreadPool(num_threads)
    else:
        pool = None

    n_chn = 0

    print('Loaded data')

    #
    # Mask R-CNN configuration
    #
    config_params = dict(
        use_focal_loss=focal_loss,
        mask_size=mask_size, mask_box_enlarge=mask_box_enlarge, mask_box_border_min=mask_box_border_min,
        rpn_train_anchors_per_image=rpn_train_anchors_per_image,
        detection_max_instances=detection_max_instances,
        pre_nms_limit=pre_nms_limit, rpn_nms_threshold=rpn_nms_threshold, post_nms_rois_training=post_nms_rois_training,
        post_nms_rois_inference=post_nms_rois_inference, train_rois_per_image=train_rois_per_image,
        detection_min_confidence=detection_min_confidence, detection_nms_threshold=detection_nms_threshold,
        num_classes=int(num_classes),
    )

    #
    # Build network
    #

    net = smallobj_network_architectures.build_network(backbone, head, config_params=config_params).cuda()

    BLOCK_SIZE = net.BLOCK_SIZE

    pretrained_params = list(net.pretrained_parameters())
    new_params = list(net.new_parameters())

    optimizer = torch.optim.Adam([
        dict(params=pretrained_params, lr=learning_rate * pretrained_lr_factor),
        dict(params=new_params, lr=learning_rate),
    ], lr=learning_rate)

    print('Built network')


    #
    # Image augmentation and data loading
    #

    aug = augmentation.ImageAugmentation(
        False, False, False, 0.0, affine_std=affine_std, rot_range_mag=math.radians(rot_range_mag),
        light_scl_std=light_scl_std, light_off_std=light_off_std,
        scale_u_range=scale_u_range, scale_x_range=scale_x_range, scale_y_range=scale_y_range)

    # Prepare a single training image
    # Apply standardisation, padding and draw augmentation parameters
    # Convert ground truth labels to ground truths suitable for use by Mask R-CNN
    def _prepare_training_image(x, labels):
        # Flip the RGB values with 50% probability if enabled
        if invert == 'random':
            flip_inv = np.random.binomial(1, 0.5, 1)
        else:
            flip_inv = np.array([0])

        if flip_inv[0] != 0:
            x = 1.0 - x

        # Standardise image
        if standardisation == 'std':
            x = (x - x.mean(axis=(0, 1), keepdims=True)) / (x.std(axis=(0, 1), keepdims=True) + 1.0e-6)
        elif standardisation == 'zeromean':
            x = x - x.mean(axis=(0, 1), keepdims=True)
        elif standardisation == 'zerocentre':
            x = x * 2.0 - 1.0
        elif standardisation == 'none':
            pass
        else:
            raise RuntimeError

        # Crop out a region for training
        pad_axis_0, crop_axis_0, offset_axis_0 = augmentation.random_shift(labels.shape[0], crop_border, fixed_size)
        pad_axis_1, crop_axis_1, offset_axis_1 = augmentation.random_shift(labels.shape[1], crop_border, fixed_size)

        x = np.pad(x, [pad_axis_0, pad_axis_1, (0, 0)], mode='constant')
        labels = np.pad(labels, [pad_axis_0, pad_axis_1], mode='constant')

        x = x[crop_axis_0, crop_axis_1, :]
        labels = labels[crop_axis_0, crop_axis_1]

        # Random flips, if enabled
        if hflip or vflip or hvflip:
            flip_flags = np.random.binomial(1, 0.5, 3)
            if hvflip and flip_flags[2] != 0:
                x = x.transpose(1, 0, 2)
                labels = labels.transpose(1, 0)
            if hflip and flip_flags[0] != 0:
                x = x[:, ::-1, :]
                labels = labels[:, ::-1]
            if vflip and flip_flags[1] != 0:
                x = x[::-1, :, :]
                labels = labels[::-1, :]

        # Apply any padding necessary to ensure that the image is a multiple of BLOCK_SIZE
        image_size = x.shape[:2]
        padded_image_shape = augmentation.round_up_shape(image_size, BLOCK_SIZE)
        img_padding = augmentation.compute_padding(image_size, padded_image_shape)

        x = np.pad(x, img_padding + [(0, 0)], mode=img_pad_mode_np)
        labels = np.pad(labels, img_padding, mode=img_pad_mode_np)

        # Get augmentation parameters
        xf = aug.aug_xforms(1, np.array(x.shape[:2]))
        light_scl, light_off = aug.aug_colour_xforms(1)

        # Apply lighting augmentation
        x = x * light_scl[None, None, :] + light_off[None, None, :]

        # Apply affine augmentation to labels *only*
        # We apply affine transformation to the input image on the GPU using PyTorch grid sampling
        # We must augment the labels on the CPU as we must extract instance masks from the *augmented*
        # label image.
        xf_cv = augmentation.centre_xf(xf, labels.shape[:2])
        labels_aug = cv2.warpAffine(labels, xf_cv[0], labels.shape[:2][::-1], flags=cv2.INTER_NEAREST)

        # Extract instance masks and boxes
        gt_boxes, gt_masks = label_image_to_gt(labels_aug, image_size, mini_mask_shape=net.config.MINI_MASK_SHAPE)

        # plt.figure(figsize=(8, 8))
        # ax = plt.subplot(1, 1, 1)
        # ax.imshow(np.clip(x * 0.1 + 0.5, 0.0, 1.0))
        # plot_boxes(ax, gt_boxes, alpha=0.8)
        # plt.show()

        # plt.figure(figsize=(8,8))
        # plt.imshow(montage2d(gt_masks), cmap='gray', vmin=0, vmax=1)
        # plt.show()

        # Ground truth class IDs; all 1
        gt_class_ids = np.ones((len(gt_boxes),), dtype=int)

        # Convert boxes to RPN targets
        rpn_match, rpn_bbox, num_positives = net.ground_truth_to_rpn_targets(x.shape[:2], gt_class_ids, gt_boxes)

        # anchors, valid_mask = net.config.ANCHOR_CACHE.get_anchors_and_valid_masks_for_image_shape(x.shape[:2])
        # plt.figure(figsize=(8, 8))
        # pos_anchors = anchors[rpn_match == 1, :5]
        # pos_deltas = rpn_bbox[rpn_match == 1, :5]
        # pos_deltas = pos_deltas * net.config.bbox_std_dev
        # pos_boxes = np.zeros_like(pos_anchors)
        # pos_boxes[:, :2] = pos_anchors[:, :2] + pos_anchors[:, 2:4] * pos_deltas[:, :2]
        # pos_boxes[:, 2:4] = pos_anchors[:, 2:4] * np.exp(pos_deltas[:, 2:4])
        # pos_boxes[:, 4:5] = pos_anchors[:, 4:5] + pos_deltas[:, 4:5]
        # ax = plt.subplot(1, 1, 1)
        # ax.imshow(np.clip(x * 0.1 + 0.5, 0.0, 1.0))
        # plot_boxes(ax, pos_boxes, alpha=0.8)
        # plt.show()

        # Convert everything to tensors of the approproate size
        x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)
        rpn_match = rpn_match[None, ...].astype(np.int32)
        rpn_bbox = rpn_bbox[None, ...].astype(np.float32)
        rpn_num_pos = np.array([num_positives], dtype=np.int32)
        gt_class_ids = gt_class_ids[None, ...].astype(np.int32)
        gt_boxes = gt_boxes[None, ...].astype(np.float32)
        gt_masks = gt_masks[None, ...].astype(np.float32)

        return x, rpn_match.astype(np.int32), rpn_bbox, rpn_num_pos, xf, gt_class_ids, gt_boxes, gt_masks

    # Using _prepare_training_image above, prepare a batch of training images
    def _prepare_training_batch(batch_X, batch_y):
        samples = [_prepare_training_image(x, y) for (x, y) in zip(batch_X, batch_y)]
        properties_per_sample = list(zip(*samples))

        properties = []
        n_gts_per_sample = []
        for prop in properties_per_sample[:-3]:
            try:
                properties.append(np.concatenate(prop, axis=0))
            except ValueError:
                print([p.shape for p in prop])
                raise
        for prop in properties_per_sample[-3:]:
            prop, n_gts_per_sample = concatenate_detection_arrays(prop)
            properties.append(prop)
        properties.append(n_gts_per_sample)

        return tuple(properties)

    if head == 'mask_rcnn':
        # Train on a single batch
        # Note that the parameters match those returned by `_prepare_training_image` and `_prepare_training_batch`
        def f_train(X, rpn_match, rpn_bbox, rpn_num_pos, xf, gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample):
            if gt_class_ids.shape[1] > 0 and gt_boxes.shape[1] > 0:
                xf = augmentation.apply_grid_to_image(augmentation.inv_nx2x3(xf), X.shape[2:])

                X_var = torch.tensor(X, dtype=torch.float, device=torch_device)
                rpn_target_match = torch.tensor(rpn_match, dtype=torch.long, device=torch_device)
                rpn_target_bbox = torch.tensor(rpn_bbox, dtype=torch.float, device=torch_device)
                gt_class_ids_var = torch.tensor(gt_class_ids, dtype=torch.long, device=torch_device)
                gt_boxes_var = torch.tensor(gt_boxes, dtype=torch.float, device=torch_device)
                gt_masks_var = torch.tensor(gt_masks, dtype=torch.float, device=torch_device)
                xf_var = torch.tensor(xf, dtype=torch.float, device=torch_device)

                # Zero the gradients
                optimizer.zero_grad()

                # Grid for augmenting X
                grid_X = F.affine_grid(xf_var, X_var.size())
                X_aug = F.grid_sample(X_var, affine_torch.torch_pad_grid(grid_X, img_pad_mode))

                # Add gaussian noise
                if gaussian_noise != 0.0:
                    noise_var = (torch.randn(X_aug.size(), device=torch_device) * gaussian_noise)
                    X_aug = X_aug + noise_var

                if per_sample_loss:
                    # Run object detection and get per-sample loss
                    rpn_num_pos = torch.tensor(rpn_num_pos, dtype=torch.int, device=torch_device)
                    n_gts_per_sample = torch.tensor(n_gts_per_sample, dtype=torch.int, device=torch_device)
                    rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = net.train_loss_forward(
                        X_aug, rpn_target_match, rpn_target_bbox, rpn_num_pos,
                        gt_class_ids_var, gt_boxes_var, gt_masks_var, n_gts_per_sample)

                    rpn_class_loss = rpn_class_loss.mean()
                    rpn_bbox_loss = rpn_bbox_loss.mean()
                    mrcnn_class_loss = mrcnn_class_loss.mean()
                    mrcnn_bbox_loss = mrcnn_bbox_loss.mean()
                    mrcnn_mask_loss = mrcnn_mask_loss.mean()
                else:
                    # Run object detection
                    rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, \
                        target_mask, mrcnn_mask, n_targets_per_sample = net.train_forward(
                            X_aug, gt_class_ids_var, gt_boxes_var, gt_masks_var, n_gts_per_sample,
                            hard_negative_mining=rcnn_hard_neg)

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = \
                        mask_rcnn.compute_maskrcnn_losses(net.config, rpn_class_logits, rpn_pred_bbox, rpn_target_match,
                                                          rpn_target_bbox, rpn_num_pos,
                                                          mrcnn_class_logits, mrcnn_bbox,target_class_ids, target_deltas,
                                                          mrcnn_mask, target_mask)

                loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

                losses = [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
                n_samples = len(X)
                loss_vals = [float(l.detach().cpu() * n_samples) for l in losses]

                loss.backward()

                optimizer.step()

                if np.isnan(loss_vals).any():
                    print('NaN detected!!!')

                return tuple(loss_vals)
            else:
                return 0.0, 0.0, 0.0, 0.0, 0.0

        # Function for performing inference on a single image and convert the Mask R-CNN output to a label image
        def f_pred_single(X, image_size):
            with torch.no_grad():
                X_var = torch.tensor(X, dtype=torch.float, device=torch_device)
                net.eval()

                # y1, x1, y2, x2
                window = np.array([[0.0, 0.0, float(image_size[0]), float(image_size[1])]])
                det_boxes, det_class_ids, det_scores, mask_boxes, mrcnn_mask = net.detect_forward_np(X_var, window)[0]
                labels, cls_map = inference.mrcnn_detections_to_label_image(
                    image_size, det_scores, det_class_ids, mask_boxes, mrcnn_mask, mask_nms_thresh=mask_nms_thresh)
            return det_boxes, det_scores, det_class_ids, mask_boxes, mrcnn_mask, labels, cls_map

    elif head == 'faster_rcnn':
        # Train on a single batch
        # Note that the parameters match those returned by `_prepare_training_image` and `_prepare_training_batch`
        def f_train(X, rpn_match, rpn_bbox, rpn_num_pos, xf, gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample):
            if gt_class_ids.shape[1] > 0 and gt_boxes.shape[1] > 0:
                xf = augmentation.apply_grid_to_image(augmentation.inv_nx2x3(xf), X.shape[2:])

                X_var = torch.tensor(X, dtype=torch.float, device=torch_device)
                rpn_target_match = torch.tensor(rpn_match, dtype=torch.long, device=torch_device)
                rpn_target_bbox = torch.tensor(rpn_bbox, dtype=torch.float, device=torch_device)
                gt_class_ids_var = torch.tensor(gt_class_ids, dtype=torch.long, device=torch_device)
                gt_boxes_var = torch.tensor(gt_boxes, dtype=torch.float, device=torch_device)
                xf_var = torch.tensor(xf, dtype=torch.float, device=torch_device)

                # Zero the gradients
                optimizer.zero_grad()

                # Grid for augmenting X
                grid_X = F.affine_grid(xf_var, X_var.size())
                X_aug = F.grid_sample(X_var, affine_torch.torch_pad_grid(grid_X, img_pad_mode))

                # Add gaussian noise
                if gaussian_noise != 0.0:
                    noise_var = (torch.randn(X_aug.size(), device=torch_device) * gaussian_noise)
                    X_aug = X_aug + noise_var

                if per_sample_loss:
                    # Run object detection and get per-sample loss
                    rpn_num_pos = torch.tensor(rpn_num_pos, dtype=torch.int, device=torch_device)
                    n_gts_per_sample = torch.tensor(n_gts_per_sample, dtype=torch.int, device=torch_device)
                    rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = net.train_loss_forward(
                        X_aug, rpn_target_match, rpn_target_bbox, rpn_num_pos,
                        gt_class_ids_var, gt_boxes_var, n_gts_per_sample)

                    rpn_class_loss = rpn_class_loss.mean()
                    rpn_bbox_loss = rpn_bbox_loss.mean()
                    rcnn_class_loss = rcnn_class_loss.mean()
                    rcnn_bbox_loss = rcnn_bbox_loss.mean()
                else:
                    # Run object detection
                    rpn_pred_logits, rpn_pred_bbox, tgt_cls_ids, rcnn_pred_logits, tgt_deltas, rcnn_pred_deltas, \
                            n_targets_per_sample = net.train_forward(
                        X_aug, gt_class_ids_var, gt_boxes_var, n_gts_per_sample,
                        hard_negative_mining=rcnn_hard_neg)

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
                        rcnn.compute_faster_rcnn_losses(net.config, rpn_pred_logits, rpn_pred_bbox, rpn_target_match,
                                                        rpn_target_bbox, rpn_num_pos,
                                                        rcnn_pred_logits, rcnn_pred_deltas, tgt_cls_ids,
                                                        tgt_deltas)

                loss = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

                losses = [rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss]
                loss_vals = [float(l.detach().cpu()) for l in losses]

                loss.backward()

                optimizer.step()

                if np.isnan(loss_vals).any():
                    print('NaN detected!!!')

                return tuple(loss_vals)
            else:
                return 0.0, 0.0, 0.0, 0.0

        # Function for performing inference on a single image
        def f_pred_single(X, image_size):
            with torch.no_grad():
                X_var = torch.tensor(X, dtype=torch.float, device=torch_device)
                net.eval()

                # y1, x1, y2, x2
                window = np.array([[0.0, 0.0, float(image_size[0]), float(image_size[1])]])
                det_boxes, det_class_ids, det_scores = net.detect_forward_np(X_var, window)[0]
            return det_boxes, det_scores, det_class_ids

    elif head == 'rpn':
        # Train on a single batch
        # Note that the parameters match those returned by `_prepare_training_image` and `_prepare_training_batch`
        def f_train(X, rpn_match, rpn_bbox, rpn_num_pos, xf, gt_class_ids, gt_boxes, gt_masks, n_gts_per_sample):
            if gt_class_ids.shape[1] > 0 and gt_boxes.shape[1] > 0:
                xf = augmentation.apply_grid_to_image(augmentation.inv_nx2x3(xf), X.shape[2:])

                X_var = torch.tensor(X, dtype=torch.float, device=torch_device)
                rpn_target_match = torch.tensor(rpn_match, dtype=torch.long, device=torch_device)
                rpn_target_bbox = torch.tensor(rpn_bbox, dtype=torch.float, device=torch_device)
                xf_var = torch.tensor(xf, dtype=torch.float, device=torch_device)

                # Zero the gradients
                optimizer.zero_grad()

                # Grid for augmenting X
                grid_X = F.affine_grid(xf_var, X_var.size())
                X_aug = F.grid_sample(X_var, affine_torch.torch_pad_grid(grid_X, img_pad_mode))

                # Add gaussian noise
                if gaussian_noise != 0.0:
                    noise_var = (torch.randn(X_aug.size(), device=torch_device) * gaussian_noise)
                    X_aug = X_aug + noise_var

                if per_sample_loss:
                    # Run object detection and get per-sample loss
                    rpn_num_pos = torch.tensor(rpn_num_pos, dtype=torch.int, device=torch_device)
                    n_gts_per_sample = torch.tensor(n_gts_per_sample, dtype=torch.int, device=torch_device)
                    rpn_class_loss, rpn_bbox_loss = net.train_loss_forward(X_aug, rpn_target_match, rpn_target_bbox,
                                                                           rpn_num_pos)

                    rpn_class_loss = rpn_class_loss.mean()
                    rpn_bbox_loss = rpn_bbox_loss.mean()
                else:
                    # Run object detection
                    rpn_pred_logits, rpn_pred_bbox = net.train_forward(X_aug)

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss = rpn.compute_rpn_losses(
                        net.config, rpn_pred_logits, rpn_pred_bbox, rpn_target_match, rpn_target_bbox, rpn_num_pos)

                loss = rpn_class_loss + rpn_bbox_loss

                losses = [rpn_class_loss, rpn_bbox_loss]
                loss_vals = [float(l.detach().cpu()) for l in losses]

                loss.backward()

                optimizer.step()

                if np.isnan(loss_vals).any():
                    print('NaN detected!!!')

                return tuple(loss_vals)
            else:
                return 0.0, 0.0, 0.0, 0.0

        # Function for performing inference on a single image and convert the Mask R-CNN output to a label image
        def f_pred_single(X, image_size):
            with torch.no_grad():
                X_var = torch.tensor(X, dtype=torch.float, device=torch_device)
                net.eval()

                # y1, x1, y2, x2
                det_boxes, det_scores = net.detect_forward_np(X_var)[0]
            return det_boxes, det_scores

    # Perform inference on a single image,
    def predict_image(x):
        image_size = x.shape[:2]

        # Read the input image
        padded_shape = augmentation.round_up_shape(x.shape[:2], BLOCK_SIZE)
        img_padding = augmentation.compute_padding(x.shape[:2], padded_shape)

        if standardisation == 'std':
            x = (x - x.mean(axis=(0, 1), keepdims=True)) / (x.std(axis=(0, 1), keepdims=True) + 1.0e-6)
        elif standardisation == 'zeromean':
            x = x - x.mean(axis=(0, 1), keepdims=True)
        elif standardisation == 'zerocentre':
            x = x * 2.0 - 1.0
        elif standardisation == 'none':
            pass
        else:
            raise RuntimeError

        x = np.pad(x, img_padding + [(0, 0)], mode=img_pad_mode_np)

        x = x.transpose(2, 0, 1)[None, :, :, :]
        x = x.astype(np.float32)

        return f_pred_single(x, image_size)

    if head == 'mask_rcnn':
        def inference_on_test_set(output_dir, eval_predictions, epoch):
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            net.eval()
            t1 = time.time()
            precs = []
            n_dets = []
            n_reals = []
            if plot_dir != '':
                os.makedirs(plot_dir, exist_ok=True)
            for i in test_indices:
                i = int(i)
                det_boxes, det_scores, det_class_ids, mask_boxes, mrcnn_mask, pred_labels, cls_map = predict_image(
                    d_test.X[i])

                if plot_dir != '':
                    plot_path = os.path.join(plot_dir, 'segmentation_{:04d}_epoch{:04d}.png'.format(i, epoch))
                    plt.figure(figsize=(7, 7))
                    ax = plt.subplot(1, 1, 1)
                    ax.imshow(visualise_labels(d_test.X[i], pred_labels, mark_edges=True))
                    plot_boxes(ax, det_boxes[0])
                    plt.savefig(plot_path)
                    plt.close()

                if output_dir is not None:
                    if hasattr(d_test, 'names'):
                        sample_name = d_test.names[i]
                    else:
                        sample_name = 'sample{:04d}'.format(i)
                    pred_path = os.path.join(output_dir, '{}.npz'.format(sample_name))
                    np.savez_compressed(pred_path, det_scores=det_scores, det_class_ids=det_class_ids, det_boxes=det_boxes,
                                        mask_boxes=mask_boxes, mrcnn_mask=mrcnn_mask, labels=pred_labels,
                                        cls_map=cls_map)

                    labels_path = os.path.join(output_dir, '{}_labels.png'.format(sample_name))
                    Image.fromarray(pred_labels.astype(np.uint32)).save(labels_path)

                    clsmap_path = os.path.join(output_dir, '{}_cls.png'.format(sample_name))
                    Image.fromarray(cls_map.astype(np.uint32)).save(clsmap_path)

                if eval_predictions and d_test.y is not None:
                    true_labels = d_test.y[i]
                    n_reals.append(len(np.unique(true_labels[true_labels>0])))
                    n_dets.append(len(np.unique(pred_labels[pred_labels>0])))
                    prec = inference.mean_precision(d_test.y[i], pred_labels)
                    precs.append(prec)

            t2 = time.time()
            if eval_predictions and d_test.y is not None:
                log('Predicted test images in {:.3f}s: mean precision = {:.3%} (avg dets={:.6f}, avg reals={:.6f})'.format(
                    t2 - t1, np.mean(precs), np.mean(n_dets), np.mean(n_reals)))
            else:
                print('Predicted test images in {:.3f}s'.format(t2 - t1))

    elif head == 'faster_rcnn':
        def inference_on_test_set(output_dir, eval_predictions, epoch):
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            net.eval()
            t1 = time.time()
            accs = []
            n_dets = []
            n_reals = []
            if plot_dir != '':
                os.makedirs(plot_dir, exist_ok=True)
            for i in test_indices:
                i = int(i)
                det_boxes, det_scores, det_class_ids = predict_image(d_test.X[i])

                if plot_dir != '':
                    plot_path = os.path.join(plot_dir, 'detections_{:04d}_epoch{:04d}.png'.format(i, epoch))
                    plt.figure(figsize=(7, 7))
                    ax = plt.subplot(1, 1, 1)
                    ax.imshow(d_test.X[i])
                    plot_boxes(ax, det_boxes[0])
                    plt.savefig(plot_path)
                    plt.close()

                if output_dir is not None:
                    pred_path = os.path.join(output_dir, '{}.npz'.format(d_test.names[i]))
                    np.savez_compressed(pred_path, det_scores=det_scores, det_class_ids=det_class_ids, det_boxes=det_boxes)

                if eval_predictions and d_test.y is not None:
                    (acc, dets, reals) = inference.evaluate_box_predictions_from_labels(
                        det_boxes[0], d_test.y[i], d_test.X[i].shape[:2])
                    accs.append(acc)
                    n_dets.append(dets)
                    n_reals.append(reals)

            t2 = time.time()
            if eval_predictions and d_test.y is not None:
                log('Predicted test images in {:.3f}s: acc={:.6%} (avg dets={:.6f}, avg reals={:.6f})'.format(
                    t2 - t1, np.mean(accs), np.mean(n_dets), np.mean(n_reals)))
            else:
                print('Predicted test images in {:.3f}s'.format(t2 - t1))

    elif head == 'rpn':
        def inference_on_test_set(output_dir, eval_predictions, epoch):
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            net.eval()
            t1 = time.time()
            accs = []
            n_dets = []
            n_reals = []
            if plot_dir != '':
                os.makedirs(plot_dir, exist_ok=True)
            for i in test_indices:
                i = int(i)
                det_boxes, det_scores = predict_image(d_test.X[i])
                det_boxes = det_boxes[det_scores >= 0.5][None, ...]

                if plot_dir != '':
                    plot_path = os.path.join(plot_dir, 'detections_{:04d}_epoch{:04d}.png'.format(i, epoch))
                    plt.figure(figsize=(7, 7))
                    ax = plt.subplot(1, 1, 1)
                    ax.imshow(d_test.X[i])
                    plot_boxes(ax, det_boxes[0])
                    plt.savefig(plot_path)
                    plt.close()

                if output_dir is not None:
                    pred_path = os.path.join(output_dir, '{}.npz'.format(d_test.names[i]))
                    np.savez_compressed(pred_path, det_scores=det_scores, det_boxes=det_boxes)

                if eval_predictions and d_test.y is not None:
                    (acc, dets, reals) = inference.evaluate_box_predictions_from_labels(
                        det_boxes[0], d_test.y[i], d_test.X[i].shape[:2])
                    accs.append(acc)
                    n_dets.append(dets)
                    n_reals.append(reals)

            t2 = time.time()
            if eval_predictions and d_test.y is not None:
                log('Predicted test images in {:.3f}s: acc={:.6%} (avg dets={:.6f}, avg reals={:.6f})'.format(
                    t2 - t1, np.mean(accs), np.mean(n_dets), np.mean(n_reals)))
            else:
                print('Predicted test images in {:.3f}s'.format(t2 - t1))

    else:
        raise RuntimeError

    # Report setttings
    log('Command line:')
    log(' '.join(sys.argv))
    log('')
    log('Settings:')
    log(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))]))

    # Report dataset size
    log('Dataset:')
    log('TRAIN len(X)={}'.format(len(train_indices)))
    log('TEST len(X)={}'.format(len(test_indices)))

    n_train_batches = sampling.num_batches(len(train_indices), batch_size)
    n_train_batches = len(train_indices) // batch_size

    print('Training...')
    train_ds = data_source.ArrayDataSource([d_train.X, d_train.y], repeats=-1,
                                           indices=train_indices)
    train_ds = train_ds.map(_prepare_training_batch)
    if pool is not None:
        train_ds = pool.parallel_data_source(train_ds, batch_buffer_size=min(20, n_train_batches))

    if seed != 0:
        shuffle_rng = np.random.RandomState(seed)
    else:
        shuffle_rng = np.random

    train_batch_iter = train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng)
    best_val_iou = 0.0

    for epoch in range(num_epochs):
        t1 = time.time()

        # Enable training mode
        net.train()

        losses = data_source.batch_map_mean(
            f_train, train_batch_iter, progress_iter_func=progress_bar, n_batches=n_train_batches)

        t2 = time.time()

        if head == 'mask_rcnn':
            log('Epoch {} took {:.2f}s: TRAIN RPN cls={:.6f}, RPN box={:.6f}, RCNN cls={:.6f}, RCNN box={:.6f}, '
                'mask={:.6f}'.format(epoch, t2 - t1, *losses))
        elif head == 'faster_rcnn':
            log('Epoch {} took {:.2f}s: TRAIN RPN cls={:.6f}, RPN box={:.6f}, RCNN cls={:.6f}, '
                'RCNN box={:.6f}'.format(epoch, t2 - t1, *losses))
        elif head == 'rpn':
            log('Epoch {} took {:.2f}s: TRAIN RPN cls={:.6f}, RPN box={:.6f}'.format(epoch, t2 - t1, *losses))
        else:
            raise RuntimeError

        epoch_name = 'epoch_{:04d}'.format(epoch + 1)

        if save_model_every_n_epochs != -1:
            save_this_epoch = ((epoch + 1) % save_model_every_n_epochs) == 0 and save_model_every_n_epochs != -1

            if save_this_epoch:
                # Save network
                if model_file != '':
                    name, ext = os.path.splitext(model_file)
                    epoch_model_file = name + '_' + epoch_name + ext
                    model_state = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
                    with open(epoch_model_file, 'wb') as f:
                        pickle.dump(model_state, f)

        if prediction_every_n_epochs != -1 or eval_every_n_epochs != -1:
            predict_this_epoch = ((epoch + 1) % prediction_every_n_epochs) == 0 and prediction_every_n_epochs != -1
            eval_this_epoch = ((epoch + 1) % eval_every_n_epochs) == 0 and eval_every_n_epochs != -1
            if predict_this_epoch or eval_this_epoch:
                out_dir = os.path.join(predictions_dir, epoch_name) if predict_this_epoch else None
                inference_on_test_set(out_dir, eval_this_epoch, epoch)

    if save_model_every_n_epochs == -1:
        # Save network
        if model_file != '':
            model_state = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
            with open(model_file, 'wb') as f:
                pickle.dump(model_state, f)

    if prediction_every_n_epochs == -1:
        # Predict on test set
        if predictions_dir is not None:
            inference_on_test_set(predictions_dir, True, None)


if __name__ == '__main__':
    experiment()