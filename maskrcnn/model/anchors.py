import collections
import threading
import math
import numpy as np
import torch


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, image_shape, feature_stride, anchor_stride,
                     valid_window=None, centre_anchors=True, round_up_feature_shape=True,
                     detectron_compatible=False):
    """
    :param scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    :param ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    :param image_shape: [height, width] the shape of the image from which the
        feature map was generated, should be divisible by `feature_stride`
    :param feature_stride: Stride of the feature map relative to the image in pixels.
    :param anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    :param valid_window: tuple `(y1, x1, y2, x2)` that specifies the window within anchors
        must completely lie in order to be considered to be valid. If `None` (default)
        when the window `(0, 0, image_shape[0], image_shape[1])` will be used.
    :param centre_anchors: bool, default = True, if True align the centres of the anchor boxes
        with the centres the centres of the feature map 'pixels', if False align anchor box centres
        with the top-left corners of the feature map 'pixels'
    :param round_up_feature_shape: bool, default = True, if True round up the feature shape
        if the image shape does not divide by the feature stride exactly
    :param detectron_compatible: if True generate anchors that are the same size as the
        ones used by Detectron
    :return: tuple `(boxes, valid_mask)` where boxes is an (N,4) array
        where each row specifies an anchor box using the form [y1, x1, y2, x2]
        and `valid_mask` is a (N,) array of bools that is true for each anchor that
        lies within `valid_window` (note that the valid anchors are pre-filtered with
        the mask, so there are `V` entries in the mask that are True)
    """
    if valid_window is None:
        valid_window = (0.0, 0.0, float(image_shape[0]), float(image_shape[1]))

    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    if detectron_compatible:
        size_ratios = feature_stride * feature_stride / ratios
        widths = np.round(np.sqrt(size_ratios))
        heights = np.round(widths * ratios)
        widths = widths * scales / feature_stride
        heights = heights * scales / feature_stride
    else:
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

    # Compute feature shape
    if round_up_feature_shape:
        feat_shape = (int(math.ceil(float(image_shape[0]) / feature_stride)),
                      int(math.ceil(float(image_shape[1]) / feature_stride)))
    else:
        feat_shape = image_shape[0] / feature_stride, image_shape[1] / feature_stride

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, feat_shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, feat_shape[1], anchor_stride) * feature_stride
    if centre_anchors:
        shifts_y = shifts_y + feature_stride * 0.5
        shifts_x = shifts_x + feature_stride * 0.5
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    # if detectron_compatible:
    #     boxes = np.concatenate([box_centers - 0.5 * box_sizes,
    #                             box_centers + 0.5 * box_sizes - 1], axis=1)
    # else:
    #     boxes = np.concatenate([box_centers - 0.5 * box_sizes,
    #                             box_centers + 0.5 * box_sizes], axis=1)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    valid_mask = ((boxes[:, 0] >= valid_window[0]) & (boxes[:, 1] >= valid_window[1]) &
                  (boxes[:, 2] <= valid_window[2]) & (boxes[:, 3] <= valid_window[3]))
    return boxes, valid_mask


def generate_pyramid_anchors(scales, ratios, image_shape, feature_strides,
                             anchor_stride, valid_window=None, centre_anchors=True,
                             round_up_feature_shape=True, detectron_compatible=False):
    """
    Generate anchors corresponding to the levels of a feature pyramid network (FPN).

    There are L levels in the pyramid.

    :param scales: 1D array (L,) of anchor sizes in pixels. Should be one for each
        pyramid level. Example: [32, 64, 128]
    :param ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    :param image_shape: [height, width] the shape of the image from which the
        feature map was generated, should be divisible by `feature_stride`
    :param feature_strides: 1D array  (L,) of feature map strides relative to the image
        in pixels. Should be one for each pyramid level. Example: [4, 8, 16]
    :param anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    :param valid_window: tuple `(y1, x1, y2, x2)` that specifies the window within anchors
        must completely lie in order to be considered to be valid. If `None` (default)
        when the window `(0, 0, image_shape[0], image_shape[1])` will be used.
    :param centre_anchors: bool, default = True, if True align the centres of the anchor boxes
        with the centres the centres of the feature map 'pixels', if False align anchor box centres
        with the top-left corners of the feature map 'pixels'
    :param round_up_feature_shape: bool, default = True, if True round up the feature shape
        if the image shape does not divide by the feature stride exactly
    :param detectron_compatible: if True generate anchors that are the same size as the
        ones used by Detectron
    :return: tuple `(valid_boxes, valid_mask)` where valid_boxes is an (V,4) array
        where each row specifies a valid anchor box using the form [y1, x1, y2, x2]
        and `valid_mask` is a (N,) array of bools that is true for each anchor that
        lies within `valid_window` (note that the valid anchors are pre-filtered with
        the mask, so there are `V` entries in the mask that are True)
    """

    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    valid_masks = []
    for i in range(len(scales)):
        anch, val = generate_anchors(scales[i], ratios, image_shape,
                                     feature_strides[i], anchor_stride,
                                     valid_window=valid_window,
                                     centre_anchors=centre_anchors,
                                     round_up_feature_shape=round_up_feature_shape,
                                     detectron_compatible=detectron_compatible)
        anchors.append(anch)
        valid_masks.append(val)
    return anchors, valid_masks


class AnchorCache (object):
    def __init__(self, config, max_cached_anchors=32, max_cached_vars=8):
        """
        Anchors cache

        Caches anchors by image shape

        :param config: Config instance
        :param max_length: max size of cache
        """
        self.__config = config
        self.__anchs_cache = collections.OrderedDict()
        self.__var_cache = collections.OrderedDict()
        self.__var_cache_by_level = collections.OrderedDict()
        self.__max_anchors = max_cached_anchors
        self.__max_vars = max_cached_vars
        self.__anchor_lock = threading.Lock()
        self.__var_lock = threading.Lock()
        self.__var_lock_by_level = threading.Lock()


    def get_anchors_and_valid_masks_for_image_shape_by_level(self, image_shape):
        with self.__anchor_lock:
            if image_shape in self.__anchs_cache:
                if self.__max_anchors is None:
                    return self.__anchs_cache[image_shape]
                else:
                    # Get cached entry
                    entry = self.__anchs_cache[image_shape]
                    # Move entry to end (most recently used)
                    self.__anchs_cache.move_to_end(image_shape)
                    return entry
            else:
                entry = generate_pyramid_anchors(
                    self.__config.RPN_ANCHOR_SCALES, self.__config.RPN_ANCHOR_RATIOS,
                    image_shape, self.__config.BACKBONE_STRIDES, self.__config.RPN_ANCHOR_STRIDE,
                    centre_anchors=self.__config.CENTRE_ANCHORS,
                    round_up_feature_shape=self.__config.ANCHORS_ROUND_UP_FEATURE_SHAPE,
                    detectron_compatible=self.__config.ANCHORS_DETECTRON)

                # Insert into cache
                self.__anchs_cache[image_shape] = entry

                if self.__max_anchors is not None:
                    # While cache is too big...
                    while len(self.__anchs_cache) > self.__max_anchors:
                        # ... remove the first (least recently used) item
                        self.__anchs_cache.popitem(last=False)

                return entry


    def get_anchors_and_valid_masks_for_image_shape(self, image_shape):
        anchors, valid_masks = self.get_anchors_and_valid_masks_for_image_shape_by_level(image_shape)
        return np.concatenate(anchors, axis=0), np.concatenate(valid_masks, axis=0)


    def _anchors_to_var(self, anchors, device):
        return torch.tensor(anchors, dtype=torch.float, device=device)


    def get_anchors_var_for_image_shape_by_level(self, image_shape, device):
        key = (image_shape, device.type, device.index)
        with self.__var_lock_by_level:
            if key in self.__var_cache_by_level:
                if self.__max_vars is None:
                    return self.__var_cache_by_level[key]
                else:
                    # Get cached entry
                    entry = self.__var_cache_by_level[key]
                    # Move entry to end (most recently used)
                    self.__var_cache_by_level.move_to_end(key)
                    return entry
            else:
                anchors_lvl, _ = self.get_anchors_and_valid_masks_for_image_shape_by_level(image_shape)

                anchor_vars = [torch.tensor(anchors, dtype=torch.float, device=device) for anchors in anchors_lvl]

                # Insert into cache
                self.__var_cache_by_level[key] = anchor_vars

                if self.__max_vars is not None:
                    # While cache is too big...
                    while len(self.__var_cache_by_level) > self.__max_vars:
                        # ... remove the first (least recently used) item
                        self.__var_cache_by_level.popitem(last=False)

                return anchor_vars


    def get_anchors_var_for_image_shape(self, image_shape, device):
        key = (image_shape, device.type, device.index)
        with self.__var_lock:
            if key in self.__var_cache:
                if self.__max_vars is None:
                    return self.__var_cache[key]
                else:
                    # Get cached entry
                    entry = self.__var_cache[key]
                    # Move entry to end (most recently used)
                    self.__var_cache.move_to_end(key)
                    return entry
            else:
                anchors, _ = self.get_anchors_and_valid_masks_for_image_shape(image_shape)
                anchors_var = torch.tensor(anchors, dtype=torch.float, device=device)

                # Insert into cache
                self.__var_cache[key] = anchors_var

                if self.__max_vars is not None:
                    # While cache is too big...
                    while len(self.__var_cache) > self.__max_vars:
                        # ... remove the first (least recently used) item
                        self.__var_cache.popitem(last=False)

                return anchors_var

