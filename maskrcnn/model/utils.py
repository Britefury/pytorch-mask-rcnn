import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data



############################################################
#  Logging Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
    # Print New Line on Complete
    if iteration == total:
        print()


############################################################
#  Pytorch Utility Functions
############################################################

_EMPTY_SIZES = {torch.Size([]), torch.Size([0])}

def not_empty(tensor):
    return tensor.size() not in _EMPTY_SIZES

def is_empty(tensor):
    return tensor.size() in _EMPTY_SIZES

def unique1d(tensor):
    device = tensor.device
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = torch.tensor([True], dtype=torch.uint8, device=device)
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]

def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    device = x.device
    ln2 = torch.log(torch.tensor([2.0], dtype=torch.float, device=device))
    return torch.log(x) / ln2

def concatenate_detection_arrays(dets):
    """
    Concatenate detection arrays along the batch axis
    Each NumPy array is of shape [batch, detection, ...]
    Concatenates NumPy arrays along the batch axis, zero-padding the detection axis if necessary.

    :param dets: list of NumPy arrays
    :return: (detections, n_dets_per_sample) where
        detections is the concatenated array
        n_dets_per_sample is a list giving the number of valid detections per batch sample
    """
    n_dets_per_sample = [d.shape[1] for d in dets]
    max_dets = max(n_dets_per_sample)
    padded_dets = []
    for det, n_dets in zip(dets, n_dets_per_sample):
        if n_dets < max_dets:
            z = np.zeros((1, max_dets - n_dets,) + det.shape[2:], dtype=det.dtype)
            padded_dets.append(np.append(det, z, axis=1))
        else:
            padded_dets.append(det)
    return np.concatenate(padded_dets, axis=0), n_dets_per_sample

def concatenate_detection_tensors(dets):
    """
    Concatenate detection tensors along the batch axis
    Each tensor is a Torch tensor of shape [batch, detection, ...]
    Concatenates torch tensors along the batch axis, zero-padding the detection axis if necessary.

    :param dets: list of Torch tensors
    :return: (detections, n_dets_per_sample) where
        detections is the concatenated tensor
        n_dets_per_sample is a list giving the number of valid detections per batch sample
    """
    n_dets_per_sample = [(d.size()[1] if len(d.size())>=2 else 0) for d in dets]

    if len(dets) == 1:
        return dets[0], n_dets_per_sample

    max_dets = max(n_dets_per_sample)
    det_shape = ()
    example_det = dets[0]
    for d, n_dets in zip(dets, n_dets_per_sample):
        if n_dets != 0:
            det_shape = d.size()[2:]
            example_det = d
            break
    if max_dets > 0:
        padded_dets = []
        for det, n_dets in zip(dets, n_dets_per_sample):
            if n_dets < max_dets:
                z = example_det.new(1, max_dets - n_dets, *det_shape).zero_().to(det.device)
                if n_dets == 0:
                    padded_dets.append(z)
                else:
                    padded_dets.append(torch.cat([det, z], dim=1))
            else:
                padded_dets.append(det)
        return torch.cat(padded_dets, dim=0), n_dets_per_sample
    else:
        return example_det.new(0), n_dets_per_sample

def concatenate_detections(*dets):
    """
    Concatenate detections along the batch axis
    Each entry in det_tuples is a tuple of detections for the corresponding sample

    :param dets: each item is a list of Torch tensors
    :return: (detections, n_dets_per_sample), where:
        detections is a tuple of concatenated tensors
        n_dets_per_sample is a list giving the number of valid detections per batch sample
    """
    detections = []
    n_dets_per_sample = []
    for dets_by_sample in dets:
        cat_dets, n_dets_per_sample = concatenate_detection_tensors(list(dets_by_sample))
        detections.append(cat_dets)
    return tuple(detections), n_dets_per_sample


def split_detections(n_dets_per_sample, *dets):
    """
    Split detection tensors from shape [batch, detections, ...] with zero padding
    into a list of tuples, where each tuple corresponds to a sample and contains
    tensors of the form [1, detections, ...]

    :param n_dets_per_sample: Number of detections in each sample in the batch
    :param dets: tensors of detections, where each tensor is of shape [batch, detection, ...]

    :return: [(sample0_detsA, sample0_detsB, ...), (sample1_detsA, sample1_detsB, ...)]
    """
    if len(n_dets_per_sample) == 1:
        return [dets]
    else:
        sample_dets = []
        for sample_i, n_dets in enumerate(n_dets_per_sample):
            sample_det = [d[sample_i:sample_i+1, :n_dets, ...] for d in dets]
            sample_dets.append(tuple(sample_det))
        return sample_dets


def flatten_detections(n_dets_per_sample, *dets):
    """
    Flatten detection tensors from shape [batch, detections, ...] with zero padding
    in the detections axis for unused detections. The number of used detections in
    each sample is specified in n_dets_per_sample.

    :param n_dets_per_sample: Number of detections in each sample in the batch
    :param dets: tensors of detections, where each tensor is of shape [batch, detection, ...]

    :return: (flat_dets0, flat_dets1, ...flat_detsN) where
        flat_dets0..flat_detsN are tensors of shape [batch/detection, ...]
    """
    # Flatten detections
    flat_dets = []
    for det in dets:
        flat = []
        for sample_i, n_dets in enumerate(n_dets_per_sample):
            if det.size() and n_dets > 0:
                flat.append(det[sample_i, :n_dets, ...])
        if len(flat) > 0:
            flat_dets.append(torch.cat(flat, dim=0))
        else:
            empty = det.new(torch.Size()).zero_()
            flat_dets.append(empty)

    return tuple(flat_dets)


def flatten_detections_with_sample_indices(n_dets_per_sample, *dets):
    """
    Flatten detection tensors from shape [batch, detections, ...] with zero padding
    in the detections axis for unused detections. The number of used detections in
    each sample is specified in n_dets_per_sample.

    :param n_dets_per_sample: Number of detections in each sample in the batch
    :param dets: tensors of detections, where each tensor is of shape [batch, detection, ...]

    :return: (flat_dets0, flat_dets1, ...flat_detsN, sample_indices) where
        flat_dets0..flat_detsN are tensors of shape [batch/detection, ...]
        sample_indices gives the index of the sample to which each detection in the preceeding
        arrays came from
    """
    sample_indices = []
    for sample_i, n_dets in enumerate(n_dets_per_sample):
        assign = torch.ones(n_dets).int() * sample_i
        sample_indices.append(assign)
    sample_indices = torch.cat(sample_indices, dim=0).to(dets[0].device)

    # Flatten detections
    flat_dets = []
    for det in dets:
        flat = []
        for sample_i, n_dets in enumerate(n_dets_per_sample):
            if det.size() and n_dets > 0:
                flat.append(det[sample_i, :n_dets, ...])
        if len(flat) > 0:
            flat_dets.append(torch.cat(flat, dim=0))
        else:
            empty = det.new(torch.Size()).zero_()
            flat_dets.append(empty)

    return tuple(flat_dets) + (sample_indices,)


def unflatten_detections(n_dets_per_sample, *flat_dets):
    """
    Un-flatten the detections in the tensors in flat_dets.

    :param n_dets_per_sample: Number of detections in each sample in the batch
    :param flat_dets: tensors of detections, where each tensor is of shape [sample/detection, ...]

    :return: dets where
        dets is a list of tensors of shape [batch, detection, ...]
        with zero-padding for unused detections
    """
    if len(n_dets_per_sample) == 1:
        return [d[None] for d in flat_dets]
    else:
        max_n_dets = max(n_dets_per_sample)
        n_samples = len(n_dets_per_sample)

        dets = [fdet.new(torch.Size((n_samples, max_n_dets, *fdet.size()[1:]))).zero_().to(fdet.device)
                for fdet in flat_dets]

        offset = 0
        for sample_i in range(n_samples):
            n_dets = n_dets_per_sample[sample_i]
            if n_dets > 0:
                for det, fdet in zip(dets, flat_dets):
                    det[sample_i, :n_dets] = fdet[offset:offset+n_dets]
                offset += n_dets
        return dets


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


############################################################
#  Box Utility Functions
############################################################

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps



def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


############################################################
#  Plotting Utility Functions
############################################################

def plot_image_with_boxes(ax, img, boxes, alpha=1.0, colours=None):
    ax.imshow(img)
    if colours is None:
        colours = ['red'] * len(boxes)
    for (y1, x1, y2, x2), col in zip(boxes, colours):
        rect = Rectangle((x1, y1), x2-x1, y2-y1, facecolor=None, edgecolor=col, fill=False, alpha=alpha)
        ax.add_patch(rect)

def plot_image_with_stratified_boxes(ax, img, boxes, alpha=1.0, colour='red'):
    ax.imshow(img)
    for group in boxes:
        group_colour = group.get('colour', colour)
        group_alpha = group.get('alpha', alpha)
        for (y1, x1, y2, x2) in group['boxes']:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=None, edgecolor=group_colour, fill=False, alpha=group_alpha)
            ax.add_patch(rect)
