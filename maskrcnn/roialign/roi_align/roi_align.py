# Adapted from Detectron.pytorch: https://github.com/roytseng-tw/Detectron.pytorch

import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from ._ext import roi_align


# TODO use save_for_backward instead
class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, sampling_ratio):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.sampling_ratio = int(sampling_ratio)
        self.feature_size = None

    def forward(self, features, rois, sample_indices):
        self.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height,
                                             self.aligned_width,
                                             self.sampling_ratio, features,
                                             sample_indices, rois, output, features.device.index)
        else:
            raise NotImplementedError

        self.save_for_backward(rois, sample_indices)

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        rois, sample_indices = self.saved_tensors

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = self.rois.new(batch_size, num_channels, data_height,
                                   data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height,
                                          self.aligned_width,
                                          self.sampling_ratio, grad_output,
                                          sample_indices, rois,
                                          grad_input, grad_output.device.index)

        # print grad_input

        return grad_input, None


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width, sampling_ratio):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois, sample_indices):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
                                self.sampling_ratio)(features, rois, sample_indices)

class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, sampling_ratio):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois, sample_indices):
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                              self.sampling_ratio)(features, rois, sample_indices)
        return avg_pool2d(x, kernel_size=2, stride=1)

class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, sampling_ratio):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois, sample_indices):
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                              self.sampling_ratio)(features, rois, sample_indices)
        return max_pool2d(x, kernel_size=2, stride=1)
