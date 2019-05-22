import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from . import crop_and_resize_cpu
try:
    from . import crop_and_resize_cuda
except ImportError:
    crop_and_resize_cuda = None


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros(boxes.shape[0], image.shape[1],
                            self.crop_height, self.crop_width,
                            dtype=torch.float, device=image.device)

        if image.is_cuda:
            status = crop_and_resize_cuda.crop_and_resize_forward_cuda(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops, image.device.index)
        else:
            status = crop_and_resize_cpu.crop_and_resize_forward_cpu(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)
        if status == -1:
            raise RuntimeError('CropAndResizeFunction.forward: crops.shape is incorrect')

        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)

        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros(self.im_size, dtype=grad_outputs.dtype, device=grad_outputs.device)

        if grad_outputs.is_cuda:
            crop_and_resize_cuda.crop_and_resize_backward_cuda(
                grad_outputs, boxes, box_ind, grad_image, grad_outputs.device.index
            )
        else:
            crop_and_resize_cpu.crop_and_resize_backward_cpu(
                grad_outputs, boxes, box_ind, grad_image
            )

        return grad_image, None, None


class CropAndResizeAligned (object):
    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self._crop = CropAndResize(crop_height, crop_width, extrapolation_value=extrapolation_value)

        self.crop_scale = float(crop_height) / float(crop_height + 1), float(crop_width) / float(crop_width + 1)


    def __call__(self, image, boxes, box_ind):
        device = image.device

        dst_scale_var = torch.tensor([
            [self.crop_scale[0], self.crop_scale[1], self.crop_scale[0], self.crop_scale[1]]], dtype=torch.float, device=device)

        src_size = image.size()[2:]
        src_scale = float(src_size[0]) / float(src_size[0] + 1), float(src_size[1]) / float(src_size[1] + 1)
        src_scale = torch.tensor([[src_scale[0], src_scale[1], src_scale[0], src_scale[1]]], dtype=torch.float, device=device)

        # Pad the image so that there is smooth interpolation along the edges
        padded_image = F.pad(image, [1, 1, 1, 1])
        box_centres = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        box_centres = torch.cat([box_centres, box_centres], dim=1)

        # Move sample locations to the centres of the crop-box pixels
        boxes_var = (boxes - box_centres) * dst_scale_var.detach() + box_centres
        # Scale sample grid to cover complete range of input
        boxes_var = (boxes_var - 0.5) * src_scale.detach() + 0.5

        return self._crop(padded_image, boxes_var, box_ind)


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)
