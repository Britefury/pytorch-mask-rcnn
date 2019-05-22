// Adapted from Detectron.pytorch: https://github.com/roytseng-tw/Detectron.pytorch

#include <torch/extension.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#include <vector>
#include <tuple>

#include "roi_align_kernel.h"

auto state = at::globalContext().getTHCState();

int roi_align_forward_cuda(int aligned_height, int aligned_width, int sampling_ratio,
                           at::Tensor features, at::Tensor sample_indices, at::Tensor rois,
                           at::Tensor output, int device_id)
{
    // Grab the input tensor
    float * data_flat = features.data<float>();
    int * sample_indices_flat = sample_indices.data<int>();
    float * rois_flat = rois.data<float>();

    float * output_flat = output.data<float>();

    // Number of ROIs
    auto num_rois = rois.size(0);
    auto size_rois = rois.size(0);
    if (size_rois != 4)
    {
        return 0;
    }

    // data height
    auto data_height = features.size(2);
    // data width
    auto data_width = features.size(3);
    // Number of channels
    auto num_channels = features.size(1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    ROIAlignForwardLaucher(
        data_flat, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, sampling_ratio, sample_indices_flat, rois_flat,
        output_flat, stream, device_id);

    return 1;
}

int roi_align_backward_cuda(int aligned_height, int aligned_width, int sampling_ratio,
                            at::Tensor top_grad, at::Tensor sample_indices, at::Tensor rois,
                            at::Tensor bottom_grad, int device_id)
{
    // Grab the input tensor
    float * top_grad_flat = top_grad.data<float>();
    int * sample_indices_flat = sample_indices.data<int>();
    float * rois_flat = rois.data<float>();

    float * bottom_grad_flat = bottom_grad.data<float>();

    // Number of ROIs
    auto num_rois = rois.size(0);
    auto size_rois = rois.size(0);
    if (size_rois != 4)
    {
        return 0;
    }

    // batch size
    auto batch_size = bottom_grad.size(0);
    // data height
    auto data_height = bottom_grad.size(2);
    // data width
    auto data_width = bottom_grad.size(3);
    // Number of channels
    auto num_channels = bottom_grad.size(1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    ROIAlignBackwardLaucher(
        top_grad_flat, batch_size, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, sampling_ratio, sample_indices_flat, rois_flat,
        bottom_grad_flat, stream, device_id);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward_cuda", &roi_align_forward_cuda, "ROI align forward pass - CUDA");
  m.def("roi_align_backward_cuda", &roi_align_backward_cuda, "ROI align backward pass - CUDA");
}