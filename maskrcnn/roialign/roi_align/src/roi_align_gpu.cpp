// Adapted from Detectron.pytorch: https://github.com/roytseng-tw/Detectron.pytorch

#include <THC/THC.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include "cuda/roi_align_kernel.h"

THCState *state = at::globalContext().thc_state;

extern "C" int roi_align_forward_cuda(int aligned_height, int aligned_width, int sampling_ratio,
                                      THCudaTensor * features, THCudaIntTensor * sample_indices, THCudaTensor * rois,
                                      THCudaTensor * output)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    int * sample_indices_flat = THCudaIntTensor_data(state, sample_indices);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 4)
    {
        return 0;
    }

    // data height
    int data_height = THCudaTensor_size(state, features, 2);
    // data width
    int data_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    ROIAlignForwardLaucher(
        data_flat, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, sampling_ratio, sample_indices_flat, rois_flat,
        output_flat, stream);

    return 1;
}

extern "C" int roi_align_backward_cuda(int aligned_height, int aligned_width, int sampling_ratio,
                                       THCudaTensor * top_grad, THCudaIntTensor * sample_indices, THCudaTensor * rois,
                                       THCudaTensor * bottom_grad)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    int * sample_indices_flat = THCudaIntTensor_data(state, sample_indices);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 4)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 2);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    ROIAlignBackwardLaucher(
        top_grad_flat, batch_size, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, sampling_ratio, sample_indices_flat, rois_flat,
        bottom_grad_flat, stream);

    return 1;
}
