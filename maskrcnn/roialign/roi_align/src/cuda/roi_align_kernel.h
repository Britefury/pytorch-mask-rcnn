// Adapted from Detectron.pytorch: https://github.com/roytseng-tw/Detectron.pytorch

#ifndef _ROI_ALIGN_KERNEL
#define _ROI_ALIGN_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

__global__ void ROIAlignForward(const int nthreads, const float* image_in,
    const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
    const int *sample_indices, const float* boxes, float* image_out);

int ROIAlignForwardLaucher(
    const float* image_in, const int num_rois, const int height,
    const int width, const int channels, const int aligned_height,
    const int aligned_width,  const int sampling_ratio, const int *sample_indices, const float* boxes,
    float* image_out, cudaStream_t stream);

__global__ void ROIAlignBackward(const int nthreads, const float* image_out_diff,
    const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
    float* image_in_diff, const int *sample_indices, const float* boxes);

int ROIAlignBackwardLaucher(const float* image_out_diff, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int aligned_height,
    const int aligned_width,  const int sampling_ratio, const int *sample_indices, const float* boxes,
    float* image_in_diff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

