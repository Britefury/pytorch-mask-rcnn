// Adapted from Detectron.pytorch: https://github.com/roytseng-tw/Detectron.pytorch

int roi_align_forward_cuda(int aligned_height, int aligned_width, int sampling_ratio,
                           THCudaTensor * features, THCudaIntTensor * sample_indices, THCudaTensor * rois,
                           THCudaTensor * output);

int roi_align_backward_cuda(int aligned_height, int aligned_width, int sampling_ratio,
                            THCudaTensor * top_grad, THCudaIntTensor * sample_indices, THCudaTensor * rois,
                            THCudaTensor * bottom_grad);
