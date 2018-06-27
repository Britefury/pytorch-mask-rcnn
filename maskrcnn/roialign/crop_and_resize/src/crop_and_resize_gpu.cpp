#include <THC/THC.h>
#include <ATen/ATen.h>
#include "cuda/crop_and_resize_kernel.h"

THCState *state = at::globalContext().thc_state;


extern "C" int crop_and_resize_gpu_forward(
    THCudaTensor * image,           // shape = (sample, feature, height, width)
    THCudaTensor * boxes,           // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    THCudaTensor * crops,            // shape = (crop, feature, crop_height, crop_width)
    int device_id
) {
    const int batch_size = THCudaTensor_size(state, image, 0);
    const int depth = THCudaTensor_size(state, image, 1);
    const int image_height = THCudaTensor_size(state, image, 2);
    const int image_width = THCudaTensor_size(state, image, 3);

    const int num_boxes = THCudaTensor_size(state, boxes, 0);

    if ((THCudaTensor_size(state, crops, 0) != num_boxes) ||
        (THCudaTensor_size(state, crops, 1) != depth) ||
        (THCudaTensor_size(state, crops, 2) != crop_height) ||
        (THCudaTensor_size(state, crops, 3) != crop_width)) {
        return -1;
    }

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeLaucher(
        THCudaTensor_data(state, image),
        THCudaTensor_data(state, boxes),
        THCudaIntTensor_data(state, box_index),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth, extrapolation_value,
        THCudaTensor_data(state, crops),
        stream, device_id
    );
}


extern "C" void crop_and_resize_gpu_backward(
    THCudaTensor * grads,       // shape = (crop, feature, crop_height, crop_width)
    THCudaTensor * boxes,      // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    THCudaTensor * grads_image, // shape = (sample, feature, height, width)
    int device_id
) {
    // shape
    const int batch_size = THCudaTensor_size(state, grads_image, 0);
    const int depth = THCudaTensor_size(state, grads_image, 1);
    const int image_height = THCudaTensor_size(state, grads_image, 2);
    const int image_width = THCudaTensor_size(state, grads_image, 3);

    const int num_boxes = THCudaTensor_size(state, grads, 0);
    const int crop_height = THCudaTensor_size(state, grads, 2);
    const int crop_width = THCudaTensor_size(state, grads, 3);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeBackpropImageLaucher(
        THCudaTensor_data(state, grads),
        THCudaTensor_data(state, boxes),
        THCudaIntTensor_data(state, box_index),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth,
        THCudaTensor_data(state, grads_image),
        stream, device_id
    );
}