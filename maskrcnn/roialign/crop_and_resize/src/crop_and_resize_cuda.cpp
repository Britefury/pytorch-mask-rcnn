#include <torch/extension.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#include <vector>
#include <tuple>

#include "crop_and_resize_kernel.h"

auto state = at::globalContext().getTHCState();

int crop_and_resize_forward_cuda(at::Tensor image,
                                 at::Tensor boxes,      // [y1, x1, y2, x2]
                                 at::Tensor box_index,  // range in [0, batch_size)
                                 const float extrapolation_value,
                                 const int crop_height,
                                 const int crop_width,
                                 at::Tensor crops,
                                 int device_id) {
    const auto batch_size = image.size(0);
    const auto depth = image.size(1);
    const auto image_height = image.size(2);
    const auto image_width = image.size(3);

    const auto num_boxes = boxes.size(0);

    // Check output size
    if ((crops.size(0) != num_boxes) ||
        (crops.size(1) != depth) ||
        (crops.size(2) != crop_height) ||
        (crops.size(3) != crop_width)) {
        return -1;
    }

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeLaucher(
        image.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth, extrapolation_value,
        crops.data<float>(),
        stream, device_id
    );

    return 0;
}


void crop_and_resize_backward_cuda(at::Tensor grads,
                                   at::Tensor boxes,      // [y1, x1, y2, x2]
                                   at::Tensor box_index,  // range in [0, batch_size)
                                   at::Tensor grads_image, // resize to [bsize, c, hc, wc]
                                   int device_id) {
    // shape
    const auto batch_size = grads_image.size(0);
    const auto depth = grads_image.size(1);
    const auto image_height = grads_image.size(2);
    const auto image_width = grads_image.size(3);

    const auto num_boxes = grads.size(0);
    const auto crop_height = grads.size(2);
    const auto crop_width = grads.size(3);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeBackpropImageLaucher(
        grads.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth,
        grads_image.data<float>(),
        stream, device_id
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("crop_and_resize_forward_cuda", &crop_and_resize_forward_cuda, "Crop and resize forward pass - CUDA");
  m.def("crop_and_resize_backward_cuda", &crop_and_resize_backward_cuda, "Crop and resize backward pass - CUDA");
}