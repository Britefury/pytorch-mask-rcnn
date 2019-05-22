// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#include <torch/extension.h>
#include <vector>
#include <tuple>

#include "nms_kernel.h"


//THCState *state = at::globalContext().thc_state;

std::tuple<int, long> nms_cuda(at::Tensor keep_out, at::Tensor boxes, float nms_overlap_thresh, int device_id) {
    AT_CHECK(keep_out.is_contiguous(), "keep_out must be contiguous");
    AT_CHECK(boxes.is_contiguous(), "boxes must be contiguous");
    // Number of ROIs
    auto boxes_num = boxes.size(0);

    const auto boxes_flat = boxes.data<float>();
    auto keep_flat = keep_out.data<int64_t>();

    long num_to_keep = _nms(boxes_num, boxes_flat, keep_flat, nms_overlap_thresh, device_id);

    return std::tuple<int, long>(1, num_to_keep);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_cuda", &nms_cuda, "NMS - Cuda");
}