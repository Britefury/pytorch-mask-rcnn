#include <torch/extension.h>
#include <vector>
#include <tuple>

std::tuple<int, long> nms_cpu(at::Tensor keep_out, at::Tensor boxes, at::Tensor order, at::Tensor areas,
                              float nms_overlap_thresh) {
    AT_CHECK(keep_out.is_contiguous(), "keep_out must be contiguous");
    AT_CHECK(boxes.is_contiguous(), "boxes must be contiguous");
    AT_CHECK(order.is_contiguous(), "order must be contiguous");
    AT_CHECK(areas.is_contiguous(), "areas must be contiguous");

    // Number of ROIs
    const auto boxes_num = boxes.size(0);
    const auto boxes_dim = boxes.size(1);

    auto *keep_out_flat = keep_out.data<int64_t>();
    const auto *boxes_flat = keep_out.data<float>();
    const auto *order_flat = keep_out.data<int64_t>();
    const auto *areas_flat = keep_out.data<float>();

    std::vector<unsigned char> suppressed;
    suppressed.resize(boxes_num);

    // nominal indices
    int i, j;
    // sorted indices
    int _i, _j;
    // temp variables for box i's (the box currently under consideration)
    float ix1, iy1, ix2, iy2, iarea;
    // variables for computing overlap with box j (lower scoring box)
    float xx1, yy1, xx2, yy2;
    float w, h;
    float inter, ovr;

    long num_to_keep = 0;
    for (_i=0; _i < boxes_num; ++_i) {
        i = order_flat[_i];
        if (suppressed[i] == 1) {
            continue;
        }
        keep_out_flat[num_to_keep++] = i;
        ix1 = boxes_flat[i * boxes_dim];
        iy1 = boxes_flat[i * boxes_dim + 1];
        ix2 = boxes_flat[i * boxes_dim + 2];
        iy2 = boxes_flat[i * boxes_dim + 3];
        iarea = areas_flat[i];
        for (_j = _i + 1; _j < boxes_num; ++_j) {
            j = order_flat[_j];
            if (suppressed[j] == 1) {
                continue;
            }
            xx1 = fmaxf(ix1, boxes_flat[j * boxes_dim]);
            yy1 = fmaxf(iy1, boxes_flat[j * boxes_dim + 1]);
            xx2 = fminf(ix2, boxes_flat[j * boxes_dim + 2]);
            yy2 = fminf(iy2, boxes_flat[j * boxes_dim + 3]);
            w = fmaxf(0.0, xx2 - xx1 + 1);
            h = fmaxf(0.0, yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter / (iarea + areas_flat[j] - inter);
            if (ovr > nms_overlap_thresh) {
                suppressed[j] = 1;
            }
        }
    }

    return std::tuple<int, long>(1, num_to_keep);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_cpu", &nms_cpu, "NMS - CPU");
}