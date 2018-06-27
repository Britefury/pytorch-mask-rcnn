// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#include <THC/THC.h>
#include <TH/TH.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "cuda/nms_kernel.h"


THCState *state = at::globalContext().thc_state;

extern "C" int gpu_nms(THLongTensor * keep, THLongTensor* num_out, THCudaTensor * boxes, float nms_overlap_thresh,
                       int device_id) {
  // boxes has to be sorted
  THArgCheck(THLongTensor_isContiguous(keep), 0, "boxes must be contiguous");
  THArgCheck(THCudaTensor_isContiguous(state, boxes), 2, "boxes must be contiguous");
  // Number of ROIs
  int boxes_num = THCudaTensor_size(state, boxes, 0);

  float* boxes_flat = THCudaTensor_data(state, boxes);
  int64_t * keep_flat = THLongTensor_data(keep);

  long num_to_keep = _nms(boxes_num, boxes_flat, keep_flat, nms_overlap_thresh, device_id);

  int64_t * num_out_flat = THLongTensor_data(num_out);
  * num_out_flat = num_to_keep;

  return 1;
}
