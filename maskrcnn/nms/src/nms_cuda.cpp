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

#include "cuda/nms_kernel.h"


THCState *state = at::globalContext().thc_state;


extern "C" int gpu_nms(THLongTensor * keep, THLongTensor* num_out, THCudaTensor * boxes, float nms_overlap_thresh) {
  // boxes has to be sorted
  THArgCheck(THLongTensor_isContiguous(keep), 0, "boxes must be contiguous");
  THArgCheck(THCudaTensor_isContiguous(state, boxes), 2, "boxes must be contiguous");
  // Number of ROIs
  int boxes_num = THCudaTensor_size(state, boxes, 0);
  int boxes_dim = THCudaTensor_size(state, boxes, 1);

  float* boxes_flat = THCudaTensor_data(state, boxes);

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  THCudaLongTensor * mask = THCudaLongTensor_newWithSize2d(state, boxes_num, col_blocks);
  int64_t* mask_flat = THCudaLongTensor_data(state, mask);

  _nms(boxes_num, boxes_flat, mask_flat, nms_overlap_thresh);

  THLongTensor * mask_cpu = THLongTensor_newWithSize2d(boxes_num, col_blocks);
  THLongTensor_copyCuda(state, mask_cpu, mask);
  THCudaLongTensor_free(state, mask);

  int64_t * mask_cpu_flat = THLongTensor_data(mask_cpu);

  THLongTensor * remv_cpu = THLongTensor_newWithSize1d(col_blocks);
  int64_t* remv_cpu_flat = THLongTensor_data(remv_cpu);
  THLongTensor_fill(remv_cpu, 0);

  int64_t * keep_flat = THLongTensor_data(keep);
  long num_to_keep = 0;

  int i, j;
  for (i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv_cpu_flat[nblock] & (1ULL << inblock))) {
      keep_flat[num_to_keep++] = i;
      int64_t *p = &mask_cpu_flat[0] + i * col_blocks;
      for (j = nblock; j < col_blocks; j++) {
        remv_cpu_flat[j] |= p[j];
      }
    }
  }

  int64_t * num_out_flat = THLongTensor_data(num_out);
  * num_out_flat = num_to_keep;

  THLongTensor_free(mask_cpu);
  THLongTensor_free(remv_cpu);

  return 1;
}
