// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
//
// Taken some code from
// https://github.com/longcw/yolo2-pytorch/blob/master/utils/nms/nms_kernel.cu
// for the purpose of allowing the NMS code to work on multiple GPUs.
// ------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include <iostream>
#include "nms_kernel.h"

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  float width = fmaxf(right - left + 1, 0.f), height = fmaxf(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, mask_t *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    mask_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)


long _nms(int boxes_num, float * boxes_dev,
         int64_t *keep_flat, float nms_overlap_thresh, int device_id) {
  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  // Get the current device ID
  int cur_device_id;
  CUDA_CHECK(cudaGetDevice(&cur_device_id));
  // Change device ID if necessary
  if (device_id != cur_device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
  }

  // Allocate device mask
  mask_t* mask_dev = NULL;
  CUDA_CHECK(cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(mask_t)));
  CUDA_CHECK(cudaMemset(mask_dev, 0, sizeof(mask_t) * boxes_num * col_blocks));

  // Run NMS
  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
  // Host mask array
  std::vector<mask_t> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0], mask_dev,
                        sizeof(mask_t) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(mask_dev));

  // Set device ID back if necessary
  if (device_id != cur_device_id) {
    CUDA_CHECK(cudaSetDevice(cur_device_id));
  }

  std::vector<mask_t> remv_host(col_blocks);
  std::fill(remv_host.begin(), remv_host.end(), 0);

  long num_to_keep = 0;

  int i, j;
  for (i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv_host[nblock] & (1ULL << inblock))) {
      keep_flat[num_to_keep++] = i;
      mask_t *p = &mask_host[0] + i * col_blocks;
      for (j = nblock; j < col_blocks; j++) {
        remv_host[j] |= p[j];
      }
    }
  }

  return num_to_keep;
}
