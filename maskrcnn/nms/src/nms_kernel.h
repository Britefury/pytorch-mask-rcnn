#ifndef _NMS_KERNEL
#define _NMS_KERNEL

typedef unsigned long long mask_t;

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(mask_t) * 8;

long _nms(int boxes_num, float * boxes_dev,
          int64_t *keep_flat, float nms_overlap_thresh, int device_id);

#endif

