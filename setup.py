import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import torch.cuda

if torch.cuda.is_available():
    setup(name='pytorch-mask-rcnn',
          ext_modules=[
              CppExtension('maskrcnn.nms.nms_cpu', [os.path.join('maskrcnn', 'nms','src', 'nms.cpp')]),
              CUDAExtension('maskrcnn.nms.nms_cuda', [os.path.join('maskrcnn', 'nms', 'src', 'nms_cuda.cpp'),
                                                      os.path.join('maskrcnn', 'nms', 'src', 'nms_kernel.cu')]),
              CppExtension('maskrcnn.roialign.crop_and_resize.crop_and_resize_cpu', [
                  os.path.join('maskrcnn', 'roialign', 'crop_and_resize', 'src', 'crop_and_resize.cpp')]),
              CUDAExtension('maskrcnn.roialign.crop_and_resize.crop_and_resize_cuda', [
                  os.path.join('maskrcnn', 'roialign', 'crop_and_resize', 'src', 'crop_and_resize_cuda.cpp'),
                  os.path.join('maskrcnn', 'roialign', 'crop_and_resize', 'src', 'crop_and_resize_kernel.cu')]),
              CUDAExtension('maskrcnn.roialign.roi_align.roi_align_cuda', [
                  os.path.join('maskrcnn', 'roialign', 'roi_align', 'src', 'roi_align_cuda.cpp'),
                  os.path.join('maskrcnn', 'roialign', 'roi_align', 'src', 'roi_align_kernel.cu')]),
          ],
          cmdclass={'build_ext': BuildExtension})
else:
    setup(name='pytorch-mask-rcnn',
          ext_modules=[
              CppExtension('maskrcnn.nms.nms_cpu', [os.path.join('maskrcnn', 'nms','src', 'nms.cpp')]),
              CppExtension('maskrcnn.roialign.crop_and_resize.crop_and_resize_cpu', [
                  os.path.join('maskrcnn', 'roialign', 'crop_and_resize', 'src', 'crop_and_resize.cpp')]),
          ],
          cmdclass={'build_ext': BuildExtension})
