# Adapted from Detectron.pytorch: https://github.com/roytseng-tw/Detectron.pytorch

from __future__ import print_function
import sys
import os
import torch
from torch.utils.ffi import create_extension

# sources = ['src/roi_align.c']
# headers = ['src/roi_align.h']
sources = []
headers = []
defines = []
with_cuda = False
libraries = []
extra_compile_args = []
extra_link_args = []
extra_objects = []

if sys.platform == 'win32':
    libraries += ["ATen", "_C", "cudart"]
    extra_link_args += ["/NODEFAULTLIB:library"]
    extra_compile_args += ['-std=c99']
    obj_ext = 'obj'
elif sys.platform == 'linux':
    extra_compile_args += ["-std=c++11"]
    obj_ext = 'o'

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_align_gpu.cpp']
    headers += ['src/roi_align_gpu.h']
    defines += [('WITH_CUDA', None)]
    extra_objects += ['src/cuda/roi_align_kernel.cu.{}'.format(obj_ext)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.roi_align',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

if __name__ == '__main__':
    ffi.build()
