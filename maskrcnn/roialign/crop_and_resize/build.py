import os
import torch
import sys
from torch.utils.ffi import create_extension

sources = ['src/crop_and_resize.c']
headers = ['src/crop_and_resize.h']
defines = []
with_cuda = False
libraries = []
extra_compile_args = ['-fopenmp']
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
    sources += ['src/crop_and_resize_gpu.cpp']
    headers += ['src/crop_and_resize_gpu.h']
    defines += [('WITH_CUDA', None)]
    extra_objects += ['src/cuda/crop_and_resize_kernel.cu.{}'.format(obj_ext)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.crop_and_resize',
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
