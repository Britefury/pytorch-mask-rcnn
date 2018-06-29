import os
import torch
import sys
from torch.utils.ffi import create_extension

sources = ['src/nms.c']
headers = ['src/nms.h']
defines = []
with_cuda = False
libraries = []
extra_compile_args = []
extra_link_args = []
extra_objects = []

if sys.platform == 'win32':
    libraries += ["ATen", "_C", "cudart"]
    extra_compile_args += ["-std=c99"]
    extra_link_args += ["/NODEFAULTLIB:library"]
    obj_ext = 'obj'
elif sys.platform == 'linux':
    extra_compile_args += ["-std=c++11"]
    obj_ext = 'o'

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/nms_cuda.cpp']
    headers += ['src/nms_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
    extra_objects = ['src/cuda/nms_kernel.cu.{}'.format(obj_ext)]
else:
    extra_objects = []

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.nms',
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
