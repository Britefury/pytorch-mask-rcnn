cd maskrcnn/nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=${1}
cd ../../
python build.py
cd ../../

cd maskrcnn/roialign/crop_and_resize/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=${1}
cd ../../
python build.py
cd ../../../

cd maskrcnn/roialign/roi_align/src/cuda/
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=${1}
cd ../../
python build.py
cd ../../../