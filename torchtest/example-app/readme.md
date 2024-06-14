2024-06-14
minimal example from:
https://pytorch.org/cppdocs/installing.html

    when searching for torch, found that needed to install cuda
    see:
    https://www.cherryservers.com/blog/install-cuda-ubuntu

        followed steps; however, terminal said had depedencies on nivada-555.*,
          so I installed those
    Completed steps
is *1 an issue?
    seems to be working


*1
james@james-Katana-17-B12VGK:~/.../example-app$ cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
CMake Warning:
  No source or binary directory provided.  Both will be assumed to be the
  same as the current working directory, but note that this warning will
  become a fatal error in future CMake releases.


-- Found CUDA: /usr/local/cuda (found version "12.5") 
-- The CUDA compiler identification is NVIDIA 12.5.40
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found CUDAToolkit: /usr/local/cuda/include (found version "12.5.40") 
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- Caffe2: CUDA detected: 12.5
-- Caffe2: CUDA nvcc is: /usr/local/cuda/bin/nvcc
-- Caffe2: CUDA toolkit directory: /usr/local/cuda
-- Caffe2: Header version is: 12.5
CMake Warning at /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Caffe2/public/cuda.cmake:184 (message):
  Failed to compute shorthash for libnvrtc.so
Call Stack (most recent call first):
  /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Caffe2/Caffe2Config.cmake:87 (include)
  /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake:68 (find_package)
  CMakeLists.txt:4 (find_package)


-- USE_CUDNN is set to 0. Compiling without cuDNN support
-- USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support
-- Autodetected CUDA architecture(s):  8.9
-- Added CUDA NVCC flags for: -gencode;arch=compute_89,code=sm_89
CMake Warning at /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake:22 (message):
  static library kineto_LIBRARY-NOTFOUND not found.
Call Stack (most recent call first):
  /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake:127 (append_torchlib_if_found)
  CMakeLists.txt:4 (find_package)


-- Found Torch: /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch.so  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/james/OneDrive/James/CompSci/004_Summer_2024/ML_PINNs/pde/torchtest/example-app

