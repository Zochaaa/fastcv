import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Ustawienia dla kompilatora C++ (pliki .cpp)
cxx_args = ["/O2", "/std:c++17", "/permissive-"]


# Ustawienia dla NVCC (pliki .cu)
# UWAGA: Flagi dla MSVC musimy przekazać przez -Xcompiler
nvcc_args = [
   "-O2",
   "-std=c++17",
   "--extended-lambda",
   "-D__CUDA_NO_HALF_OPERATORS__",
   "-D__CUDA_NO_HALF_CONVERSIONS__",
   "-D__CUDA_NO_HALF2_OPERATORS__",
   # Przekazanie flag do host compiler (MSVC):
   "-Xcompiler", "/std:c++17",
   "-Xcompiler", "/permissive-",
   "-Xcompiler", "/Zc:preprocessor",
]


setup(
   name="fastcv",
   ext_modules=[
       CUDAExtension(
           name="fastcv",
           sources=[
               "kernels/grayscale.cu",
               "kernels/box_blur.cu",
               "kernels/sobel.cu",
               "kernels/dilation.cu",
               "kernels/erosion.cu",
               "kernels/module.cpp",
		       "kernels/pyrDown.cu"
           ],
           extra_compile_args={
               "cxx": cxx_args,
               "nvcc": nvcc_args,
           },
       ),
   ],
   cmdclass={"build_ext": BuildExtension},
)
