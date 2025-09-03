# setup.py
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

ext_modules = [
    CUDAExtension(
        name='attention_cuda',
        sources=[
            'csrc/binding.cpp',
            'csrc/attention_kernel.cu',
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math', '-lineinfo']
        }
    ),
]

if __name__ == "__main__":
    from setuptools import setup
    setup(
        name='attention_cuda',
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )
