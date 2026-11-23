#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import shutil

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

import torch
# On Windows with ROCm/HIP, MSVC cl.exe cannot link the inherited c10::ValueError
# constructor from a Clang-built c10.dll.  Route the pybind11 wrapper through
# hipcc/amdclang by giving it a .cu extension so torch's BuildExtension picks the
# HIP toolchain for it.  The shim is byte-identical to the original .cpp; the
# .gitignore already lists *_winhip.cu so it is not committed.
sources = ["spatial.cu", "simple_knn.cu", "ext.cpp"]
if os.name == 'nt' and torch.version.hip:
    shutil.copy("ext.cpp", "ext_winhip.cu")
    sources = ["spatial.cu", "simple_knn.cu", "ext_winhip.cu"]

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=sources,
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
