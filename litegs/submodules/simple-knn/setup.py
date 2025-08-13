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

import torch; import torch_musa
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension
from torch_musa.utils.simple_porting import SimplePorting
from setuptools import setup

cxx_compiler_flags = []

setup(
    name="simple_knn",
    ext_modules=[
        MUSAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.mu", 
            "simple_knn.mu",
            "ext.cpp"],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
