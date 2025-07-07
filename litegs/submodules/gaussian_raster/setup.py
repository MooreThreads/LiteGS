import torch; import torch_musa
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension
from torch_musa.utils.simple_porting import SimplePorting
from setuptools import setup

setup(
    name="litegs_fused",
    packages=['litegs_fused'],
    package_dir={'litegs_fused':"."},
    ext_modules=[
        MUSAExtension(
            name="litegs_fused",
            sources=[
            "binning.mu",
            "compact.mu",
            "cuda_errchk.cpp",
            "ext_cuda.cpp",
            "raster.mu",
            "transform.mu"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
