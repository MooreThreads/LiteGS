import torch; import torch_musa
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension
from torch_musa.utils.simple_porting import SimplePorting
from setuptools import setup

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        MUSAExtension(
            name="fused_ssim_musa",
            sources=[
                "ssim.mu",
                "ext.cpp"],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

