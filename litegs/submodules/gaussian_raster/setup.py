import torch; import torch_musa
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension
from torch_musa.utils.simple_porting import SimplePorting
from setuptools import setup

musa_flags = {
    "mcc": ['--offload-arch=mp_31', '-resource-usage','-use_fast_math'],  
}

if __name__ == '__main__':
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name="litegs_fused",
        packages=['litegs_fused'],
        package_dir={'litegs_fused':"."},
        ext_modules=[
            CUDAExtension(
                name="litegs_fused",
                sources=[
                "binning.cu",
                "compact.cu",
                "cuda_errchk.cpp",
                "ext_cuda.cpp",
                "raster.cu",
                "transform.cu"],
                extra_compile_args={
                        'cxx': ['-O3'],
                        'nvcc': ['-O3', '--use_fast_math']
                },
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
