from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension,COMMON_NVCC_FLAGS

def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

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
