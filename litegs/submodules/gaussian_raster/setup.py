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
    # The rasterizer uses half2(float,float) ctors and half2 += operators
    # throughout; torch's HIP build prepends __HIP_NO_HALF_OPERATORS__ /
    # __HIP_NO_HALF_CONVERSIONS__ which disable exactly those, so strip them on
    # ROCm (mirrors the nvcc removals above for the HIP toolchain).
    import torch
    if torch.version.hip:
        from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS
        for flag in ['-D__HIP_NO_HALF_OPERATORS__=1', '-D__HIP_NO_HALF_CONVERSIONS__=1']:
            try:
                COMMON_HIPCC_FLAGS.remove(flag)
            except ValueError:
                pass

if __name__ == '__main__':
    remove_unwanted_pytorch_nvcc_flags()
    import torch
    import shutil
    if torch.version.hip:
        # hipcc/clang rejects nvcc's --use_fast_math; -ffast-math is the HIP
        # equivalent (matches the gsplat/fused-ssim ROCm builds).
        nvcc_args = ['-O3', '-ffast-math']
    else:
        nvcc_args = ['-O3', '--use_fast_math']
    # On Windows with ROCm/HIP, MSVC cl.exe cannot link the inherited c10::ValueError
    # constructor from a Clang-built c10.dll.  Route the .cpp wrappers through
    # hipcc/amdclang by giving them a .cu extension.  The shims are byte-identical
    # to the originals; *_winhip.cu is already gitignored.
    import os
    cpp_sources = ["cuda_errchk.cpp", "ext_cuda.cpp"]
    if os.name == 'nt' and torch.version.hip:
        for src in cpp_sources:
            shutil.copy(src, src.replace('.cpp', '_winhip.cu'))
        cpp_sources = [s.replace('.cpp', '_winhip.cu') for s in cpp_sources]
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
                ] + cpp_sources + [
                "raster.cu",
                "transform.cu"],
                extra_compile_args={
                        'cxx': ['-O3'],
                        'nvcc': nvcc_args
                },
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
