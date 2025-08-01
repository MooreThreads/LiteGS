from setuptools import setup, find_packages

from torch.utils.cpp_extension import CUDAExtension, BuildExtension,COMMON_NVCC_FLAGS

def get_cuda_modules():
    from torch.utils.cpp_extension import CUDAExtension
    ext_modules = [
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "litegs/submodules/fused_ssim/ssim.cu",
            "litegs/submodules/fused_ssim/ext.cpp"]),
        CUDAExtension(
                name="simple_knn._C",
                sources=[
                "litegs/submodules/simple-knn/spatial.cu", 
                "litegs/submodules/simple-knn/simple_knn.cu",
                "litegs/submodules/simple-knn/ext.cpp"]),
        CUDAExtension(
                name="litegs_fused",
                sources=[
                "litegs/submodules/gaussian_raster/binning.cu",
                "litegs/submodules/gaussian_raster/compact.cu",
                "litegs/submodules/gaussian_raster/cuda_errchk.cpp",
                "litegs/submodules/gaussian_raster/ext_cuda.cpp",
                "litegs/submodules/gaussian_raster/raster.cu",
                "litegs/submodules/gaussian_raster/transform.cu"])
    ]
    return ext_modules

def get_cmdclass():
    from torch.utils.cpp_extension import BuildExtension
    return {'build_ext': BuildExtension}

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
        name="lite-gaussian-splatting",
        version="1.0.1",
        author="Kaimin Liao",
        author_email="kaiminliao@gmail.com",
        description="A High-Performance Modular Framework for Gaussian Splatting Training",
        long_description_content_type="text/markdown",
        url="https://github.com/MooreThreads/LiteGS",
        packages=find_packages(where=".", include=["litegs", "litegs.*"])+["fused_ssim"],
        package_dir={"litegs": "litegs","fused_ssim":"litegs/submodules/fused_ssim/fused_ssim"},
        setup_requires=["torch","wheel"],
        install_requires=["torch","wheel","numpy","plyfile","tqdm","pillow"],
        ext_modules=get_cuda_modules(),
        include_package_data=True,
        cmdclass=get_cmdclass(),
        python_requires=">=3.8",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    )