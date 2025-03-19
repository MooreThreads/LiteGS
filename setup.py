from setuptools import setup, find_packages

def get_cuda_modules():
    from torch.utils.cpp_extension import CUDAExtension
    ext_modules = [
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

setup(
    name="litegs",
    version="1.0.0",
    author="Kaimin Liao",
    author_email="kaiminliao@gmail.com",
    description="A High-Performance Modular Framework for Gaussian Splatting Training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MooreThreads/LiteGS",
    packages=find_packages(where=".", include=["litegs", "litegs.*"]),
    package_dir={"litegs": "litegs"},
    setup_requires=["torch","wheel"],
    install_requires=["torch","wheel","numpy","fused_ssim","plyfile","tqdm","pillow"],
    ext_modules=get_cuda_modules(),
    cmdclass=get_cmdclass(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)