from setuptools import setup
import os.path as osp

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = None
exec(open('rasterbinning/version.py', 'r').read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

try:
    ext_modules = [
        CUDAExtension('rasterbinning._C', [
            'rasterbinning/ext.cpp',
            'rasterbinning/rasterbinning.cu',
        ], include_dirs=[osp.join(ROOT_DIR, "rasterbinning")],
        optional=True),
    ]
except:
    import warnings
    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

setup(
    name='rasterbinning',
    version=__version__,
    author='Kaimin Liao',
    author_email='314863230@qq.com',
    description='raster binning using CUDA for the training of 3d-gaussian splatting',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
