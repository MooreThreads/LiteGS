import platform
import torch
plat = platform.system().lower()

#torch.compile
if plat == 'windows':
    def empty_compile(model,*args,**kwargs):
        if model is None:
            def empty_decorator(func):
                return func
            return empty_decorator
        return model
    platform_torch_compile=empty_compile
elif plat == 'linux':
    platform_torch_compile=torch.compile


#load dynamic library
def load_dynamic_lib():
    if plat == 'windows':
        torch.ops.load_library("gaussian_splatting/submodules/gaussian_raster/build/Release/GaussianRaster.dll")
    elif plat == 'linux':
        torch.ops.load_library("gaussian_splatting/submodules/gaussian_raster/build/libGaussianRaster.so")