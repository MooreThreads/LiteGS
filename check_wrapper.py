import torch
from gaussian_splatting import wrapper


for wrapper_class in wrapper.BaseWrapper.__subclasses__():
    wrapper_class.validate()
