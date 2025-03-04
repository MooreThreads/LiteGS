import torch

from ..utils import wrapper

def get_3d_transform_matrix(scale:torch.Tensor,rot:torch.Tensor)->torch.Tensor:
    transform_matrix=wrapper.CreateTransformMatrix.call(scale,rot)    
    return transform_matrix