import torch
from gaussian_splatting.model import GaussiansRaster
import numpy as np

tile_size=16
tiles_num=[82,53]

sorted_pointId=torch.Tensor(np.load("profiler_input_data/sorted_pointId.npy")).cuda().int()
tile_start_index=torch.Tensor(np.load("profiler_input_data/tile_start_index.npy") ).cuda().int()
mean2d=torch.Tensor(np.load("profiler_input_data/mean2d.npy")).cuda().requires_grad_()
cov2d_inv=torch.Tensor(np.load("profiler_input_data/cov2d_inv.npy")).cuda().requires_grad_()
color=torch.Tensor(np.load("profiler_input_data/color.npy") ).cuda().requires_grad_()
opacities=torch.Tensor(np.load("profiler_input_data/opacities.npy") ).cuda().requires_grad_()

img,transmitance=GaussiansRaster.apply(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tile_size,tiles_num[0],tiles_num[1])
img.mean().backward()

n=1