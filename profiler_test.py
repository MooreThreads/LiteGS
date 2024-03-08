import torch
from gaussian_splatting.model import GaussiansRaster
import numpy as np
from matplotlib import pyplot as plt 
from util import tiles2img_torch

tile_size=16
tiles_num=[82,53]

sorted_pointId=torch.Tensor(np.load("profiler_input_data/sorted_pointId.npy")).cuda().int()
tile_start_index=torch.Tensor(np.load("profiler_input_data/tile_start_index.npy") ).cuda().int()
mean2d=torch.Tensor(np.load("profiler_input_data/mean2d.npy")).cuda().requires_grad_()
cov2d_inv=torch.Tensor(np.load("profiler_input_data/cov2d_inv.npy")).cuda().requires_grad_()
color=torch.Tensor(np.load("profiler_input_data/color.npy") ).cuda().requires_grad_()
opacities=torch.Tensor(np.load("profiler_input_data/opacities.npy") ).cuda().requires_grad_()

tiles=torch.arange(1,tiles_num[0]*tiles_num[1]+1).unsqueeze(0).cuda().int()

img,transmitance=GaussiansRaster.apply(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,tile_size,tiles_num[0],tiles_num[1],840, 1297)
img.mean().backward()
with torch.no_grad():
    img=tiles2img_torch(img,82,53)[:,:,0:840,0:1297]

n=1