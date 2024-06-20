import torch
from gaussian_splatting.model import GaussianSplattingModel
from gaussian_splatting import wrapper
import numpy as np
from matplotlib import pyplot as plt 
from util import tiles2img_torch,cg_torch
import time

color=torch.Tensor(np.load("profiler_input_data/color.npy")).cuda().requires_grad_()
cov2d_inv=torch.Tensor(np.load("profiler_input_data/cov2d_inv.npy")).cuda().requires_grad_()
mean2d=torch.Tensor(np.load("profiler_input_data/mean2d.npy")).cuda().requires_grad_()
opacities=torch.Tensor(np.load("profiler_input_data/opacities.npy")).cuda().requires_grad_()

sorted_pointId=torch.tensor(np.load("profiler_input_data/sorted_pointId.npy"),dtype=torch.int32,device='cuda')
tile_start_index=torch.tensor(np.load("profiler_input_data/tile_start_index.npy"),dtype=torch.int32,device='cuda')
tiles=torch.tensor(np.load("profiler_input_data/tiles.npy"),dtype=torch.int32,device='cuda')

start_time=time.time()
for i in range(100):
    img,trans=wrapper.rasterize_2d_gaussian(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                  16,100,65,1036,1600)
    img.mean().backward()
torch.cuda.synchronize()
end_time=time.time()
print((end_time-start_time)*1000/100,"ms")


