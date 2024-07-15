import torch
from gaussian_splatting.model import GaussianSplattingModel
from gaussian_splatting import wrapper
import numpy as np
from matplotlib import pyplot as plt 
from util import tiles2img_torch,cg_torch
import time

imgH=1036
imgW=1600
tilesize=8
tilesnumX=int(np.ceil(imgW/tilesize))
tilesnumY=int(np.ceil(imgH/tilesize))

if tilesize==8:
    color=torch.Tensor(np.load("profiler_input_data/profiler_input_data_8x8/color.npy")).cuda().requires_grad_()
    cov2d_inv=torch.Tensor(np.load("profiler_input_data/profiler_input_data_8x8/cov2d_inv.npy")).cuda().requires_grad_()
    mean2d=torch.Tensor(np.load("profiler_input_data/profiler_input_data_8x8/mean2d.npy")).cuda().requires_grad_()
    opacities=torch.Tensor(np.load("profiler_input_data/profiler_input_data_8x8/opacities.npy")).cuda().requires_grad_()

    sorted_pointId=torch.tensor(np.load("profiler_input_data/profiler_input_data_8x8/sorted_pointId.npy"),dtype=torch.int32,device='cuda')
    tile_start_index=torch.tensor(np.load("profiler_input_data/profiler_input_data_8x8/tile_start_index.npy"),dtype=torch.int32,device='cuda')
    tiles=torch.tensor(np.load("profiler_input_data/profiler_input_data_8x8/tiles.npy"),dtype=torch.int32,device='cuda')
elif tilesize==16:
    color=torch.Tensor(np.load("profiler_input_data/color.npy")).cuda().requires_grad_()
    cov2d_inv=torch.Tensor(np.load("profiler_input_data/cov2d_inv.npy")).cuda().requires_grad_()
    mean2d=torch.Tensor(np.load("profiler_input_data/mean2d.npy")).cuda().requires_grad_()
    opacities=torch.Tensor(np.load("profiler_input_data/opacities.npy")).cuda().requires_grad_()

    sorted_pointId=torch.tensor(np.load("profiler_input_data/sorted_pointId.npy"),dtype=torch.int32,device='cuda')
    tile_start_index=torch.tensor(np.load("profiler_input_data/tile_start_index.npy"),dtype=torch.int32,device='cuda')
    tiles=torch.tensor(np.load("profiler_input_data/tiles.npy"),dtype=torch.int32,device='cuda')


#check result
img_ord=torch.Tensor(np.load("profiler_input_data/img.npy")).cuda()
transmitance_ord=torch.Tensor(np.load("profiler_input_data/transmitance.npy")).cuda()
color_grad=torch.Tensor(np.load("profiler_input_data/color_grad.npy")).cuda()
cov2d_inv_grad=torch.Tensor(np.load("profiler_input_data/cov2d_inv_grad.npy")).cuda()
mean2d_grad=torch.Tensor(np.load("profiler_input_data/mean2d_grad.npy")).cuda()
opacities_grad=torch.Tensor(np.load("profiler_input_data/opacities_grad.npy")).cuda()




img,trans=wrapper.rasterize_2d_gaussian(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,tilesize,tilesnumX,tilesnumY,imgH,imgW)
img.mean().backward()
with torch.no_grad():
    img=tiles2img_torch(img,tilesnumX,tilesnumY)[...,:imgH,:imgW]
    trans=tiles2img_torch(trans.unsqueeze(2),tilesnumX,tilesnumY)[...,:imgH,:imgW]

print("diff img:            ",(img-img_ord).abs().sum())
print("diff transmitance:   ",(trans-transmitance_ord).abs().sum())
print("diff color_grad:     ",(color.grad-color_grad).abs().max())
print("diff cov2d_inv_grad: ",(cov2d_inv.grad-cov2d_inv_grad).abs().max())
print("diff mean2d_grad:    ",(mean2d.grad-mean2d_grad).abs().max())
print("diff opacities_grad: ",(opacities.grad-opacities_grad).abs().max())


start_time=time.time()
for i in range(100):
    img,trans=wrapper.rasterize_2d_gaussian(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                  tilesize,tilesnumX,tilesnumY,imgH,imgW)
    img.mean().backward()
torch.cuda.synchronize()
end_time=time.time()
print((end_time-start_time)*1000/100,"ms")


