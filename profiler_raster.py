import torch
from gaussian_splatting.model import GaussianSplattingModel
from gaussian_splatting import wrapper
import numpy as np
from matplotlib import pyplot as plt 
from util import tiles2img_torch,cg_torch
import time

imgH=1036
imgW=1600
tilesize=16
tilesnumX=int(np.ceil(imgW/tilesize))
tilesnumY=int(np.ceil(imgH/tilesize))

if tilesize==8:
    color=torch.Tensor(np.load("profiler_input_data/8x8/color.npy")).cuda().requires_grad_()
    inv_cov2d=torch.Tensor(np.load("profiler_input_data/8x8/inv_cov2d.npy")).cuda().requires_grad_()
    ndc_pos=torch.Tensor(np.load("profiler_input_data/8x8/ndc_pos.npy")).cuda().requires_grad_()
    opacities=torch.Tensor(np.load("profiler_input_data/8x8/opacities.npy")).cuda().requires_grad_()
    sorted_pointId=torch.tensor(np.load("profiler_input_data/8x8/sorted_pointId.npy"),dtype=torch.int32,device='cuda')
    tile_start_index=torch.tensor(np.load("profiler_input_data/8x8/tile_start_index.npy"),dtype=torch.int32,device='cuda')
    tiles=torch.tensor(np.load("profiler_input_data/8x8/tiles.npy"),dtype=torch.int32,device='cuda')

    grad_color=torch.Tensor(np.load("profiler_input_data/8x8/grad_color.npy")).cuda()
    grad_inv_cov2d=torch.Tensor(np.load("profiler_input_data/8x8/grad_inv_cov2d.npy")).cuda()
    grad_ndc_pos=torch.Tensor(np.load("profiler_input_data/8x8/grad_ndc_pos.npy")).cuda()
    grad_opacities=torch.Tensor(np.load("profiler_input_data/8x8/grad_opacities.npy")).cuda()


elif tilesize==16:
    color=torch.Tensor(np.load("profiler_input_data/16x16/color.npy")).cuda().requires_grad_()
    inv_cov2d=torch.Tensor(np.load("profiler_input_data/16x16/inv_cov2d.npy")).cuda().requires_grad_()
    ndc_pos=torch.Tensor(np.load("profiler_input_data/16x16/ndc_pos.npy")).cuda().requires_grad_()
    opacities=torch.Tensor(np.load("profiler_input_data/16x16/opacities.npy")).cuda().requires_grad_()
    sorted_pointId=torch.tensor(np.load("profiler_input_data/16x16/sorted_pointId.npy"),dtype=torch.int32,device='cuda')
    tile_start_index=torch.tensor(np.load("profiler_input_data/16x16/tile_start_index.npy"),dtype=torch.int32,device='cuda')
    tiles=torch.tensor(np.load("profiler_input_data/16x16/tiles.npy"),dtype=torch.int32,device='cuda')

    grad_color=torch.Tensor(np.load("profiler_input_data/16x16/grad_color.npy")).cuda()
    grad_inv_cov2d=torch.Tensor(np.load("profiler_input_data/16x16/grad_inv_cov2d.npy")).cuda()
    grad_ndc_pos=torch.Tensor(np.load("profiler_input_data/16x16/grad_ndc_pos.npy")).cuda()
    grad_opacities=torch.Tensor(np.load("profiler_input_data/16x16/grad_opacities.npy")).cuda()




img,trans=wrapper.rasterize_2d_gaussian(sorted_pointId,tile_start_index,ndc_pos,inv_cov2d,color,opacities,tiles,tilesize,tilesnumX,tilesnumY,imgH,imgW)
img.mean().backward()

print("diff color_grad:     ",(color.grad-grad_color).abs().max())
print("diff inv_cov2d_grad: ",(inv_cov2d.grad-grad_inv_cov2d).abs().max())
print("diff mean2d_grad:    ",(ndc_pos.grad-grad_ndc_pos).abs().max())
print("diff opacities_grad: ",(opacities.grad-grad_opacities).abs().max())


start_time=time.time()
for i in range(100):
    img,trans=wrapper.rasterize_2d_gaussian(sorted_pointId,tile_start_index,ndc_pos,inv_cov2d,color,opacities,tiles,
                                  tilesize,tilesnumX,tilesnumY,imgH,imgW)
    img.mean().backward()
torch.cuda.synchronize()
end_time=time.time()
print((end_time-start_time)*1000/100,"ms")


