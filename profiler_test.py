import torch
from gaussian_splatting.model import GaussianSplattingModel
from gaussian_splatting import wrapper
import numpy as np
from matplotlib import pyplot as plt 
from util import tiles2img_torch,cg_torch
import time

_xyz=torch.Tensor(np.load("profiler_input_data/xyz.npy")).cuda().requires_grad_()
_scaling=torch.Tensor(np.load("profiler_input_data/scaling.npy")).cuda().requires_grad_()
_rotation=torch.Tensor(np.load("profiler_input_data/rotation.npy")).cuda().requires_grad_()
_opacity=torch.Tensor(np.load("profiler_input_data/opacity.npy")).cuda().requires_grad_()
_features_dc=torch.Tensor(np.load("profiler_input_data/features_dc.npy")).cuda().requires_grad_()

view_matrix=torch.Tensor(np.load("profiler_input_data/view_matrix.npy")).cuda()
view_project_matrix=torch.Tensor(np.load("profiler_input_data/view_project_matrix.npy")).cuda()

start_time=time.time()
for i in range(100):
    with torch.no_grad():
        ndc_pos=cg_torch.world_to_ndc(_xyz,view_project_matrix)
        translated_pos=cg_torch.world_to_view(_xyz,view_matrix)
        visible_points,visible_points_num=GaussianSplattingModel.culling_and_sort(None,ndc_pos,translated_pos)

        scales=_scaling[visible_points].exp()
        rotators=torch.nn.functional.normalize(_rotation[visible_points],dim=-1)
        visible_positions=_xyz[visible_points]
        visible_opacities=_opacity[visible_points].sigmoid()
        visible_sh0=_features_dc[visible_points]
    
torch.cuda.synchronize()
end_time=time.time()
print((end_time-start_time)*1000/100,"ms")


