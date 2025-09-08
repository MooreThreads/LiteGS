import torch
import litegs_info
import time
from litegs.utils.statistic_helper import StatisticsHelperInst

gs, cam = torch.load('./profiler_input_data/data.pth')
complex_tile_id=torch.load('./profiler_input_data/complex_tile_2048.pth').int()
sorted_tile_list=torch.load('./profiler_input_data/sorted_tile_list_2048.pth').int()
StatisticsHelperInst.cur_sample="cross_road"
StatisticsHelperInst.cached_complex_tile["cross_road"]=complex_tile_id
StatisticsHelperInst.cached_sorted_tile_list["cross_road"]=sorted_tile_list

# init & warmup
cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity = litegs_info.cluster(
    means=gs['_means'],
    quats=gs['_quats'],
    scales=gs['_scales'],
    opacities=gs['_opacities'].squeeze(),
    colors=gs['_rgbs'],
)
xyz=torch.nn.Parameter(xyz.contiguous())
scale=torch.nn.Parameter(scale.contiguous())
rot=torch.nn.Parameter(rot.contiguous())
sh_0=torch.nn.Parameter(sh_0.contiguous())
opacity=torch.nn.Parameter(opacity.contiguous())
renders, alphas, info = litegs_info.rasterization(
    cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity,
    viewmats=torch.linalg.inv(cam['camera_to_world'])[None, ...],  # [C, 4, 4]
    Ks=cam['intrinsics'][None, ...],  # [C, 3, 3]
    width=cam['width'].item(),
    height=cam['height'].item(),
)
renders.mean().backward()
torch.cuda.synchronize()

# test forward + backward time
start = time.time()
loop_num=20
for _ in range(loop_num):
    renders, alphas, info = litegs_info.rasterization(
        cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity,
        viewmats=torch.linalg.inv(cam['camera_to_world'])[None, ...],  # [C, 4, 4]
        Ks=cam['intrinsics'][None, ...],  # [C, 3, 3]
        width=cam['width'].item(),
        height=cam['height'].item(),
    )
    renders.mean().backward()
    xyz.gard=None
    scale.grad=None
    rot.grad=None
    sh_0.grad=None
    opacity.grad=None
torch.cuda.synchronize()
print('forward&backward: ', (time.time()-start)*1000/loop_num, 'ms')

