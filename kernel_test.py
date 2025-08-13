import torch
import litegs_info
import time
gs, cam = torch.load('./profiler_input_data/data.pth')

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
xyz_grad,scale_grad,rot_grad,sh_0_grad,opacity_grad=torch.load("./profiler_input_data/cross_road_grad.pth")
assert (xyz_grad-xyz.grad).abs().sum()<1e-6
assert (scale_grad-scale.grad).abs().sum()<1e-6
assert (rot_grad-rot.grad).abs().sum()<1e-6
assert (sh_0_grad-sh_0.grad).abs().sum()<1e-6
assert (opacity_grad-opacity.grad).abs().sum()<1e-6
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
torch.musa.synchronize()
print('forward&backward: ', (time.time()-start)*1000/loop_num, 'ms')

