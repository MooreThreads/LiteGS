import torch;import torch_musa
import litegs_info
import time
from litegs.utils.statistic_helper import StatisticsHelperInst
import math

def get_frustumplane(view_matrix,proj_matrix):
    viewproj_matrix=view_matrix@proj_matrix
    frustumplane=torch.zeros((6,4), device=view_matrix.device)
    #left plane
    frustumplane[0,0]=viewproj_matrix[0,3]+viewproj_matrix[0,0]
    frustumplane[0,1]=viewproj_matrix[1,3]+viewproj_matrix[1,0]
    frustumplane[0,2]=viewproj_matrix[2,3]+viewproj_matrix[2,0]
    frustumplane[0,3]=viewproj_matrix[3,3]+viewproj_matrix[3,0]
    #right plane
    frustumplane[1,0]=viewproj_matrix[0,3]-viewproj_matrix[0,0]
    frustumplane[1,1]=viewproj_matrix[1,3]-viewproj_matrix[1,0]
    frustumplane[1,2]=viewproj_matrix[2,3]-viewproj_matrix[2,0]
    frustumplane[1,3]=viewproj_matrix[3,3]-viewproj_matrix[3,0]

    #bottom plane
    frustumplane[2,0]=viewproj_matrix[0,3]+viewproj_matrix[0,1]
    frustumplane[2,1]=viewproj_matrix[1,3]+viewproj_matrix[1,1]
    frustumplane[2,2]=viewproj_matrix[2,3]+viewproj_matrix[2,1]
    frustumplane[2,3]=viewproj_matrix[3,3]+viewproj_matrix[3,1]

    #top plane
    frustumplane[3,0]=viewproj_matrix[0,3]-viewproj_matrix[0,1]
    frustumplane[3,1]=viewproj_matrix[1,3]-viewproj_matrix[1,1]
    frustumplane[3,2]=viewproj_matrix[2,3]-viewproj_matrix[2,1]
    frustumplane[3,3]=viewproj_matrix[3,3]-viewproj_matrix[3,1]

    #near plane
    frustumplane[4,0]=viewproj_matrix[0,2]
    frustumplane[4,1]=viewproj_matrix[1,2]
    frustumplane[4,2]=viewproj_matrix[2,2]
    frustumplane[4,3]=viewproj_matrix[3,2]

    #far plane
    frustumplane[5,0]=viewproj_matrix[0,3]-viewproj_matrix[0,2]
    frustumplane[5,1]=viewproj_matrix[1,3]-viewproj_matrix[1,2]
    frustumplane[5,2]=viewproj_matrix[2,3]-viewproj_matrix[2,2]
    frustumplane[5,3]=viewproj_matrix[3,3]-viewproj_matrix[3,2]
    return frustumplane

def get_projection_matrix(znear, zfar, fovX, fovY, device="musa"):
    """Create OpenGL-style projection matrix"""
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

if __name__ == "__main__":

    gs, cam = torch.load('./profiler_input_data/data.pth',map_location=torch.device('musa'))
    complex_tile_id=torch.load('./profiler_input_data/complex_tile_2048.pth',map_location=torch.device('musa')).int()
    sorted_tile_list=torch.load('./profiler_input_data/sorted_tile_list_2048.pth',map_location=torch.device('musa')).int()
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

    W = cam['width']#int(max(Ks[0, 0, 2], width-Ks[0, 0, 2])*2)
    H = cam['height']#int(max(Ks[0, 1, 2], height-Ks[0, 1, 2])*2)
    Ks = cam['intrinsics'][None, ...]
    
    FoVx = 2 * math.atan(W / (2 * Ks[0, 0, 0].item()))
    FoVy = 2 * math.atan(H / (2 * Ks[0, 1, 1].item()))
    
    viewmats=torch.linalg.inv(cam['camera_to_world'])[None, ...]
    view_matrix = viewmats[0].transpose(0, 1)
    near_plane: float = 0.01
    far_plane: float = 100.0
    proj_matrix = get_projection_matrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=xyz.device
    ).transpose(0, 1)
    frustumplane=get_frustumplane(view_matrix, proj_matrix)


    #warm up 
    renders, alphas, info = litegs_info.rasterization(
        cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity,
        view_matrix=view_matrix,  # [C, 4, 4]
        proj_matrix=proj_matrix,
        frustumplane=frustumplane,
        Ks=Ks,  # [C, 3, 3]
        width=W,
        height=H)
    renders.mean().backward()
    # xyz_grad,scale_grad,rot_grad,sh_0_grad,opacity_grad=torch.load("./profiler_input_data/cross_road_grad.pth")
    # assert (xyz_grad-xyz.grad).abs().sum()<1e-6
    # assert (scale_grad-scale.grad).abs().sum()<1e-6
    # assert (rot_grad-rot.grad).abs().sum()<1e-6
    # assert (sh_0_grad-sh_0.grad).abs().sum()<1e-6
    # assert (opacity_grad-opacity.grad).abs().sum()<1e-6




    # test forward + backward time
    torch.musa.synchronize()
    start = time.time()
    loop_num=40
    viewmats=torch.linalg.inv(cam['camera_to_world'])[None, ...]
    for _ in range(loop_num):
        renders, alphas, info = litegs_info.rasterization(
            cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity,
            view_matrix=view_matrix,  # [C, 4, 4]
            proj_matrix=proj_matrix,
            frustumplane=frustumplane,
            Ks=Ks,  # [C, 3, 3]
            width=W,
            height=H)
        renders.mean().backward()
        xyz.gard=None
        scale.grad=None
        rot.grad=None
        sh_0.grad=None
        opacity.grad=None
    torch.musa.synchronize()
    print('litegs forward&backward: ', (time.time()-start)*1000/loop_num, 'ms')

