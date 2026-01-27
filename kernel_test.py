import torch
import litegs_info
import time
from litegs.utils.statistic_helper import StatisticsHelperInst
from litegs.utils.spherical_harmonics import rgb_to_sh0
import math
import matplotlib.pyplot as plt

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

def get_morton_code(coords: torch.Tensor) -> torch.Tensor:
    """
    计算2D坐标的Morton Code (Z-Order Curve value)
    
    Args:
        coords: shape为 [N, 2] 的Tensor，类型为 torch.int16 或 torch.int32
                coords[:, 0] 为 x 坐标
                coords[:, 1] 为 y 坐标
                
    Returns:
        morton_code: shape为 [N] 的Tensor，类型为 torch.int32
    """
    # 1. 必须先转为 int32，因为 int16 展开后会变成 32位，
    #    如果在 int16 下做位移会溢出或截断
    x = coords[:, 0].to(torch.int32)
    y = coords[:, 1].to(torch.int32)
    
    # 2. 定义用于位扩展(Bit Spreading)的掩码 (Magic Numbers)
    # 这些掩码用于将16位整数的位隔开，扩展分布到32位中
    # 过程: abcd -> ..a.b.c.d
    
    # Step 1: 0000000011111111 -> 00000000111111110000000011111111
    # 这一步对于int16输入其实不是必须的，但为了通用性保留
    mask1 = 0x00FF00FF
    mask2 = 0x0F0F0F0F
    mask3 = 0x33333333
    mask4 = 0x55555555

    def spread_bits(v):
        v = (v | (v << 8)) & mask1
        v = (v | (v << 4)) & mask2
        v = (v | (v << 2)) & mask3
        v = (v | (v << 1)) & mask4
        return v

    # 3. 对 x 和 y 分别进行位扩展
    xx = spread_bits(x)
    yy = spread_bits(y)

    # 4. 合并 (y 左移一位，占据奇数位; x 占据偶数位)
    # Morton = ... y1 x1 y0 x0
    morton_code = (yy << 1) | xx

    return morton_code

if __name__ == "__main__":

    gs, cam = torch.load('./profiler_input_data/crossroad.pt')
    sorted_tile_list=torch.load('./profiler_input_data/sorted_tile_list.pt').int()
    StatisticsHelperInst.cur_sample="cross_road"
    StatisticsHelperInst.cached_sorted_tile_list["cross_road"]=sorted_tile_list
    StatisticsHelperInst.cached_heavy_tile["cross_road"]=sorted_tile_list[:512]

    #morton code scheduling
    # tiles_num_x=math.ceil(cam['width']/litegs_info.Config.tile_size[1])
    # tiles_num_y=math.ceil(cam['height']/litegs_info.Config.tile_size[0])
    # coords=torch.meshgrid(torch.arange(tiles_num_x, device='cuda', dtype=torch.int16),torch.arange(tiles_num_y, device='cuda', dtype=torch.int16),indexing='xy')
    # coords=torch.cat((coords[1].unsqueeze(-1),coords[0].unsqueeze(-1)),dim=-1)
    # morton_code=get_morton_code(coords.view(-1,2))
    # _,sorted_indices=morton_code.sort(stable=True)
    # StatisticsHelperInst.cached_sorted_tile_list["cross_road"]=sorted_indices.int()+1# tile index start from 1

    #inverse activate
    opacities = gs['_opacities'].squeeze()
    opacities = torch.log(opacities/(1-opacities))
    colors=gs['_rgbs']
    sh_0=rgb_to_sh0(colors)
    scales=gs['_scales']
    scales=scales.log()
    

    # init & warmup
    cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity = litegs_info.cluster(
        means=gs['_means'],
        quats=gs['_quats'],
        scales=scales,
        opacities=opacities,
        sh_0=sh_0,
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


    loop_num=50

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
    xyz.grad=None
    scale.grad=None
    rot.grad=None
    sh_0.grad=None
    opacity.grad=None
    # xyz_grad,scale_grad,rot_grad,sh_0_grad,opacity_grad=torch.load("./profiler_input_data/cross_road_grad.pth")
    # assert (xyz_grad-xyz.grad).abs().sum()<1e-6
    # assert (scale_grad-scale.grad).abs().sum()<1e-6
    # assert (rot_grad-rot.grad).abs().sum()<1e-6
    # assert (sh_0_grad-sh_0.grad).abs().sum()<1e-6
    # assert (opacity_grad-opacity.grad).abs().sum()<1e-6




    # test forward + backward time
    torch.cuda.synchronize()
    start = time.time()
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
        xyz.grad=None
        scale.grad=None
        rot.grad=None
        sh_0.grad=None
        opacity.grad=None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print('litegs forward&backward: ', (time.time()-start)*1000/loop_num, 'ms')

