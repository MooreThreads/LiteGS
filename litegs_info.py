import sys
import torch#; import torch_musa
from torch import Tensor
import math

import litegs
from litegs.utils.statistic_helper import StatisticsHelperInst

class Config:
    cluster_size = 128
    sparse_grad = False
    tile_size = (16, 8)
    enable_transmitance = True
    enable_depth = False    

def cluster(means,quats,scales,opacities,colors):
    means,quats,scales,opacities,colors=litegs.scene.point.spatial_refine(False,None,means.T,quats.T,scales.T,opacities.T,colors.T)
    opacities = torch.log(opacities/(1-opacities))
    xyz,scale,rot,sh_0,opacity=litegs.scene.cluster.cluster_points(128,means,scales.log(),quats,colors[None],opacities[None])
    cluster_origin,cluster_extend=litegs.scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    return cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity
    
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

def get_projection_matrix(znear, zfar, fovX, fovY, device="cuda"):
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

def rasterization(
    cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity,
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
):
    pp = Config()
    pp.input_color_type='rgb'
    # StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],lambda x: True)
    # opacities = torch.log(opacities/(1-opacities))
    # xyz,scale,rot,sh_0,opacity=litegs.scene.cluster.cluster_points(128,means.T.contiguous(),scales.log().T.contiguous(),quats.T.contiguous(),colors.T.contiguous()[None],opacities[None])
    # cluster_origin,cluster_extend=litegs.scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    C = len(viewmats)
    assert C == 1
    W = width#int(max(Ks[0, 0, 2], width-Ks[0, 0, 2])*2)
    H = height#int(max(Ks[0, 1, 2], height-Ks[0, 1, 2])*2)
    
    FoVx = 2 * math.atan(W / (2 * Ks[0, 0, 0].item()))
    FoVy = 2 * math.atan(H / (2 * Ks[0, 1, 1].item()))
    
    view_matrix = viewmats[0].transpose(0, 1)
    proj_matrix = get_projection_matrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=xyz.device
    ).transpose(0, 1)
    
    with StatisticsHelperInst.try_start(0):
        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=litegs.render.render_preprocess(cluster_origin,cluster_extend,get_frustumplane(view_matrix, proj_matrix)[None],
                                                                                                xyz,scale,rot,sh_0,torch.zeros(0,*xyz.shape,device=xyz.device),opacity,None,pp)        
        render_colors_,render_alphas_,depth,normal,_=litegs.render.render(view_matrix[None],proj_matrix[None],culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,0,(H, W),pp)

    return render_colors_[None], render_alphas_[None], {'info':StatisticsHelperInst}