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
    

def rasterization(
    cluster_origin,cluster_extend,xyz,scale,rot,sh_0,opacity,
    view_matrix: Tensor,  # [C, 4, 4]
    proj_matrix: Tensor,
    frustumplane: Tensor,
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
):
    pp = Config()
    pp.input_color_type='rgb'

    W = width#int(max(Ks[0, 0, 2], width-Ks[0, 0, 2])*2)
    H = height#int(max(Ks[0, 1, 2], height-Ks[0, 1, 2])*2)
    
    with StatisticsHelperInst.try_start(0):
        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=litegs.render.render_preprocess(cluster_origin,cluster_extend,frustumplane[None],
                                                                                                xyz,scale,rot,sh_0,torch.zeros(0,*xyz.shape,device=xyz.device),opacity,None,pp)        
        render_colors_,render_alphas_,depth,normal,_=litegs.render.render(view_matrix[None],proj_matrix[None],culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,0,(H, W),pp)

    return render_colors_[None], render_alphas_[None], {'info':StatisticsHelperInst}