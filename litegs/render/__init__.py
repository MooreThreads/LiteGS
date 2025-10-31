import torch
import math
import typing
import torch.cuda.nvtx as nvtx

from .. import utils
from ..utils.statistic_helper import StatisticsHelperInst,StatisticsHelper
from .. import arguments
from .. import scene

def render_preprocess(cluster_origin:torch.Tensor|None,cluster_extend:torch.Tensor|None,frustumplane:torch.Tensor,view_matrix:torch.Tensor,
                      xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,sh_0:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor,
                      op:arguments.OptimizationParams,pp:arguments.PipelineParams,actived_sh_degree:int):

    if pp.cluster_size:
        if cluster_origin is None or cluster_extend is None:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))

        if pp.sparse_grad:#enable sparse gradient
            visible_chunkid,culled_xyz,culled_scale,culled_rot,color,culled_opacity=utils.wrapper.CullCompactActivateWithSparseGrad.apply(
                cluster_origin,cluster_extend,frustumplane,view_matrix,actived_sh_degree,xyz,scale,rot,sh_0,sh_rest,opacity)
            culled_xyz,culled_scale,culled_rot,color,culled_opacity=scene.cluster.uncluster(culled_xyz,culled_scale,culled_rot,color,culled_opacity)  
            if StatisticsHelperInst.bStart:
                StatisticsHelperInst.set_compact_mask(visible_chunkid)
            return visible_chunkid,culled_xyz,culled_scale,culled_rot,color,culled_opacity
        else:
            visibility,visible_num,visible_chunkid=utils.wrapper.litegs_fused.frustum_culling_aabb_cuda(cluster_origin,cluster_extend,frustumplane)
            visible_chunkid=visible_chunkid[:visible_num]
            culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=scene.cluster.culling(visible_chunkid,xyz,scale,rot,sh_0,sh_rest,opacity)
            culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=scene.cluster.uncluster(culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity)

        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.set_compact_mask(visible_chunkid)
    else:
        culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=xyz,scale,rot,sh_0,sh_rest,opacity
        visible_chunkid=None

    nvtx.range_push("Activate")
    pad_one=torch.ones((1,culled_xyz.shape[-1]),dtype=culled_xyz.dtype,device=culled_xyz.device)
    culled_xyz=torch.concat((culled_xyz,pad_one),dim=0)
    culled_scale=culled_scale.exp()
    culled_rot=torch.nn.functional.normalize(culled_rot,dim=0)
    culled_opacity=culled_opacity.sigmoid()
    with torch.no_grad():
        camera_center=(-view_matrix[...,3:4,:3]@(view_matrix[...,:3,:3].transpose(-1,-2))).squeeze(1)
        dirs=culled_xyz[:3]-camera_center.unsqueeze(-1)
        dirs=torch.nn.functional.normalize(dirs,dim=-2)
    color=utils.wrapper.SphericalHarmonicToRGB.call_fused(actived_sh_degree,culled_sh_0,culled_sh_rest,dirs)
    nvtx.range_pop()

    return visible_chunkid,culled_xyz,culled_scale,culled_rot,color,culled_opacity

def render(view_matrix:torch.Tensor,proj_matrix:torch.Tensor,
           xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,color:torch.Tensor,opacity:torch.Tensor,
           actived_sh_degree:int,output_shape:tuple[int,int],pp:arguments.PipelineParams)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    #gs projection
    nvtx.range_push("Proj")
    transform_matrix=utils.wrapper.CreateTransformMatrix.call_fused(scale,rot)
    J=utils.wrapper.CreateRaySpaceTransformMatrix.call_fused(xyz,view_matrix,proj_matrix,output_shape,False)#todo script
    cov2d=utils.wrapper.CreateCov2dDirectly.call_fused(J,view_matrix,transform_matrix)
    eigen_val,eigen_vec,inv_cov2d=utils.wrapper.EighAndInverse2x2Matrix.call_fused(cov2d)
    #ndc_pos=utils.wrapper.World2NdcFunc.apply(xyz,view_matrix@proj_matrix)
    hom_pos=(xyz.transpose(0,1)@(view_matrix@proj_matrix)).transpose(1,2).contiguous()
    ndc_pos=hom_pos/(hom_pos[:,3:4,:]+1e-7)

    view_depth=(view_matrix.transpose(1,2)@xyz)[:,2]
    nvtx.range_pop()
    
    #visibility table
    tile_start_index,sorted_pointId,primitive_visible=utils.wrapper.Binning.call_fused(ndc_pos,view_depth,inv_cov2d,opacity,output_shape,pp.tile_size)

    #raster
    tiles_x=int(math.ceil(output_shape[1]/float(pp.tile_size[1])))
    tiles_y=int(math.ceil(output_shape[0]/float(pp.tile_size[0])))
    tiles=None
    try:
        tiles=StatisticsHelperInst.cached_sorted_tile_list[StatisticsHelperInst.cur_sample].unsqueeze(0)
    except:
        pass
    img,transmitance,depth,normal,lst_contributor=utils.wrapper.GaussiansRasterFunc.apply(sorted_pointId,tile_start_index,ndc_pos,inv_cov2d,color,opacity,tiles,
                                            output_shape[0],output_shape[1],pp.tile_size[0],pp.tile_size[1],pp.enable_transmitance,pp.enable_depth)
    
    if StatisticsHelperInst.bStart:
        StatisticsHelperInst.update_tile_blend_count(lst_contributor)


    img=utils.tiles2img_torch(img,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    if transmitance is not None:
        transmitance=utils.tiles2img_torch(transmitance,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    if depth is not None:
        depth=utils.tiles2img_torch(depth,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    if normal is not None:
        normal=utils.tiles2img_torch(normal,tiles_x,tiles_y)[...,:output_shape[0],:output_shape[1]].contiguous()
    return img,transmitance,depth,normal,primitive_visible
