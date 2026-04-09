import torch
import math
import typing
import torch.cuda.nvtx as nvtx

from .. import utils
from ..utils.statistic_helper import StatisticsHelperInst,StatisticsHelper
from .. import arguments
from .. import scene
from ..data import FramesBuffer

def render_preprocess(
    cluster_origin:torch.Tensor|None,cluster_extend:torch.Tensor|None,frustumplane:torch.Tensor,view_matrix:torch.Tensor,
    xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,sh_0:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor,
    feedback_buffer:torch.Tensor|None,idx_tensor:torch.Tensor|None,
    bCluster,bSparseGrad,actived_sh_degree:int
):

    visible_chunkid=None
    visible_chunks_num=None
    
    if bCluster:
        if cluster_origin is None or cluster_extend is None:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))

        visibility,visible_chunks_num,visible_chunkid=utils.ops.frustum_culling_aabb(cluster_origin,cluster_extend,frustumplane,feedback_buffer,idx_tensor)
        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.set_compact_mask(visible_chunkid,visible_chunks_num)

        # Step 1: Compact + Activate (without SH)
        culled_xyz,culled_scale,culled_rot,culled_opacity=utils.ops.compact_activate_nosh(
            bSparseGrad,
            visible_chunkid,visible_chunks_num,
            xyz,scale,rot,opacity
        )

        # Step 2: Compact + SH (using activated position for view direction)
        color=utils.ops.compact_sh(
            bSparseGrad,
            actived_sh_degree,
            visible_chunkid,visible_chunks_num,
            view_matrix,
            xyz,sh_0,sh_rest
        )

        culled_xyz,culled_scale,culled_rot,color,culled_opacity=scene.cluster.uncluster(culled_xyz,culled_scale,culled_rot,color,culled_opacity)  
    else:
        nvtx.range_push("Activate")
        pad_one=torch.ones((1,xyz.shape[-1]),dtype=xyz.dtype,device=xyz.device)
        culled_xyz=torch.concat((xyz,pad_one),dim=0)
        culled_scale=scale.exp()
        culled_rot=torch.nn.functional.normalize(rot,dim=0)
        culled_opacity=opacity.sigmoid()
        with torch.no_grad():
            camera_center=(-view_matrix[...,3:4,:3]@(view_matrix[...,:3,:3].transpose(-1,-2))).squeeze(1)
            dirs=culled_xyz[:3]-camera_center.unsqueeze(-1)
            dirs=torch.nn.functional.normalize(dirs,dim=-2)
        color=utils.ops.spherical_harmonic_to_rgb(actived_sh_degree,sh_0,sh_rest,dirs)
        nvtx.range_pop()


    return visible_chunkid,visible_chunks_num,culled_xyz,culled_scale,culled_rot,color,culled_opacity

def render(
    view_matrix:torch.Tensor,proj_matrix:torch.Tensor,
    xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,color:torch.Tensor,opacity:torch.Tensor,
    valid_length:torch.Tensor|None,training_frame_buffer:FramesBuffer|None,idx_tensor:torch.Tensor|None,
    output_shape:tuple[int,int],pp:arguments.PipelineParams
)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    #gs projection
    nvtx.range_push("Proj")
    view_pos,ndc_pos=utils.ops.mvp_transform(xyz,view_matrix,proj_matrix,valid_length)
    transform_matrix=utils.ops.create_transform_matrix(scale,rot,valid_length=valid_length)
    J=utils.ops.create_rayspace_transform_matrix(view_pos,proj_matrix,output_shape,valid_length=valid_length)
    cov2d=utils.ops.create_cov2d_directly(J,view_matrix,transform_matrix,valid_length)
    eigen_val,eigen_vec,inv_cov2d=utils.ops.eigh_and_inverse_2x2_matrix(cov2d,valid_length)
    view_depth=view_pos[:,2,:]
    nvtx.range_pop()
    
    #visibility table
    feedback_binning_allocate_size = None
    if training_frame_buffer is not None:
        feedback_binning_allocate_size=training_frame_buffer.feedback_binning_allocate_size
    tile_start_index,sorted_pointId,primitive_visible=utils.ops.binning(
        ndc_pos,view_depth,inv_cov2d,opacity,
        valid_length,feedback_binning_allocate_size,idx_tensor,
        output_shape,pp.tile_size
    )

    #raster
    tiles=None
    if training_frame_buffer is not None:
        views_num=idx_tensor.shape[0]
        temp_list=[]
        try:
            for view_i in range(views_num):
                temp_list.append(training_frame_buffer.cache_sorted_tile_list[int(idx_tensor[view_i])].unsqueeze(0))
            tiles=torch.cat(temp_list,dim=0)
        except:
            pass

    img,transmitance,depth,normal,lst_contributor=utils.ops.rasterize_gaussians(
        sorted_pointId,tile_start_index,
        ndc_pos,inv_cov2d,color,opacity,
        tiles,
        output_shape[0],output_shape[1],pp.tile_size[0],pp.tile_size[1],
        pp.enable_transmitance,pp.enable_depth
    )
    
    if StatisticsHelperInst.bStart and training_frame_buffer is not None:
        training_frame_buffer.update_tile_blend_count(lst_contributor,idx_tensor,pp.tile_size[0],pp.tile_size[1])


    img=img[...,:output_shape[0],:output_shape[1]].clamp(0,1).contiguous()
    if transmitance is not None:
        transmitance=transmitance[...,:output_shape[0],:output_shape[1]].contiguous()
    if depth is not None:
        depth=depth[...,:output_shape[0],:output_shape[1]].contiguous()
    if normal is not None:
        normal=normal[...,:output_shape[0],:output_shape[1]].contiguous()
    return img,transmitance,depth,normal,primitive_visible
