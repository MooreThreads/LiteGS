import torch
import math
import typing

from .. import utils
from .. import arguments


@torch.no_grad()
def binning(ndc:torch.Tensor,eigen_val:torch.Tensor,eigen_vec:torch.Tensor,opacity:torch.Tensor,
            img_pixel_shape:tuple[int,int],tile_size:int):

    def craete_2d_AABB(ndc:torch.Tensor,eigen_val:torch.Tensor,eigen_vec:torch.Tensor,opacity:torch.Tensor,tile_size:int,img_pixel_shape:tuple[int,int],img_tile_shape:tuple[int,int]):
        # Major and minor axes -> AABB extensions
        opacity_clamped=opacity.unsqueeze(0).clamp_min(1/255)
        coefficient=2*((255*opacity_clamped).log())#-2*(1/(255*opacity.squeeze(-1))).log()
        axis_length=(coefficient*eigen_val.abs()).sqrt().ceil()
        extension=(axis_length.unsqueeze(-2)*eigen_vec).abs().sum(dim=-3)

        screen_uv=(ndc[:,:2]+1.0)*0.5
        screen_uv[:,0]*=img_pixel_shape[1]#x
        screen_uv[:,1]*=img_pixel_shape[0]#y
        screen_coord=screen_uv-0.5
        b_visible=~((ndc[:,0]<-1.3)|(ndc[:,0]>1.3)|(ndc[:,1]<-1.3)|(ndc[:,1]>1.3)|(ndc[:,2]>1)|(ndc[:,2]<0))
        left_up=((screen_coord-extension)/tile_size).int()*b_visible
        right_down=((screen_coord+extension)/tile_size).ceil().int()*b_visible
        left_up[:,0].clamp_(0,img_tile_shape[1])#x
        left_up[:,1].clamp_(0,img_tile_shape[0])#y
        right_down[:,0].clamp_(0,img_tile_shape[1])
        right_down[:,1].clamp_(0,img_tile_shape[0])

        #splatting area of each points
        rect_length=right_down-left_up
        tiles_touched=rect_length[:,0]*rect_length[:,1]
        b_visible=(tiles_touched!=0)

        return left_up,right_down,tiles_touched,b_visible
    
    img_tile_shape=(int(math.ceil(img_pixel_shape[0]/float(tile_size))),int(math.ceil(img_pixel_shape[1]/float(tile_size))))
    tiles_num=img_tile_shape[0]*img_tile_shape[1]

    left_up,right_down,tiles_touched,b_visible=craete_2d_AABB(ndc,eigen_val,eigen_vec,opacity,tile_size,img_pixel_shape,img_tile_shape)

    #sort by depth
    values,point_ids=ndc[:,2].sort(dim=-1)
    for i in range(ndc.shape[0]):
        tiles_touched[i]=tiles_touched[i,point_ids[i]]

    #calc the item num of table and the start index in table of each point
    prefix_sum=tiles_touched.cumsum(1,dtype=torch.int32)#start index of points
    total_tiles_num_batch=prefix_sum[:,-1]
    allocate_size=total_tiles_num_batch.max().cpu()

    # allocate table and fill it (Table: tile_id-uint16,point_id-uint16)
    large_points_index=(tiles_touched>=32).nonzero()
    my_table=torch.ops.GaussianRaster.duplicateWithKeys(left_up,right_down,prefix_sum,point_ids,large_points_index,int(allocate_size),img_tile_shape[1])
    tileId_table:torch.Tensor=my_table[0]
    pointId_table:torch.Tensor=my_table[1]

    # sort tile_id with torch.sort
    sorted_tileId,indices=torch.sort(tileId_table,dim=1,stable=True)
    sorted_pointId=pointId_table.gather(dim=1,index=indices)

    # range
    tile_start_index=torch.ops.GaussianRaster.tileRange(sorted_tileId,int(allocate_size),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)
        
    return tile_start_index,sorted_pointId,sorted_tileId,b_visible

def render(view_matrix:torch.Tensor,proj_matrix:torch.Tensor,actived_sh_degree:int,
           xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,sh_0:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor,
           output_shape,pp:arguments.PipelineParams)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    #activate
    #!!!!!!!!!!!todo xyz pad 1!!!!!!!!!!!!!!!!!!!
    pad_one=torch.ones((1,xyz.shape[-1]),dtype=xyz.dtype,device=xyz.device)
    xyz=torch.concat((xyz,pad_one),dim=0)
    scale=scale.exp()
    rot=torch.nn.functional.normalize(rot,dim=0)
    opacity=opacity.sigmoid()

    #gs projection
    transform_matrix=utils.wrapper.CreateTransformMatrix.call_fused(scale,rot)
    J=utils.wrapper.CreateRaySpaceTransformMatrix.call_script(xyz,view_matrix,proj_matrix,False)#todo script
    cov2d=utils.wrapper.CreateCov2dDirectly.call_fused(J,view_matrix,transform_matrix)
    eigen_val,eigen_vec,inv_cov2d=utils.wrapper.EighAndInverse2x2Matrix.call_fused(cov2d)
    ndc_pos=utils.wrapper.World2NdcFunc.apply(xyz,view_matrix@proj_matrix)

    #color
    camera_center=(-view_matrix[...,3:4,:3]@(view_matrix[...,:3,:3].transpose(-1,-2))).squeeze(1)
    dirs=xyz[:3]-camera_center.unsqueeze(-1)
    dirs=torch.nn.functional.normalize(dirs,dim=-2)
    color=utils.wrapper.SphericalHarmonicToRGB.call_fused(actived_sh_degree,sh_0,sh_rest,dirs)
    
    #visibility table
    tile_start_index,sorted_pointId,sorted_tileId,b_visible=binning(ndc_pos,eigen_val,eigen_vec,opacity,output_shape,pp.tile_size)
    if StatisticsHelperInst.bStart:
        StatisticsHelperInst.update_visible_count(b_visible)

    #rasterization
    if tiles is None:
        tiles=self.cached_tiles_map
    if StatisticsHelperInst.bStart and ndc_pos.requires_grad:
        def gradient_wrapper(tensor:torch.Tensor) -> torch.Tensor:
            return tensor[:,:2].norm(dim=1)
        StatisticsHelperInst.register_tensor_grad_callback('mean2d_grad',ndc_pos,StatisticsHelper.update_mean_std_compact,gradient_wrapper)
    img,transmitance=utils.wrapper.GaussiansRasterFunc.apply(sorted_pointId,tile_start_index,ndc_pos,inv_cov2d,color,opacity,tiles,
                                            pp.tile_size,self.cached_tiles_size[0],self.cached_tiles_size[1],output_shape[0],output_shape[1])

    img=None
    transmitance=None
    depth=None
    normal=None
    return img,transmitance,depth,normal
