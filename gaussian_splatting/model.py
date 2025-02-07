import torch
import typing
import numpy as np
import math

from gaussian_splatting.scene import GaussianScene
from util import cg_torch
from gaussian_splatting import wrapper
from util.statistic_helper import StatisticsHelperInst,StatisticsHelper
from util.platform import platform_torch_compile
from util.cg_torch import gen_morton_code

class GaussianSplattingModel:
    @torch.no_grad()
    def __init__(self,scene:GaussianScene,actived_sh_degree:int,chunk_size=128):
        assert(chunk_size>0)
        self.actived_sh_degree=actived_sh_degree
        self.chunk_size=chunk_size
        self.chunk_AABB_origin=None
        self.chunk_AABB_extend=None
        self.b_split_into_chunk=False
        
        if scene is not None:
            self.load_from_scene(scene)
        else:
            self.max_sh_degree=0
        return
    
    @torch.no_grad()
    def load_from_scene(self,scene:GaussianScene):
        self._xyz = torch.nn.Parameter(torch.Tensor(np.pad(scene.position,((0,0),(0,1)),'constant',constant_values=1)).transpose(0,1).cuda())
        self._features_dc = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_dc).transpose(0, -1).cuda())
        self._features_rest = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_rest).transpose(0, -1).cuda())
        self._rotation = torch.nn.Parameter(torch.Tensor(scene.rotator).transpose(0,1).cuda())#torch.nn.functional.normalize
        self._scaling = torch.nn.Parameter(torch.Tensor(scene.scale).transpose(0,1).cuda())#.exp 
        self._opacity = torch.nn.Parameter(torch.Tensor(scene.opacity).transpose(0,1).cuda())#.sigmoid
        self.max_sh_degree=scene.sh_degree
        self.__split_into_chunk()
        self.rebuild_AABB()
        assert(self.actived_sh_degree<=self.max_sh_degree)#mismatch
        return
    

    @torch.no_grad()
    def save_to_scene(self,scene:GaussianScene):
        if self.b_split_into_chunk:
            self.__concate_param_chunks()
        scene.position=self._xyz.permute(1,0)[...,:3].cpu().numpy()
        scene.sh_coefficient_dc=self._features_dc.permute(2,1,0).cpu().numpy()
        scene.sh_coefficient_rest=self._features_rest.permute(2,1,0).cpu().numpy()
        scene.rotator=self._rotation.permute(1,0).cpu().numpy()
        scene.scale=self._scaling.permute(1,0).cpu().numpy()
        scene.opacity=self._opacity.permute(1,0).cpu().numpy()
        scene.sh_degree=self.max_sh_degree
        if self.b_split_into_chunk==False:#recover
            self.__split_into_chunk()
        return
    
    @torch.no_grad()
    def __split_into_chunk(self):
        assert(self.b_split_into_chunk==False)

        morton_code=gen_morton_code(self._xyz.transpose(0,1).contiguous()[:,:3])
        _,index=morton_code.sort()

        padding_num=morton_code.shape[0]%self.chunk_size
        if padding_num!=0:
            padding_num=self.chunk_size-padding_num
            index=torch.concat([index,index[-padding_num:]])

        def reorder_parameters_and_split(tensor:torch.Tensor,index:torch.Tensor)->torch.Tensor:
            assert(index.shape[0]%self.chunk_size==0)
            chunks_num=int(index.shape[0]/self.chunk_size)
            chunk_tensor=tensor[...,index].reshape(*tensor.shape[:-1],chunks_num,self.chunk_size)
            return chunk_tensor
        xyz_chunk=reorder_parameters_and_split(self._xyz,index)
        features_dc_chunk=reorder_parameters_and_split(self._features_dc,index)
        features_rest_chunk=reorder_parameters_and_split(self._features_rest,index)
        opacity_chunk=reorder_parameters_and_split(self._opacity,index)
        scaling_chunk=reorder_parameters_and_split(self._scaling,index)
        rotation_chunk=reorder_parameters_and_split(self._rotation,index)
    
        self._xyz=torch.nn.Parameter(xyz_chunk.contiguous())
        self._features_dc=torch.nn.Parameter(features_dc_chunk.contiguous())
        self._features_rest=torch.nn.Parameter(features_rest_chunk.contiguous())
        self._opacity=torch.nn.Parameter(opacity_chunk.contiguous())
        self._scaling=torch.nn.Parameter(scaling_chunk.contiguous())
        self._rotation=torch.nn.Parameter(rotation_chunk.contiguous())

        self.chunk_AABB_origin=None
        self.chunk_AABB_extend=None
        self.b_split_into_chunk=True
        return
    
    @torch.no_grad()
    def __concate_param_chunks(self):
        assert(self.b_split_into_chunk==True)
        new_xyz=self._xyz.reshape(*self._xyz.shape[:-2],-1)
        new_features_dc=self._features_dc.reshape(*self._features_dc.shape[:-2],-1)
        new_features_rest=self._features_rest.reshape(*self._features_rest.shape[:-2],new_features_dc.shape[-1])
        new_opacity=self._opacity.reshape(*self._opacity.shape[:-2],-1)
        new_scaling=self._scaling.reshape(*self._scaling.shape[:-2],-1)
        new_rotation=self._rotation.reshape(*self._rotation.shape[:-2],-1)

        self._xyz=torch.nn.Parameter(new_xyz)
        self._features_dc=torch.nn.Parameter(new_features_dc)
        self._features_rest=torch.nn.Parameter(new_features_rest)
        self._opacity=torch.nn.Parameter(new_opacity)
        self._scaling=torch.nn.Parameter(new_scaling)
        self._rotation=torch.nn.Parameter(new_rotation)

        self.b_split_into_chunk=False
        return
    
    def rebuild_chunk_morton(self,chunk_size):
        assert(chunk_size>0)
        self.chunk_size=chunk_size
        self.__concate_param_chunks()
        self.__split_into_chunk()
        self.rebuild_AABB()
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def build_AABB_for_additional_chunks(self,chunks_num,valid_mask):
        if valid_mask is not None:
            self.chunk_AABB_origin=self.chunk_AABB_origin[valid_mask]
            self.chunk_AABB_extend=self.chunk_AABB_extend[valid_mask]

        if chunks_num>=1:
            scale=self._scaling[:,-chunks_num:,:].reshape(3,-1).exp()
            rotator=torch.nn.functional.normalize(self._rotation[:,-chunks_num:,:],dim=0).reshape(4,-1)
            transform_matrix=wrapper.CreateTransformMatrix.call_fused(scale,rotator)
            coefficient=2*math.log(255)
            extend_axis=transform_matrix*math.sqrt(coefficient)# == (coefficient*eigen_val).sqrt()*eigen_vec
            point_extend=extend_axis.abs().sum(dim=0).reshape(3,-1,self.chunk_size).permute(1,2,0)

            position=(self._xyz[:3,-chunks_num:,:]).permute(1,2,0)
            max_xyz=(position+point_extend).max(dim=-2).values
            min_xyz=(position-point_extend).min(dim=-2).values
            origin=(max_xyz+min_xyz)/2
            extend=(max_xyz-min_xyz)/2
            
            if self.chunk_AABB_origin is None:
                self.chunk_AABB_origin=origin
            else:
                self.chunk_AABB_origin=torch.cat((self.chunk_AABB_origin,origin))

            if self.chunk_AABB_extend is None:
                self.chunk_AABB_extend=extend
            else:
                self.chunk_AABB_extend=torch.cat((self.chunk_AABB_extend,extend))
        return 

    @torch.no_grad()
    def rebuild_AABB(self):
        assert(self.b_split_into_chunk)
        chunks_num=self._xyz.shape[-2]
        self.chunk_AABB_origin=None
        self.chunk_AABB_extend=None
        self.build_AABB_for_additional_chunks(chunks_num,None)
        return 

    def get_params(self):
        return (self._xyz,self._features_dc,self._features_rest,self._rotation,self._scaling,self._opacity)
    
    def load_params(self,params_tuple):
        (self._xyz,self._features_dc,self._features_rest,self._rotation,self._scaling,self._opacity)=params_tuple
        return
    
    def reset_statistics_helper(self):
        self.statistics_helper.reset(self._xyz.shape[0])
        return
    
    def oneupSHdegree(self):
        if self.actived_sh_degree < self.max_sh_degree:
            self.actived_sh_degree += 1
        return

    def create_cov2d_optimized(self,scaling_vec,rotator_vec,point_positions,view_matrix,camera_focal):
        '''
        Create 2d cov directly so that some intermediate variables are invisible in Python.

        If you need to modifiy the intermediate variables, use the following functions: transform_to_cov3d -> proj_cov3d_to_cov2d
        '''
        transform_matrix=wrapper.CreateTransformMatrix.call_fused(scaling_vec,rotator_vec)
        J=wrapper.CreateRaySpaceTransformMatrix.call_fused(point_positions,view_matrix,camera_focal,False)
        cov2d=wrapper.CreateCov2dDirectly.call_fused(J,view_matrix,transform_matrix)
        return cov2d
    
    def transform_to_cov3d(self,scaling_vec,rotator_vec)->torch.Tensor:
        transform_matrix=wrapper.CreateTransformMatrix.call_fused(scaling_vec,rotator_vec)#[3,3,P]
        cov3d=wrapper.CreateCovarianceMatrixFunc.apply(transform_matrix.permute((2,0,1))).permute((1,2,0))
        return cov3d,transform_matrix
    
    #@torch.compile
    def proj_cov3d_to_cov2d(self,cov3d,point_positions,view_matrix,camera_focal)->torch.Tensor:
        '''
        J^t @ M^t @ Cov3d @ M @J
        '''
        trans_J=wrapper.CreateRaySpaceTransformMatrix.call_fused(point_positions,view_matrix,camera_focal,True)[:,:2].transpose(-1,-2).transpose(-2,-3)

        trans_M=view_matrix[:,0:3,0:3].unsqueeze(0).transpose(-1,-2)
        trans_T=(trans_J@trans_M).contiguous()

        cov2d=wrapper.ProjCov3dTo2dFunc.apply(cov3d,trans_T)

        cov2d[:,:,0,0]+=0.3
        cov2d[:,:,1,1]+=0.3
        return cov2d.transpose(1,2).transpose(2,3)

    @torch.no_grad()
    def update_tiles_coord(self,image_size,tile_size,batch_size=1):
        self.cached_image_size=image_size
        self.cached_image_size_tensor=torch.Tensor(image_size).cuda().int()
        self.cached_tile_size=tile_size
        self.cached_tiles_size=(math.ceil(image_size[0]/tile_size),math.ceil(image_size[1]/tile_size))
        self.cached_tiles_map=torch.arange(0,self.cached_tiles_size[0]*self.cached_tiles_size[1]).int().reshape(self.cached_tiles_size[1],self.cached_tiles_size[0]).cuda()+1#tile_id 0 is invalid
        self.cached_tiles_map=self.cached_tiles_map.reshape(1,-1).repeat((batch_size,1))
        return
    
    @torch.no_grad()
    def culling_and_sort(self,ndc_pos,limit_LURD=None):
        '''
        input: ViewMatrix,ProjMatrix
        output: sorted_visible_points,num_of_points
        '''
        if limit_LURD is None:
            culling_result=torch.any(ndc_pos[...,0:2]<-1.3,dim=2)|torch.any(ndc_pos[...,0:2]>1.3,dim=2)|(ndc_pos[...,2]<0)|(ndc_pos[...,2]>1.0)#near plane 0.01
        else:
            culling_result=torch.any(ndc_pos[...,0:2]<limit_LURD.unsqueeze(1)[...,0:2]*1.3,dim=2)|torch.any(ndc_pos[...,0:2]>limit_LURD.unsqueeze(1)[...,2:4]*1.3,dim=2)|(ndc_pos[...,2]<0)|(ndc_pos[...,2]>1.0)

        culling_result=torch.all(culling_result,dim=0)
        visible_points_num=(~culling_result).sum()
        visible_point=~culling_result


        return visible_point,visible_points_num
    
    def cluster_compact(self,chunk_visibility):

        positions,scales,rotators,sh_base,sh_rest,opacities,_=wrapper.compact_visible_params(chunk_visibility,self._xyz,self._scaling,self._rotation,self._features_dc,self._features_rest,self._opacity)

        scales=scales.exp()
        rotators=torch.nn.functional.normalize(rotators,dim=0)
        opacities=opacities.sigmoid()

        return positions,scales,rotators,sh_base,sh_rest,opacities

    @platform_torch_compile
    def __craete_2d_AABB(self,ndc:torch.Tensor,eigen_val:torch.Tensor,eigen_vec:torch.Tensor,opacity:torch.Tensor,tile_size:int,tilesX:int,tilesY:int):
        # Major and minor axes -> AABB extensions
        opacity_clamped=opacity.unsqueeze(0).clamp_min(1/255)
        coefficient=2*((255*opacity_clamped).log())#-2*(1/(255*opacity.squeeze(-1))).log()
        axis_length=(coefficient*eigen_val.abs()).sqrt().ceil()
        extension=(axis_length.unsqueeze(-2)*eigen_vec).abs().sum(dim=-3)

        screen_coord=((ndc[:,:2]+1.0)*0.5*self.cached_image_size_tensor.unsqueeze(-1)-0.5)
        b_visible=~((ndc[:,0]<-1.3)|(ndc[:,0]>1.3)|(ndc[:,1]<-1.3)|(ndc[:,1]>1.3)|(ndc[:,2]>1)|(ndc[:,2]<0))
        left_up=((screen_coord-extension)/tile_size).int()*b_visible
        right_down=((screen_coord+extension)/tile_size).ceil().int()*b_visible
        left_up[:,0].clamp_(0,tilesX)
        left_up[:,1].clamp_(0,tilesY)
        right_down[:,0].clamp_(0,tilesX)
        right_down[:,1].clamp_(0,tilesY)

        #splatting area of each points
        rect_length=right_down-left_up
        tiles_touched=rect_length[:,0]*rect_length[:,1]
        b_visible=(tiles_touched!=0)

        return left_up,right_down,tiles_touched,b_visible

    @torch.no_grad()
    def binning(self,ndc:torch.Tensor,eigen_val:torch.Tensor,eigen_vec:torch.Tensor,opacity:torch.Tensor):

        tilesX=self.cached_tiles_size[0]
        tilesY=self.cached_tiles_size[1]
        tiles_num=tilesX*tilesY
        tile_size=self.cached_tile_size
        image_size=self.cached_image_size

        left_up,right_down,tiles_touched,b_visible=self.__craete_2d_AABB(ndc,eigen_val,eigen_vec,opacity,tile_size,tilesX,tilesY)

        #sort by depth
        values,point_ids=ndc[:,2].sort(dim=-1)
        for i in range(ndc.shape[0]):
            tiles_touched[i]=tiles_touched[i,point_ids[i]]

        #calc the item num of table and the start index in table of each point
        prefix_sum=tiles_touched.cumsum(1,dtype=torch.int32)#start index of points
        total_tiles_num_batch=prefix_sum[:,-1]
        allocate_size=total_tiles_num_batch.max().cpu()

        # allocate table and fill it (Table: tile_id-uint16,point_id-uint16)
        my_table=torch.ops.GaussianRaster.duplicateWithKeys(left_up,right_down,prefix_sum,point_ids,int(allocate_size),int(tilesX))
        tileId_table:torch.Tensor=my_table[0]
        pointId_table:torch.Tensor=my_table[1]

        # sort tile_id with torch.sort
        sorted_tileId,indices=torch.sort(tileId_table,dim=1,stable=True)
        sorted_pointId=pointId_table.gather(dim=1,index=indices)

        # range
        tile_start_index=torch.ops.GaussianRaster.tileRange(sorted_tileId,int(allocate_size),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)
            
        return tile_start_index,sorted_pointId,sorted_tileId,b_visible
    
    def raster(self,ndc_pos:torch.Tensor,inv_cov2d:torch.Tensor,color:torch.Tensor,opacities:torch.Tensor,tile_start_index:torch.Tensor,sorted_pointId:torch.Tensor,sorted_tileId:torch.Tensor,tiles:torch.Tensor):
    
        if StatisticsHelperInst.bStart and ndc_pos.requires_grad:
            def gradient_wrapper(tensor:torch.Tensor) -> torch.Tensor:
                return tensor[:,:2].norm(dim=1)
            StatisticsHelperInst.register_tensor_grad_callback('mean2d_grad',ndc_pos,StatisticsHelper.update_mean_std_compact,gradient_wrapper)
        img,transmitance=wrapper.GaussiansRasterFunc.apply(sorted_pointId,tile_start_index,ndc_pos,inv_cov2d,color,opacities,tiles,
                                               self.cached_tile_size,self.cached_tiles_size[0],self.cached_tiles_size[1],self.cached_image_size[1],self.cached_image_size[0])

        return img,transmitance
    
    def render(self,view_matrix:torch.Tensor,view_project_matrix:torch.Tensor,camera_focal:torch.Tensor,camera_center_batch:torch.Tensor,
               tiles:torch.Tensor=None,prebackward_func:typing.Callable=None):
        
        assert(self.b_split_into_chunk)

        #cluster culling
        with torch.no_grad():
            frustumplane=cg_torch.viewproj_to_frustumplane(view_project_matrix)
            chunk_visibility=cg_torch.frustum_culling_aabb(frustumplane,self.chunk_AABB_origin,self.chunk_AABB_extend)
            chunk_visibility=chunk_visibility.any(dim=0)
            visible_chunkid=chunk_visibility.nonzero()[:,0]
            if StatisticsHelperInst.bStart:
                StatisticsHelperInst.set_compact_mask(chunk_visibility)

        #compact
        positions,scales,rotators,sh_base,sh_rest,opacities=self.cluster_compact(chunk_visibility)
        if prebackward_func is not None:
            prebackward_func(positions,scales,rotators,sh_base,sh_rest,opacities)

        #gs projection
        cov2d=self.create_cov2d_optimized(scales,rotators,positions,view_matrix,camera_focal)
        eigen_val,eigen_vec,inv_cov2d=wrapper.EighAndInverse2x2Matrix.call_fused(cov2d)
        ndc_pos_batch=wrapper.World2NdcFunc.apply(positions,view_project_matrix)

        #color
        dirs=positions[:3]-camera_center_batch.unsqueeze(-1)
        dirs=torch.nn.functional.normalize(dirs,dim=-2)
        colors=wrapper.SphericalHarmonicToRGB.call_fused(self.actived_sh_degree,sh_base,sh_rest,dirs)
        
        #visibility table
        tile_start_index,sorted_pointId,sorted_tileId,b_visible=self.binning(ndc_pos_batch,eigen_val,eigen_vec,opacities)
        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.update_visible_count(b_visible)

        #rasterization
        if tiles is None:
            tiles=self.cached_tiles_map
        tile_img,tile_transmitance=self.raster(ndc_pos_batch,inv_cov2d,colors,opacities,tile_start_index,sorted_pointId,sorted_tileId,tiles)

        return tile_img,tile_transmitance.unsqueeze(1),visible_chunkid
