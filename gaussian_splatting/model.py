from gaussian_splatting.scene import GaussianScene
from util import cg_torch
from gaussian_splatting import wrapper


import torch
import typing
import numpy as np
import math
from util.statistic_helper import StatisticsHelperInst,StatisticsHelper
from util.BVH.Object import GSpointBatch
from util.BVH.PytorchBVH import BVH
from util.BVH.visualization import VisualizationHelper


class GaussianSplattingModel:
    @torch.no_grad()
    def __init__(self,scene:GaussianScene,chunk_size=1024):
        assert(chunk_size>0)
        self.actived_sh_degree=0
        self.max_sh_degree=scene.sh_degree
        self.chunk_size=chunk_size
        self.chunk_AABB_origin=None
        self.chunk_AABB_extend=None
        self.b_split_into_chunk=False
        
        self.load_from_scene(scene)
        return
    
    @torch.no_grad()
    def load_from_scene(self,scene:GaussianScene):
        self._xyz = torch.nn.Parameter(torch.Tensor(np.pad(scene.position,((0,0),(0,1)),'constant',constant_values=1)).transpose(0,1).cuda())
        self._features_dc = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_dc).transpose(0, -1).cuda())
        self._features_rest = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_rest).transpose(0, -1).cuda())
        self._rotation = torch.nn.Parameter(torch.Tensor(scene.rotator).transpose(0,1).cuda())#torch.nn.functional.normalize
        self._scaling = torch.nn.Parameter(torch.Tensor(scene.scale).transpose(0,1).cuda())#.exp 
        self._opacity = torch.nn.Parameter(torch.Tensor(scene.opacity).transpose(0,1).cuda())#.sigmoid

        self.__split_into_chunk()
        return
    

    @torch.no_grad()
    def save_to_scene(self,scene:GaussianScene):
        if self.b_split_into_chunk:
            self.__concate_param_chunks()
        scene.position=self._xyz[...,:3].cpu().numpy()
        scene.sh_coefficient_dc=self._features_dc.cpu().numpy()
        scene.sh_coefficient_rest=self._features_rest.cpu().numpy()
        scene.rotator=self._rotation.cpu().numpy()
        scene.scale=self._scaling.cpu().numpy()
        scene.opacity=self._opacity.cpu().numpy()
        scene.sh_degree=0
        return
    
    @torch.no_grad()
    def __split_into_chunk(self):
        assert(self.b_split_into_chunk==False)
        #build BVH
        temporary_point_id=torch.arange(self._xyz.shape[-1],device='cuda')
        scale=self._scaling.exp()
        roator=torch.nn.functional.normalize(self._rotation,dim=0)
        temporary_cov3d,_=self.transform_to_cov3d(scale,roator)
        points_batch=GSpointBatch(temporary_point_id,self._xyz[:3,:].permute(1,0),{'cov':temporary_cov3d.permute(2,0,1)})
        bvh=BVH([points_batch,])
        bvh.build(self.chunk_size)

        #split into chunk
        chunk_num=len(bvh.leaf_nodes)
        xyz_chunk=torch.zeros((*self._xyz.shape[:-1],chunk_num,self.chunk_size),device='cuda')
        features_dc_chunk=torch.zeros((*self._features_dc.shape[:-1],chunk_num,self.chunk_size),device='cuda')
        features_rest_chunk=torch.zeros((*self._features_rest.shape[:-1],chunk_num,self.chunk_size),device='cuda')
        opacity_chunk=torch.zeros((*self._opacity.shape[:-1],chunk_num,self.chunk_size),device='cuda')
        scaling_chunk=torch.zeros((*self._scaling.shape[:-1],chunk_num,self.chunk_size),device='cuda')
        rotation_chunk=torch.zeros((*self._rotation.shape[:-1],chunk_num,self.chunk_size),device='cuda')

        origin_list=[]
        extend_list=[]
        for chunk_index,node in enumerate(bvh.leaf_nodes):
            points_num=node.objs.shape[0]
            repeat_n=int(self.chunk_size/points_num)
            xyz_chunk[:,chunk_index,:repeat_n*points_num]=self._xyz[...,node.objs].repeat(1,repeat_n)
            features_dc_chunk[:,:,chunk_index,:repeat_n*points_num]=self._features_dc[...,node.objs].repeat(1,1,repeat_n)
            features_rest_chunk[:,:,chunk_index,:repeat_n*points_num]=self._features_rest[...,node.objs].repeat(1,1,repeat_n)
            opacity_chunk[:,chunk_index,:repeat_n*points_num]=self._opacity[...,node.objs].repeat(1,repeat_n)
            scaling_chunk[:,chunk_index,:repeat_n*points_num]=self._scaling[...,node.objs].repeat(1,repeat_n)
            rotation_chunk[:,chunk_index,:repeat_n*points_num]=self._rotation[...,node.objs].repeat(1,repeat_n)

            remain_num=self.chunk_size-repeat_n*points_num
            if remain_num>0:
                remain_points=node.objs[:remain_num]
                xyz_chunk[:,chunk_index,repeat_n*points_num:]=self._xyz[...,remain_points]
                features_dc_chunk[:,:,chunk_index,repeat_n*points_num:]=self._features_dc[...,remain_points]
                features_rest_chunk[:,:,chunk_index,repeat_n*points_num:]=self._features_rest[...,remain_points]
                opacity_chunk[:,chunk_index,repeat_n*points_num:]=-1e5
                scaling_chunk[:,chunk_index,repeat_n*points_num:]=self._scaling[...,remain_points]
                rotation_chunk[:,chunk_index,repeat_n*points_num:]=self._rotation[...,remain_points]

            origin_list.append(node.origin.unsqueeze(0))
            extend_list.append(node.extend.unsqueeze(0))

        self.chunk_AABB_origin=torch.cat(origin_list,dim=-2).contiguous()
        self.chunk_AABB_extend=torch.cat(extend_list,dim=-2).contiguous()
    
        self._xyz=torch.nn.Parameter(xyz_chunk.contiguous())
        self._features_dc=torch.nn.Parameter(features_dc_chunk.contiguous())
        self._features_rest=torch.nn.Parameter(features_rest_chunk.contiguous())
        self._opacity=torch.nn.Parameter(opacity_chunk.contiguous())
        self._scaling=torch.nn.Parameter(scaling_chunk.contiguous())
        self._rotation=torch.nn.Parameter(rotation_chunk.contiguous())
        self.b_split_into_chunk=True
        return
    
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
    
    def rebuild_BVH(self,chunk_size):
        assert(chunk_size>0)
        self.chunk_size=chunk_size
        self.__concate_param_chunks()
        self.__split_into_chunk()
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def build_AABB_for_additional_chunks(self,chunks_num,valid_mask):
        if valid_mask is not None:
            self.chunk_AABB_origin=self.chunk_AABB_origin[valid_mask]
            self.chunk_AABB_extend=self.chunk_AABB_extend[valid_mask]

        if chunks_num>=1:
            scale=self._scaling[:,-1-chunks_num:-1,:].reshape(3,-1).exp()
            roator=torch.nn.functional.normalize(self._rotation[:,-1-chunks_num:-1,:],dim=0).reshape(4,-1)
            cov3d,_=self.transform_to_cov3d(scale,roator)
            cov3d=cov3d.reshape(3,3,-1,self.chunk_size).permute(2,3,0,1)#[chunks_num,chunk_size,3,3]

            eigen_val,eigen_vec=torch.linalg.eigh(cov3d)
            eigen_val=eigen_val.abs()
            coefficient=2*math.log(255)
            point_extend=((coefficient*eigen_val.unsqueeze(-1)).sqrt()*eigen_vec).abs().sum(dim=-2)
            position=(self._xyz[:3,-1-chunks_num:-1,:]).permute(1,2,0)
            max_xyz=(position+point_extend).max(dim=-2).values
            min_xyz=(position-point_extend).min(dim=-2).values
            origin=(max_xyz+min_xyz)/2
            extend=(max_xyz-min_xyz)/2

            self.chunk_AABB_origin=torch.cat((self.chunk_AABB_origin,origin))
            self.chunk_AABB_extend=torch.cat((self.chunk_AABB_extend,extend))
        return 

    @torch.no_grad()
    def rebuild_AABB(self):
        scale=self._scaling.exp()
        roator=torch.nn.functional.normalize(self._rotation,dim=0)
        cov3d,_=self.transform_to_cov3d(scale.reshape(3,-1),roator.reshape(4,-1))
        cov3d=cov3d.reshape(3,3,-1,self.chunk_size).permute(2,3,0,1)

        eigen_val_list=[]
        eigen_vec_list=[]
        for start_inedx in range(0,cov3d.shape[0],1024):
            eigen_val,eigen_vec=torch.linalg.eigh(cov3d[start_inedx:start_inedx+1024])
            eigen_val_list.append(eigen_val)
            eigen_vec_list.append(eigen_vec)
        eigen_val=torch.cat(eigen_val_list)
        eigen_vec=torch.cat(eigen_vec_list)

        eigen_val=eigen_val.abs()
        coefficient=2*math.log(255)
        point_extend=((coefficient*eigen_val.unsqueeze(-1)).sqrt()*eigen_vec).abs().sum(dim=-2)
        position=self._xyz.permute(1,2,0)[...,:3]
        max_xyz=(position+point_extend).max(dim=-2).values
        min_xyz=(position-point_extend).min(dim=-2).values
        origin=(max_xyz+min_xyz)/2
        extend=(max_xyz-min_xyz)/2

        self.chunk_AABB_origin=origin
        self.chunk_AABB_extend=extend
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
        transform_matrix=wrapper.create_transform_matrix(scaling_vec,rotator_vec)
        J=wrapper.create_rayspace_transform(point_positions,view_matrix,camera_focal,False)
        cov2d=wrapper.create_2dcov_directly(J,view_matrix,transform_matrix)
        return cov2d
    
    def transform_to_cov3d(self,scaling_vec,rotator_vec)->torch.Tensor:
        transform_matrix=wrapper.create_transform_matrix(scaling_vec,rotator_vec)#[3,3,P]
        cov3d=wrapper.create_cov3d(transform_matrix.permute((2,0,1))).permute((1,2,0))
        return cov3d,transform_matrix
    
    #@torch.compile
    def proj_cov3d_to_cov2d(self,cov3d,point_positions,view_matrix,camera_focal)->torch.Tensor:
        '''
        J^t @ M^t @ Cov3d @ M @J
        '''
        trans_J=wrapper.create_rayspace_transform(point_positions,view_matrix,camera_focal,True)[:,:2].transpose(-1,-2).transpose(-2,-3)

        trans_M=view_matrix[:,0:3,0:3].unsqueeze(0).transpose(-1,-2)
        trans_T=(trans_J@trans_M).contiguous()

        cov2d=wrapper.project_3dcov_to_2d(cov3d,trans_T)#backward improvement

        cov2d[:,:,0,0]+=0.3
        cov2d[:,:,1,1]+=0.3
        return cov2d.transpose(1,2).transpose(2,3)

    
    def update_tiles_coord(self,image_size,tile_size):
        self.cached_image_size=image_size
        self.cached_image_size_tensor=torch.Tensor(image_size).cuda().int()
        self.cached_tile_size=tile_size
        self.cached_tiles_size=(math.ceil(image_size[0]/tile_size),math.ceil(image_size[1]/tile_size))
        self.cached_tiles_map=torch.arange(0,self.cached_tiles_size[0]*self.cached_tiles_size[1]).int().reshape(self.cached_tiles_size[1],self.cached_tiles_size[0]).cuda()+1#tile_id 0 is invalid
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
    
    def sample_by_visibility(self,chunk_visibility):

        positions,scales,rotators,sh_base,sh_rest,opacities,_=wrapper.compact_visible_params(chunk_visibility,self._xyz,self._scaling,self._rotation,self._features_dc,self._features_rest,self._opacity)

        scales=scales.exp()
        rotators=torch.nn.functional.normalize(rotators,dim=0)
        opacities=opacities.sigmoid()

        return positions,scales,rotators,sh_base,sh_rest,opacities

    
    @torch.no_grad()
    def binning(self,ndc:torch.Tensor,eigen_val:torch.Tensor,eigen_vec:torch.Tensor,opacity:torch.Tensor):
        #!!!befor duplicateWithKeys 5.7ms!!!!
        tilesX=self.cached_tiles_size[0]
        tilesY=self.cached_tiles_size[1]
        tiles_num=tilesX*tilesY
        tile_size=self.cached_tile_size
        image_size=self.cached_image_size

        # Major and minor axes -> AABB extensions
        opacity_clamped=opacity.unsqueeze(0).clamp_min(1/255)
        coefficient=2*((255*opacity_clamped).log())#-2*(1/(255*opacity.squeeze(-1))).log()
        axis_length=(coefficient*eigen_val.abs()).sqrt().ceil()
        extension=(axis_length.unsqueeze(-2)*eigen_vec).abs().sum(dim=-3)

        screen_coord=((ndc[:,:2]+1.0)*0.5*self.cached_image_size_tensor.unsqueeze(-1)-0.5)
        b_visible=~((ndc[:,0]<-1.3)|(ndc[:,0]>1.3)|(ndc[:,1]>1.3)|(ndc[:,1]>1.3)|(ndc[:,2]>1)|(ndc[:,2]<0))
        left_up=((screen_coord-extension)/tile_size).int()*b_visible
        right_down=((screen_coord+extension)/tile_size).ceil().int()*b_visible
        left_up[:,0].clamp_(0,tilesX)
        left_up[:,1].clamp_(0,tilesY)
        right_down[:,0].clamp_(0,tilesX)
        right_down[:,1].clamp_(0,tilesY)

        #splatting area of each points
        rect_length=right_down-left_up
        tiles_touched=rect_length[:,0]*rect_length[:,1]
        radius_pixel=(axis_length.max(-2).values*(tiles_touched!=0))

        #sort by depth
        values,point_ids=ndc[:,2].sort(dim=-1)#0.8ms
        for i in range(ndc.shape[0]):
            tiles_touched[i]=tiles_touched[i,point_ids[i]]
            left_up[i]=left_up[i,:,point_ids[i]]
            right_down[i]=right_down[i,:,point_ids[i]]

        #calc the item num of table and the start index in table of each point
        prefix_sum=tiles_touched.cumsum(1)#start index of points
        total_tiles_num_batch=prefix_sum[:,-1]
        allocate_size=total_tiles_num_batch.max().cpu()

        # allocate table and fill it (Table: tile_id-uint16,point_id-uint16)
        my_table=torch.ops.RasterBinning.duplicateWithKeys(left_up,right_down,prefix_sum,int(allocate_size),int(tilesX))#2ms
        tileId_table:torch.Tensor=my_table[0]
        pointId_table:torch.Tensor=my_table[1]
        pointId_table=point_ids.gather(dim=1,index=pointId_table.long()).int()#!!!!2ms!!!

        # sort tile_id with torch.sort
        sorted_tileId,indices=torch.sort(tileId_table,dim=1,stable=True)
        sorted_pointId=pointId_table.gather(dim=1,index=indices)#!!!1.2ms!!!

        #debug:check total_tiles_num_batch
        #cmp_result=(tileId_table!=0).sum(dim=1)==total_tiles_num_batch[:,0]
        #print(cmp_result)
        #cmp_result=(sorted_tileId!=0).sum(dim=1)==total_tiles_num_batch[:,0]
        #print(cmp_result)

        # range 0.3ms
        tile_start_index=torch.ops.RasterBinning.tileRange(sorted_tileId,int(allocate_size),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)
            
        return tile_start_index,sorted_pointId,sorted_tileId,radius_pixel
    
    def raster(self,ndc_pos:torch.Tensor,inv_cov2d:torch.Tensor,color:torch.Tensor,opacities:torch.Tensor,tile_start_index:torch.Tensor,sorted_pointId:torch.Tensor,sorted_tileId:torch.Tensor,tiles:torch.Tensor):
    
        mean2d=(ndc_pos[:,0:2]+1.0)*0.5*self.cached_image_size_tensor.unsqueeze(-1)-0.5
        if StatisticsHelperInst.bStart and ndc_pos.requires_grad:
            def gradient_wrapper(tensor:torch.Tensor) -> torch.Tensor:
                return tensor[:,:2].norm(dim=1)
            StatisticsHelperInst.register_tensor_grad_callback('mean2d_grad',ndc_pos,StatisticsHelper.update_mean_std_compact,gradient_wrapper)
        img,transmitance=wrapper.rasterize_2d_gaussian(sorted_pointId,tile_start_index,mean2d,inv_cov2d,color,opacities,tiles,
                                               self.cached_tile_size,self.cached_tiles_size[0],self.cached_tiles_size[1],self.cached_image_size[1],self.cached_image_size[0])

        return img,transmitance
    
    def render(self,view_matrix:torch.Tensor,view_project_matrix:torch.Tensor,camera_focal:torch.Tensor,camera_center_batch:torch.Tensor,
               tiles:torch.Tensor=None,prebackward_func:typing.Callable=None):
        
        assert(self.b_split_into_chunk)

        #compute visibility
        with torch.no_grad():
            frustumplane=cg_torch.viewproj_to_frustumplane(view_project_matrix)
            chunk_visibility=cg_torch.frustum_culling_aabb(frustumplane,self.chunk_AABB_origin,self.chunk_AABB_extend)
            chunk_visibility=chunk_visibility.any(dim=0)
            if StatisticsHelperInst.bStart:
                StatisticsHelperInst.set_compact_mask(chunk_visibility)

        positions,scales,rotators,sh_base,sh_rest,opacities=self.sample_by_visibility(chunk_visibility)#forward 1.5ms(compact 0.7ms)
        if prebackward_func is not None:
            prebackward_func(positions,scales,rotators,sh_base,sh_rest,opacities)

        ### (scale,rot)->3d covariance matrix->2d covariance matrix ###
        #cov3d,transform_matrix=self.transform_to_cov3d(scales,rotators)
        #cov2d=self.proj_cov3d_to_cov2d(cov3d,positions,view_matrix,camera_focal)
        cov2d=self.create_cov2d_optimized(scales,rotators,positions,view_matrix,camera_focal)#1.46ms
        eigen_val,eigen_vec,inv_cov2d=wrapper.eigh_and_inverse_cov2d(cov2d)#0.5ms
        
        ### mean of 2d-gaussian ###
        ndc_pos_batch=wrapper.wrold2ndc(positions,view_project_matrix)#1ms

        ### color ###
        dirs=positions[:3]-camera_center_batch.unsqueeze(-1)
        dirs=torch.nn.functional.normalize(dirs,dim=-2)#dir 0.5ms
        colors=wrapper.sh2rgb(self.actived_sh_degree,sh_base,sh_rest,dirs)#degree0: 0.1ms
        
        #### binning ###
        tile_start_index,sorted_pointId,sorted_tileId,radii=self.binning(ndc_pos_batch,eigen_val,eigen_vec,opacities)#15ms sort1 0.8ms sort2 2ms
        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.update_max_min_compact('radii',radii)
            StatisticsHelperInst.update_visible_count(radii>0)

        #### raster ###
        if tiles is None:
            batch_size=view_matrix.shape[0]
            tiles=self.cached_tiles_map.reshape(1,-1).repeat((batch_size,1))
        tile_img,tile_transmitance=self.raster(ndc_pos_batch,inv_cov2d,colors,opacities,tile_start_index,sorted_pointId,sorted_tileId,tiles)#5.7ms

        
        return tile_img,tile_transmitance.unsqueeze(1)
