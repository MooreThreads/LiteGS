from gaussian_splatting.scene import GaussianScene
from training.arguments import OptimizationParams
from util import spherical_harmonics,cg_torch
from gaussian_splatting import wrapper


import torch
import typing
import numpy as np
import math
from torch.cuda.amp import autocast
from util.statistic_helper import StatisticsHelperInst,StatisticsHelper


class GaussianSplattingModel:
    @torch.no_grad()
    def __init__(self,scene:GaussianScene,spatial_lr_scale):
        
        self._xyz = torch.nn.Parameter(torch.Tensor(np.pad(scene.position,((0,0),(0,1)),'constant',constant_values=1)).cuda())
        self._features_dc = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_dc).transpose(1, 2).contiguous().cuda())
        self._features_rest = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_rest).transpose(1, 2).contiguous().cuda())
        self._rotation = torch.nn.Parameter(torch.Tensor(scene.rotator).cuda())#torch.nn.functional.normalize
        self.spatial_lr_scale=spatial_lr_scale
        self.actived_sh_degree=0
        self.max_sh_degree=scene.sh_degree

        #exp scale
        self._scaling = torch.nn.Parameter(torch.Tensor(scene.scale).cuda())#.exp 
        self._opacity = torch.nn.Parameter(torch.Tensor(scene.opacity).cuda())#.sigmoid

        return
    
    def get_params(self):
        return (self._xyz,self._features_dc,self._features_rest,self._rotation,self.spatial_lr_scale,self._scaling,self._opacity)
    
    def load_params(self,params_tuple):
        (self._xyz,self._features_dc,self._features_rest,self._rotation,self.spatial_lr_scale,self._scaling,self._opacity)=params_tuple
        return
    
    def reset_statistics_helper(self):
        self.statistics_helper.reset(self._xyz.shape[0])
        return

    @torch.no_grad()
    def save_to_scene(self,scene:GaussianScene):
        scene.position=self._xyz[...,:3].cpu().numpy()
        scene.sh_coefficient_dc=self._features_dc.cpu().numpy()
        scene.sh_coefficient_rest=self._features_rest.cpu().numpy()
        scene.rotator=self._rotation.cpu().numpy()
        scene.scale=self._scaling.cpu().numpy()
        scene.opacity=self._opacity.cpu().numpy()
        scene.sh_degree=0
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
        J=wrapper.create_rayspace_transform(None,point_positions,view_matrix,camera_focal,False)
        cov2d=wrapper.create_2dcov_directly(J,view_matrix,transform_matrix)
        cov2d[:,:,0,0]+=0.3
        cov2d[:,:,1,1]+=0.3
        return cov2d
    
    def transform_to_cov3d(self,scaling_vec,rotator_vec)->torch.Tensor:
        transform_matrix=wrapper.create_transform_matrix(scaling_vec,rotator_vec)
        cov3d=wrapper.create_cov3d(transform_matrix)
        return cov3d,transform_matrix
    
    #@torch.compile
    def proj_cov3d_to_cov2d(self,cov3d,point_positions,view_matrix,camera_focal)->torch.Tensor:
        '''
        J^t @ M^t @ Cov3d @ M @J
        '''
        trans_J=wrapper.create_rayspace_transform(cov3d,point_positions,view_matrix,camera_focal,True)[:,:,:2,:]

        trans_M=view_matrix.unsqueeze(1)[:,:,0:3,0:3].transpose(-1,-2)
        trans_T=(trans_J@trans_M).contiguous()

        cov2d=wrapper.project_3dcov_to_2d(cov3d,trans_T)#backward improvement

        cov2d[:,:,0,0]+=0.3
        cov2d[:,:,1,1]+=0.3
        return cov2d

    
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
    
    def sample_by_visibility(self,visible_points_for_views):
        scales=self._scaling[visible_points_for_views].contiguous().exp()
        rotators=torch.nn.functional.normalize(self._rotation[visible_points_for_views].contiguous(),dim=-1)

        visible_positions=self._xyz[visible_points_for_views].contiguous()
        visible_opacities=self._opacity[visible_points_for_views].contiguous().sigmoid()
        visible_sh=torch.concat((self._features_dc[visible_points_for_views],self._features_rest[visible_points_for_views]),dim=1).contiguous()
        return scales,rotators,visible_positions,visible_opacities,visible_sh

    
    @torch.no_grad()
    def binning(self,ndc:torch.Tensor,cov2d:torch.Tensor,opacity:torch.Tensor,valid_points_num:torch.Tensor,bTraining=False):
        
        tilesX=self.cached_tiles_size[0]
        tilesY=self.cached_tiles_size[1]
        tiles_num=tilesX*tilesY
        tile_size=self.cached_tile_size
        image_size=self.cached_image_size

        det=cov2d[:,:,0,0]*cov2d[:,:,1,1]-cov2d[:,:,0,1]*cov2d[:,:,0,1]
        mid=0.5*(cov2d[:,:,0,0]+cov2d[:,:,1,1])
        temp=(mid*mid-det).clamp_min(0.1).sqrt()
        major_eigen_val=torch.max(mid+temp,mid-temp)
        # todo: rectangle formed by the major and minor axes
        # eigen_val,eigen_vec=torch.linalg.eigh(cov2d)
        # major_eigen_val=eigen_val.max(dim=-1)[0]
        opacity_clamped=opacity.squeeze(-1).clamp_min(0.005)
        coefficient=2*((255*opacity_clamped).log())#-2*(1/(255*opacity.squeeze(-1))).log()
        pixel_radius=(coefficient*major_eigen_val).sqrt().ceil()
        #pixel_radius=3.0*((major_eigen_val).sqrt()).ceil()

        values,point_ids=ndc[...,2].sort(dim=-1)
        ndc=ndc.clone()
        for i in range(ndc.shape[0]):
            ndc[i]=ndc[i,point_ids[i]]
            pixel_radius[i]=pixel_radius[i,point_ids[i]]

        coordX=(ndc[:,:,0]+1.0)*0.5*image_size[0]-0.5
        coordY=(ndc[:,:,1]+1.0)*0.5*image_size[1]-0.5
        
        L=((coordX-pixel_radius)/tile_size).floor().int().clamp(0,tilesX)
        U=((coordY-pixel_radius)/tile_size).floor().int().clamp(0,tilesY)
        R=((coordX+pixel_radius+tile_size-1)/tile_size).floor().int().clamp(0,tilesX)
        D=((coordY+pixel_radius+tile_size-1)/tile_size).floor().int().clamp(0,tilesY)

        #calculate params of allocation
        tiles_touched=(R-L)*(D-U)
        prefix_sum=tiles_touched.cumsum(1)
        total_tiles_num_batch=prefix_sum[:,-1]
        pixel_radius=(pixel_radius*(tiles_touched!=0))
        allocate_size=total_tiles_num_batch.max().cpu()
        

        
        # allocate table and fill tile_id in it(uint 16)
        my_table=torch.ops.RasterBinning.duplicateWithKeys(L,U,R,D,prefix_sum,int(allocate_size),int(tilesX))
        tileId_table:torch.Tensor=my_table[0]
        pointId_table:torch.Tensor=my_table[1]
        pointId_table=point_ids.gather(dim=1,index=pointId_table.long()).int()

        # sort tile_id with torch.sort
        sorted_tileId,indices=torch.sort(tileId_table,dim=1,stable=True)
        sorted_pointId=pointId_table.gather(dim=1,index=indices)

        #debug:check total_tiles_num_batch
        #cmp_result=(tileId_table!=0).sum(dim=1)==total_tiles_num_batch[:,0]
        #print(cmp_result)
        #cmp_result=(sorted_tileId!=0).sum(dim=1)==total_tiles_num_batch[:,0]
        #print(cmp_result)

        # range
        tile_start_index=torch.ops.RasterBinning.tileRange(sorted_tileId,int(allocate_size),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)
            
        return tile_start_index,sorted_pointId,sorted_tileId,pixel_radius.unsqueeze(-1)
    
    def raster(self,ndc_pos:torch.Tensor,cov2d:torch.Tensor,color:torch.Tensor,opacities:torch.Tensor,tile_start_index:torch.Tensor,sorted_pointId:torch.Tensor,sorted_tileId:torch.Tensor,tiles:torch.Tensor):
        
        # cov2d_inv=torch.linalg.inv(cov2d)#forward backward 1s
        #faster but unstable
        reci_det=1/(torch.det(cov2d)+1e-7)
        cov2d_inv=torch.zeros_like(cov2d)
        cov2d_inv[...,0,1]=-cov2d[...,0,1]*reci_det
        cov2d_inv[...,1,0]=-cov2d[...,1,0]*reci_det
        cov2d_inv[...,0,0]=cov2d[...,1,1]*reci_det
        cov2d_inv[...,1,1]=cov2d[...,0,0]*reci_det

        mean2d=(ndc_pos[:,:,0:2]+1.0)*0.5*self.cached_image_size_tensor-0.5
        img,transmitance=wrapper.rasterize_2d_gaussian(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                               self.cached_tile_size,self.cached_tiles_size[0],self.cached_tiles_size[1],self.cached_image_size[1],self.cached_image_size[0])

        return img,transmitance
    
    def render(self,visible_points_num:torch.Tensor,visible_points:torch.Tensor,
               view_matrix:torch.Tensor,view_project_matrix:torch.Tensor,camera_focal:torch.Tensor,camera_center_batch:torch.Tensor,
               tiles:torch.Tensor=None,
               prebackward_func:typing.Callable=None):
        
        ###process visibility
        if visible_points_num is None or visible_points is None:
            #compute visibility
            with torch.no_grad():
                ndc_pos=cg_torch.world_to_ndc(self._xyz,view_project_matrix)
                visible_points,visible_points_num=self.culling_and_sort(ndc_pos)

        visible_scales,visible_rotators,visible_positions,visible_opacities,visible_sh=self.sample_by_visibility(visible_points)
        if prebackward_func is not None:
            prebackward_func(visible_scales,visible_rotators,visible_positions,visible_opacities,visible_sh)

        ### (scale,rot)->3d covariance matrix->2d covariance matrix ###
        #cov3d,transform_matrix=self.transform_to_cov3d(visible_scales,visible_rotators)
        #visible_cov2d=self.proj_cov3d_to_cov2d(cov3d,visible_positions,view_matrix,camera_focal)
        visible_cov2d=self.create_cov2d_optimized(visible_scales,visible_rotators,visible_positions,view_matrix,camera_focal)
        
        ### color ###
        dirs=visible_positions[...,:3]-camera_center_batch.unsqueeze(1)#[N,P,3]
        dirs=torch.nn.functional.normalize(dirs,dim=-1)
        visible_color=wrapper.sh2rgb(self.actived_sh_degree,visible_sh,dirs)
        
        ### mean of 2d-gaussian ###
        ndc_pos_batch=wrapper.wrold2ndc(visible_positions,view_project_matrix)
        
        #### binning ###
        tile_start_index,sorted_pointId,sorted_tileId,radii=self.binning(ndc_pos_batch,visible_cov2d,visible_opacities,visible_points_num)

        #### raster ###
        if tiles is None:
            batch_size=view_matrix.shape[0]
            tiles=self.cached_tiles_map.reshape(1,-1).repeat((batch_size,1))
        tile_img,tile_transmitance=self.raster(ndc_pos_batch,visible_cov2d,visible_color,visible_opacities,tile_start_index,sorted_pointId,sorted_tileId,tiles)

        
        return tile_img,tile_transmitance.unsqueeze(2)
