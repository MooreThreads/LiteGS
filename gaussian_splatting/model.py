from gaussian_splatting.scene import GaussianScene
from training.arguments import OptimizationParams
from util import spherical_harmonics,cg_torch


import torch
import typing
import numpy as np
import math
from torch.cuda.amp import autocast

torch.ops.load_library("gaussian_splatting/submodules/gaussian_raster/build/Release/GaussianRaster.dll")

class GaussianSplattingModel:
    @torch.no_grad()
    def __init__(self,scene:GaussianScene,spatial_lr_scale):
        
        self._xyz = torch.nn.Parameter(torch.Tensor(np.pad(scene.position,((0,0),(0,1)),'constant',constant_values=1)).cuda())
        self._features_dc = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_dc).transpose(1, 2).contiguous().cuda())
        self._features_rest = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_rest).transpose(1, 2).contiguous().cuda())
        self._rotation = torch.nn.Parameter(torch.Tensor(scene.rotator).cuda())#torch.nn.functional.normalize
        self.spatial_lr_scale=spatial_lr_scale
        self.cached_cov3d=None

        #exp scale
        temp=torch.Tensor(scene.scale).cuda()
        #temp=temp.clamp_max(temp.mean()+temp.std()*2)
        self._scaling = torch.nn.Parameter(temp)#.exp 
        self._opacity = torch.nn.Parameter(torch.Tensor(scene.opacity).cuda())#.sigmoid
        return
    
    def get_params(self):
        return (self._xyz,self._features_dc,self._features_rest,self._rotation,self.spatial_lr_scale,self._scaling,self._opacity)
    
    def load_params(self,params_tuple):
        (self._xyz,self._features_dc,self._features_rest,self._rotation,self.spatial_lr_scale,self._scaling,self._opacity)=params_tuple
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

    def transform_to_cov3d(self,scaling_vec,rotator_vec)->torch.Tensor:
        #todo implement cuda: scale&rot -> matrix
        scale_matrix=torch.zeros((*(scaling_vec.shape[0:-1]),3,3),device='cuda')
        scale_matrix[...,0,0]=scaling_vec[...,0]
        scale_matrix[...,1,1]=scaling_vec[...,1]
        scale_matrix[...,2,2]=scaling_vec[...,2]

        rotation_matrix=torch.zeros((*(rotator_vec.shape[0:-1]),3,3),device='cuda')

        r=rotator_vec[...,0]
        x=rotator_vec[...,1]
        y=rotator_vec[...,2]
        z=rotator_vec[...,3]


        rotation_matrix[...,0,0]=1 - 2 * (y * y + z * z)
        rotation_matrix[...,0,1]=2 * (x * y + r * z)
        rotation_matrix[...,0,2]=2 * (x * z - r * y)

        rotation_matrix[...,1,0]=2 * (x * y - r * z)
        rotation_matrix[...,1,1]=1 - 2 * (x * x + z * z)
        rotation_matrix[...,1,2]=2 * (y * z + r * x)

        rotation_matrix[...,2,0]=2 * (x * z + r * y)
        rotation_matrix[...,2,1]=2 * (y * z - r * x)
        rotation_matrix[...,2,2]=1 - 2 * (x * x + y * y)

        M_matrix=torch.matmul(scale_matrix,rotation_matrix.transpose(-1,-2))
        cov3d=torch.matmul(M_matrix.transpose(-1,-2),M_matrix)
        return cov3d,M_matrix
    
    def transform_to_cov3d_faster(self,scaling_vec,rotator_vec)->torch.Tensor:
        rotation_matrix=torch.zeros((*(rotator_vec.shape[0:-1]),3,3),device='cuda')

        r=rotator_vec[...,0]
        x=rotator_vec[...,1]
        y=rotator_vec[...,2]
        z=rotator_vec[...,3]


        rotation_matrix[...,0,0]=1 - 2 * (y * y + z * z)
        rotation_matrix[...,0,1]=2 * (x * y + r * z)
        rotation_matrix[...,0,2]=2 * (x * z - r * y)

        rotation_matrix[...,1,0]=2 * (x * y - r * z)
        rotation_matrix[...,1,1]=1 - 2 * (x * x + z * z)
        rotation_matrix[...,1,2]=2 * (y * z + r * x)

        rotation_matrix[...,2,0]=2 * (x * z + r * y)
        rotation_matrix[...,2,1]=2 * (y * z - r * x)
        rotation_matrix[...,2,2]=1 - 2 * (x * x + y * y)

        transform_matrix=rotation_matrix*scaling_vec.unsqueeze(3)
        cov3d=TransformCovarianceMatrix.apply(transform_matrix)
        #cov3d=torch.matmul(transform_matrix.transpose(-1,-2),transform_matrix)
        return cov3d,transform_matrix
    
    def proj_cov3d_to_cov2d(self,cov3d,point_positions,view_matrix,camera_focal)->torch.Tensor:
        with torch.no_grad():
            t=torch.matmul(point_positions,view_matrix)
            
            J_trans=torch.zeros_like(cov3d,device='cuda')#view point mat3x3
            camera_focal=camera_focal.unsqueeze(1)
            tz_square=t[:,:,2]*t[:,:,2]
            J_trans[:,:,0,0]=camera_focal[:,:,0]/t[:,:,2]#focal x
            J_trans[:,:,1,1]=camera_focal[:,:,1]/t[:,:,2]#focal y
            J_trans[:,:,0,2]=-(camera_focal[:,:,0]*t[:,:,0])/tz_square
            J_trans[:,:,1,2]=-(camera_focal[:,:,1]*t[:,:,1])/tz_square
            #with autocast():
            view_matrix=view_matrix.unsqueeze(1)[:,:,0:3,0:3]
            T=J_trans@view_matrix.transpose(-1,-2)
        #T' x cov3d' x T
        cov2d=(T@cov3d.transpose(-1,-2)@T.transpose(-1,-2))[:,:,0:2,0:2]#forward backward 1s
        cov2d[:,:,0,0]+=0.3
        cov2d[:,:,1,1]+=0.3

        #todo cov3d -> DepthGaussianStd

        return cov2d

    
    def update_tiles_coord(self,image_size,tile_size):
        self.cached_image_size=image_size
        self.cached_image_size_tensor=torch.Tensor(image_size).cuda().int()
        self.cached_tile_size=tile_size
        self.cached_tiles_size=(math.ceil(image_size[0]/tile_size),math.ceil(image_size[1]/tile_size))
        self.cached_tiles_map=torch.arange(0,self.cached_tiles_size[0]*self.cached_tiles_size[1]).int().reshape(self.cached_tiles_size[1],self.cached_tiles_size[0]).cuda()+1#tile_id 0 is invalid
        return
    
    @torch.no_grad()
    def culling_and_sort(self,ndc_pos,translated_pos,limit_LURD=None):
        '''
        todo implement in cuda
        input: ViewMatrix,ProjMatrix
        output: sorted_visible_points,num_of_points
        '''
        if limit_LURD is None:
            culling_result=torch.any(ndc_pos[...,0:2]<-1.3,dim=2)|torch.any(ndc_pos[...,0:2]>1.3,dim=2)|(translated_pos[...,2]<=0)
        else:
            culling_result=torch.any(ndc_pos[...,0:2]<limit_LURD.unsqueeze(1)[...,0:2]*1.3,dim=2)|torch.any(ndc_pos[...,0:2]>limit_LURD.unsqueeze(1)[...,2:4]*1.3,dim=2)|(translated_pos[...,2]<=0)

        max_visible_points_num=(~culling_result).sum(1).max()
        threshhold=translated_pos[...,2].max()+1

        masked_depth=translated_pos[...,2]*(~culling_result)+threshhold*culling_result
        sorted_masked_depth,visible_point=torch.sort(masked_depth,1)
        point_index_mask=(sorted_masked_depth<threshhold)[...,:max_visible_points_num]
        points_num=point_index_mask.sum(1)
        visible_point=visible_point[...,:max_visible_points_num]*point_index_mask

        return visible_point,points_num
    
    def sample_by_visibility(self,visible_points_for_views,visible_points_num_for_views):
        #visible_cov3d=GaussianSplattingModel.__calc_cov3d(self._scaling[visible_points_for_views].exp(),torch.nn.functional.normalize(self._rotation[visible_points_for_views]))
        scales=self._scaling[visible_points_for_views].exp()
        rotators=torch.nn.functional.normalize(self._rotation[visible_points_for_views],dim=2)

        visible_positions=self._xyz[visible_points_for_views]
        visible_opacities=self._opacity[visible_points_for_views].sigmoid()
        visible_sh0=self._features_dc[visible_points_for_views]
        return scales,rotators,visible_positions,visible_opacities,visible_sh0

    
    @torch.no_grad()
    def binning(self,ndc:torch.Tensor,cov2d:torch.Tensor,valid_points_num:torch.Tensor,b_gather=False):
        
        tilesX=self.cached_tiles_size[0]
        tilesY=self.cached_tiles_size[1]
        tiles_num=tilesX*tilesY
        tile_size=self.cached_tile_size
        image_size=self.cached_image_size


        coordX=(ndc[:,:,0]+1.0)*0.5*image_size[0]-0.5
        coordY=(ndc[:,:,1]+1.0)*0.5*image_size[1]-0.5

        det=cov2d[:,:,0,0]*cov2d[:,:,1,1]-cov2d[:,:,0,1]*cov2d[:,:,0,1]
        mid=0.5*(cov2d[:,:,0,0]+cov2d[:,:,1,1])
        temp=(mid*mid-det).clamp_min(0.1).sqrt()
        pixel_radius=(3*(torch.max(mid+temp,mid-temp).sqrt())).ceil()
        
        L=((coordX-pixel_radius)/tile_size).floor().int().clamp(0,tilesX)
        U=((coordY-pixel_radius)/tile_size).floor().int().clamp(0,tilesY)
        R=((coordX+pixel_radius+tile_size-1)/tile_size).floor().int().clamp(0,tilesX)
        D=((coordY+pixel_radius+tile_size-1)/tile_size).floor().int().clamp(0,tilesY)

        #calculate params of allocation
        tiles_touched=(R-L)*(D-U)
        prefix_sum=tiles_touched.cumsum(1)
        total_tiles_num_batch=prefix_sum.gather(1,valid_points_num.unsqueeze(1)-1)
        allocate_size=total_tiles_num_batch.max().cpu()

        
        # allocate table and fill tile_id in it(uint 16)
        my_table=torch.ops.RasterBinning.duplicateWithKeys(L,U,R,D,valid_points_num,prefix_sum,int(allocate_size),int(tilesX))
        tileId_table:torch.Tensor=my_table[0]
        pointId_table:torch.Tensor=my_table[1]


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

        if b_gather:
            sorted_pointId=sorted_pointId.long()
            
        return tile_start_index,sorted_pointId,sorted_tileId,tiles_touched
    
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


        img,transmitance=GaussiansRaster.apply(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                               self.cached_tile_size,self.cached_tiles_size[0],self.cached_tiles_size[1],self.cached_image_size[1],self.cached_image_size[0])

        return img,transmitance
    
    def render(self,visible_points_num:torch.Tensor,visible_points:torch.Tensor,
               view_matrix:torch.Tensor,view_project_matrix:torch.Tensor,camera_focal:torch.Tensor,tiles:torch.Tensor=None,
               prebackward_func:typing.Callable=None):
        
        visible_scales,visible_rotators,visible_positions,visible_opacities,visible_sh0=self.sample_by_visibility(visible_points,visible_points_num)
        if prebackward_func is not None:
            prebackward_func(visible_scales,visible_rotators,visible_positions,visible_opacities,visible_sh0)

        ### (scale,rot)->3d covariance matrix->2d covariance matrix ###
        cov3d,transform_matrix=self.transform_to_cov3d_faster(visible_scales,visible_rotators)
        visible_cov2d=self.proj_cov3d_to_cov2d(cov3d,visible_positions,view_matrix,camera_focal)
        
        ### color ###
        visible_color=(visible_sh0*spherical_harmonics.C0+0.5).squeeze(2).clamp_min(0)
        
        ### mean of 2d-gaussian ###
        ndc_pos_batch=cg_torch.world_to_ndc(visible_positions,view_project_matrix)
        
        #### binning ###
        tile_start_index,sorted_pointId,sorted_tileId,tiles_touched=self.binning(ndc_pos_batch,visible_cov2d,visible_points_num)

        #### raster ###
        if tiles is None:
            batch_size=visible_points_num.shape[0]
            tiles=self.cached_tiles_map.reshape(1,-1).repeat((batch_size,1))
        tile_img,tile_transmitance=self.raster(ndc_pos_batch,visible_cov2d,visible_color,visible_opacities,tile_start_index,sorted_pointId,sorted_tileId,tiles)
        
        return tile_img,tile_transmitance.unsqueeze(2)

class TransformCovarianceMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,transforms:torch.Tensor):
        ctx.save_for_backward(transforms)
        cov=transforms.transpose(-1,-2)@transforms
        return cov
    
    @staticmethod
    def backward(ctx,CovarianceMatrixGradient:torch.Tensor):
        (transforms,)=ctx.saved_tensors
        return (2*transforms@CovarianceMatrixGradient)

class GaussiansRaster(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sorted_pointId:torch.Tensor,
        tile_start_index:torch.Tensor,
        mean2d:torch.Tensor,
        cov2d_inv:torch.Tensor,
        color:torch.Tensor,
        opacities:torch.Tensor,
        tiles:torch.Tensor,
        tile_size:int,
        tiles_num_x:int,
        tiles_num_y:int,
        img_h:int,
        img_w:int
    ):
        img,transmitance,lst_contributor=torch.ops.RasterBinning.rasterize_forward(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                                                                   tile_size,tiles_num_x,tiles_num_y,img_h,img_w)
        ctx.save_for_backward(sorted_pointId,tile_start_index,transmitance,lst_contributor,mean2d,cov2d_inv,color,opacities,tiles)
        ctx.arg_tile_size=tile_size
        ctx.tiles_num=(tiles_num_x,tiles_num_y,img_h,img_w)
        return img,transmitance
    
    @staticmethod
    def backward(ctx, grad_out_color:torch.Tensor, grad_out_transmitance):
        sorted_pointId,tile_start_index,transmitance,lst_contributor,mean2d,cov2d_inv,color,opacities,tiles=ctx.saved_tensors
        (tiles_num_x,tiles_num_y,img_h,img_w)=ctx.tiles_num
        tile_size=ctx.arg_tile_size



        grad_mean2d,grad_cov2d_inv,grad_color,grad_opacities=torch.ops.RasterBinning.rasterize_backward(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                                                                                        transmitance,lst_contributor,grad_out_color,
                                                                                                        tile_size,tiles_num_x,tiles_num_y,img_h,img_w)

        #print(grad_cov2d_inv[0,68362])

        grads = (
            None,
            None,
            grad_mean2d,
            grad_cov2d_inv,
            grad_color,
            grad_opacities,
            None,
            None,
            None,
            None,
            None,
            None
        )

        return grads