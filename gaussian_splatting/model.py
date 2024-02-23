from gaussian_splatting.gaussian_util import GaussianScene,View
from training.arguments import OptimizationParams

import torch
import typing
import numpy as np
import math
from torch.cuda.amp import autocast

torch.ops.load_library("gaussian_splatting/submodules/rasterbinning/rasterbinning/build/Release/RasterBinning.dll")

class GaussianSplattingModel:
    @torch.no_grad
    def __init__(self,scene:GaussianScene,spatial_lr_scale):
        
        self._xyz = torch.nn.Parameter(torch.Tensor(np.pad(scene.position,((0,0),(0,1)),'constant',constant_values=1)).cuda())
        self._features_dc = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_dc).transpose(1, 2).contiguous().cuda())
        self._features_rest = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_rest).transpose(1, 2).contiguous().cuda())
        self._rotation = torch.nn.Parameter(torch.Tensor(scene.rotator).cuda())#torch.nn.functional.normalize
        self.spatial_lr_scale=spatial_lr_scale
        self.cached_cov3d=None

        #exp scale
        self._scaling = torch.nn.Parameter(torch.Tensor(scene.scale).cuda())#.exp 
        #sigmoid(opacity)
        self._opacity = torch.nn.Parameter(torch.Tensor(scene.opacity).cuda())#.sigmoid
        return
    
    def cuda(self):
        return

    def __calc_cov3d(scaling_vec,rotator_vec):
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

        M_matrix=torch.matmul(scale_matrix,rotation_matrix)
        cov3d=torch.matmul(M_matrix.transpose(-1,-2),M_matrix)
        return cov3d
    
    def __calc_cov3d_faster(scaling_vec,rotator_vec):
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

        M_matrix_3d=rotation_matrix*scaling_vec.unsqueeze(3)
        #cov3d=torch.matmul(M_matrix_3d.transpose(-1,-2),M_matrix_3d)
        return M_matrix_3d
    
    def __calc_cov2d(cov3d,point_positions,view_matrix,camera_focal):
        #assert(cov3d.shape[0]==point_positions.shape[0] and cov3d.shape[1]==point_positions.shape[1])

        t=torch.matmul(point_positions,view_matrix)
        
        J_transposed=torch.zeros_like(cov3d,device='cuda')#view point mat3x3
        camera_focal=camera_focal.unsqueeze(1)
        tz_square=t[:,:,2]*t[:,:,2]
        J_transposed[:,:,0,0]=camera_focal[:,:,0]/t[:,:,2]#focal x
        J_transposed[:,:,1,1]=camera_focal[:,:,1]/t[:,:,2]#focal y
        J_transposed[:,:,0,2]=-(camera_focal[:,:,0]*t[:,:,0])/tz_square
        J_transposed[:,:,1,2]=-(camera_focal[:,:,1]*t[:,:,1])/tz_square
        #with autocast():
        view_matrix=view_matrix.unsqueeze(1)[:,:,0:3,0:3]
        T_trans=torch.matmul(J_transposed,view_matrix.transpose(2,3))
        #T' x cov3d' x T
        cov2d=torch.matmul(torch.matmul(T_trans,cov3d.transpose(2,3)),T_trans.transpose(2,3))

        return cov2d[:,:,0:2,0:2]
    
    def __calc_cov2d_faster(M_matrix_3d,point_positions,view_matrix,camera_focal):
        #assert(cov3d.shape[0]==point_positions.shape[0] and cov3d.shape[1]==point_positions.shape[1])

        t=torch.matmul(point_positions,view_matrix)
        
        J=torch.zeros_like(M_matrix_3d,device='cuda')#view point mat3x3
        camera_focal=camera_focal.unsqueeze(1)
        tz_square=t[:,:,2]*t[:,:,2]
        J[:,:,0,0]=camera_focal[:,:,0]/t[:,:,2]#focal x
        J[:,:,1,1]=camera_focal[:,:,1]/t[:,:,2]#focal y
        J[:,:,2,0]=-(camera_focal[:,:,0]*t[:,:,0])/tz_square
        J[:,:,2,1]=-(camera_focal[:,:,1]*t[:,:,1])/tz_square
        #with autocast():
        view_matrix=view_matrix.unsqueeze(1)[:,:,0:3,0:3]
        T=view_matrix@J
        temp=M_matrix_3d@T
        #T' x cov3d' x T
        cov2d=temp.transpose(-1,-2)@temp
        #cov2d_slow=T.transpose(-1,-2)@(M_matrix_3d.transpose(-1,-2)@M_matrix_3d).transpose(-1,-2)@T

        return cov2d[:,:,0:2,0:2]

    
    def update_tiles_coord(self,image_size,tile_size):
        self.cached_image_size=image_size
        self.cached_image_size_tensor=torch.Tensor(image_size).cuda()
        self.cached_tile_size=tile_size
        self.cached_tiles_size=(math.ceil(image_size[0]/tile_size),math.ceil(image_size[1]/tile_size))

        return
    

    def worldpose_2_ndc(self,pos,view_matrix,project_matrix):
        translated_pos=torch.matmul(pos,view_matrix)
        hom_pos=torch.matmul(translated_pos,project_matrix)
        ndc_pos=hom_pos/(hom_pos[:,:,3:4]+1e-6)
        return ndc_pos,translated_pos

    @torch.no_grad()
    def culling_and_sort(self,ndc_pos,translated_pos):
        '''
        todo implement in cuda
        input: ViewMatrix,ProjMatrix
        output: sorted_visible_points,num_of_points
        '''
        culling_result=torch.any(ndc_pos[:,:,0:2]<-1.3,dim=2)|torch.any(ndc_pos[:,:,0:2]>1.3,dim=2)|(translated_pos[:,:,2]<=0)

        max_visible_points_num=(~culling_result).sum(1).max()
        threshhold=translated_pos[:,:,2].max()+1

        masked_depth=translated_pos[:,:,2]*(~culling_result)+threshhold*culling_result
        sorted_masked_depth,visible_point=torch.sort(masked_depth,1)
        point_index_mask=(sorted_masked_depth<threshhold)[:,:max_visible_points_num]
        points_num=point_index_mask.sum(1)
        visible_point=visible_point[:,:max_visible_points_num]*point_index_mask

        return visible_point,points_num
    
    def sample_by_visibility(self,visible_points_for_views,visible_points_num_for_views):
        #visible_cov3d=GaussianSplattingModel.__calc_cov3d(self._scaling[visible_points_for_views].exp(),torch.nn.functional.normalize(self._rotation[visible_points_for_views]))
        M_matrix_3d=GaussianSplattingModel.__calc_cov3d_faster(self._scaling[visible_points_for_views].exp(),torch.nn.functional.normalize(self._rotation[visible_points_for_views]))

        visible_positions=self._xyz[visible_points_for_views]
        visible_opacities=self._opacity[visible_points_for_views].sigmoid()
        visible_sh0=self._features_dc[visible_points_for_views]
        return M_matrix_3d,visible_positions,visible_opacities,visible_sh0
    
    def proj_cov3d_to_cov2d(self,M_matrix_3d,visible_positions,
                            view_matrix,camera_focal):
        '''
        output: conv2d tensor  
            tensor size: ViewsNum x PointsNum(max num: 8M) x 2 x 2 
            memory size: 32 x 8M x 2 x 2 x sizeof(float)    ->  4G(backward +4G)

        '''
        #cov2d=GaussianSplattingModel.__calc_cov2d(visible_cov3d,visible_positions,view_matrix,camera_focal)
        cov2d=GaussianSplattingModel.__calc_cov2d_faster(M_matrix_3d,visible_positions,view_matrix,camera_focal)

        return cov2d

    
    @torch.no_grad
    def tile_raster(self,ndc:torch.Tensor,cov2d:torch.Tensor,valid_points_num:torch.Tensor,b_gather=False):
        
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
        R=((coordX+pixel_radius)/tile_size).ceil().int().clamp(0,tilesX)
        D=((coordY+pixel_radius)/tile_size).ceil().int().clamp(0,tilesY)

        #calc allocate params
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

        #calc range
        tile_start_index=torch.ops.RasterBinning.tileRange(sorted_tileId,int(allocate_size),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)

        if b_gather:
            sorted_pointId=sorted_pointId.long()
            
        return tile_start_index,sorted_pointId,sorted_tileId,tiles_touched
    
    def pixel_raster_in_tile(self,ndc_pos:torch.Tensor,cov2d:torch.Tensor,color:torch.Tensor,opacities:torch.Tensor,tile_start_index:torch.Tensor,sorted_pointId:torch.Tensor,sorted_tileId:torch.Tensor,b_gather=False):

        cov2d_inv=torch.linalg.inv(cov2d)
        mean2d=(ndc_pos[:,:,0:2]+1.0)*0.5*self.cached_image_size_tensor-0.5

        if b_gather:
            gathered_mean2d=mean2d.gather(1,sorted_pointId.unsqueeze(2).repeat(1,1,2))#[batch,N,2]
            #screen coords to tile local coords
            gathered_mean2d[:,:,0]-=(sorted_tileId-1)%self.cached_tiles_size[0]*self.cached_tile_size
            gathered_mean2d[:,:,1]-=((sorted_tileId-1)/self.cached_tiles_size[0]).int()*self.cached_tile_size
            gathered_cov2d_inv=cov2d_inv.gather(1,sorted_pointId.unsqueeze(2).unsqueeze(2).repeat(1,1,2,2))
            gathered_color=color.gather(1,sorted_pointId.unsqueeze(2).repeat(1,1,3))
            gathered_opacities=opacities.gather(1,sorted_pointId.unsqueeze(2))
            img,transmitance=gathered_mean2d,gathered_color
            img,transmitance,lst_contributor=torch.ops.RasterBinning.rasterize_forward_gathered(tile_start_index,gathered_mean2d,gathered_cov2d_inv,gathered_color,gathered_opacities,self.cached_tile_size,self.cached_tiles_size[0]*self.cached_tiles_size[1])

        else:
            img,transmitance=GaussiansRaster.apply(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,self.cached_tile_size,self.cached_tiles_size[0],self.cached_tiles_size[1])

        return img,transmitance
    
    
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
        tile_size:int,
        tiles_num_x:int,
        tiles_num_y:int
    ):
        img,transmitance,lst_contributor=torch.ops.RasterBinning.rasterize_forward(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tile_size,tiles_num_x,tiles_num_y)
        ctx.save_for_backward(sorted_pointId,tile_start_index,transmitance,lst_contributor,mean2d,cov2d_inv,color,opacities)
        ctx.arg_tile_size=tile_size
        ctx.tiles_num=(tiles_num_x,tiles_num_y)
        return img,transmitance
    
    @staticmethod
    def backward(ctx, grad_out_color, grad_out_transmitance):
        sorted_pointId,tile_start_index,transmitance,lst_contributor,mean2d,cov2d_inv,color,opacities=ctx.saved_tensors
        (tiles_num_x,tiles_num_y)=ctx.tiles_num
        tile_size=ctx.arg_tile_size

        grad_mean2d,grad_cov2d_inv,grad_color,grad_opacities=torch.ops.RasterBinning.rasterize_backward(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,transmitance,lst_contributor,grad_out_color,tile_size,tiles_num_x,tiles_num_y)

        grads = (
            None,
            None,
            grad_mean2d,
            grad_cov2d_inv,
            grad_color,
            grad_opacities,
            None,
            None,
            None
        )

        return grads