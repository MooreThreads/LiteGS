from gaussian_splatting.gaussian_util import GaussianScene,View
from training.arguments import OptimizationParams

import torch
import typing
import numpy as np
import math
from torch.cuda.amp import autocast

class GaussianSplattingModel:
    def __init__(self,optimizer,scene:GaussianScene,spatial_lr_scale):
        torch.ops.load_library("gaussian_splatting/submodules/rasterbinning/rasterbinning/build/Release/RasterBinning.dll")
        self.optimizer=optimizer
        self._xyz = torch.nn.Parameter(torch.Tensor(np.pad(scene.position,((0,0),(0,1)),'constant',constant_values=1)).requires_grad_(True))
        self._features_dc = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_dc).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(torch.Tensor(scene.sh_coefficient_rest).transpose(1, 2).contiguous().requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.Tensor(scene.rotator).requires_grad_(True))
        self.spatial_lr_scale=spatial_lr_scale
        self.cached_cov3d=None

        #exp scale
        self._scaling = torch.nn.Parameter(torch.Tensor(scene.scale).exp_().requires_grad_(True))
        #sigmoid(opacity)
        self._opacity = torch.nn.Parameter(torch.Tensor(scene.opacity).sigmoid_().requires_grad_(True))
        return
    
    def cuda(self):
        self._xyz=self._xyz.cuda()
        self._features_dc=self._features_dc.cuda()
        self._features_rest=self._features_rest.cuda()
        self._scaling=self._scaling.cuda()
        self._rotation=self._rotation.cuda()
        self._opacity=self._opacity.cuda()

    def __calc_cov3d(scaling_vec,rotator_vec):
        scale_matrix=torch.zeros((scaling_vec.shape[0],3,3),device='cuda')
        scale_matrix[:,0,0]=scaling_vec[:,0]
        scale_matrix[:,1,1]=scaling_vec[:,1]
        scale_matrix[:,2,2]=scaling_vec[:,2]

        rotation_matrix=torch.zeros((rotator_vec.shape[0],3,3),device='cuda')

        r=rotator_vec[:,0]
        x=rotator_vec[:,1]
        y=rotator_vec[:,2]
        z=rotator_vec[:,3]


        rotation_matrix[:,0,0]=1 - 2 * (y * y + z * z)
        rotation_matrix[:,0,1]=2 * (x * y + r * z)
        rotation_matrix[:,0,2]=2 * (x * z - r * y)

        rotation_matrix[:,1,0]=2 * (x * y - r * z)
        rotation_matrix[:,1,1]=1 - 2 * (x * x + z * z)
        rotation_matrix[:,1,2]=2 * (y * z + r * x)

        rotation_matrix[:,2,0]=2 * (x * z + r * y)
        rotation_matrix[:,2,1]=2 * (y * z - r * x)
        rotation_matrix[:,2,2]=1 - 2 * (x * x + y * y)

        M_matrix=torch.matmul(scale_matrix,rotation_matrix)
        cov3d=torch.matmul(M_matrix.transpose(1,2),M_matrix)
        return cov3d

    
    def __calc_cov2d(cov3d,point_positions,view_matrix,camera_focal):
        #assert(cov3d.shape[0]==point_positions.shape[0] and cov3d.shape[1]==point_positions.shape[1])

        t=torch.matmul(point_positions,view_matrix)
        
        J_transposed=torch.zeros_like(cov3d,device='cuda')#view point mat3x3
        camera_focal=camera_focal.unsqueeze(1)
        temp=camera_focal/t[:,:,2]
        tz_square=t[:,:,2]*t[:,:,2]+1e-9
        J_transposed[:,:,0,0]=temp
        J_transposed[:,:,1,1]=temp
        J_transposed[:,:,0,2]=-(camera_focal*t[:,:,0])/tz_square
        J_transposed[:,:,1,2]=-(camera_focal*t[:,:,1])/tz_square
        #with autocast():
        view_matrix=view_matrix.unsqueeze(1)[:,:,0:3,0:3]
        T_trans=torch.matmul(J_transposed,view_matrix.transpose(2,3))
        #T' x cov3d' x T
        cov2d=torch.matmul(torch.matmul(T_trans,cov3d.transpose(2,3)),T_trans.transpose(2,3))

        return cov2d[:,:,0:2,0:2]

    def update_cov3d(self):
        self.cached_cov3d=GaussianSplattingModel.__calc_cov3d(self._scaling,self._rotation)
        return
    
    def update_tiles_coord(self,image_size,tile_size):
        self.cached_image_size=image_size
        self.cached_image_size_tensor=torch.Tensor(image_size).cuda()
        self.cached_tile_size=tile_size
        self.cached_tiles_size=(math.ceil(image_size[0]/tile_size),math.ceil(image_size[1]/tile_size))

        #tiles_sizeX tiles_sizeY tile_size tile_size 2
        row=np.arange(0,tile_size).reshape((1,tile_size)).repeat(tile_size,axis=0)
        col=np.arange(0,tile_size).reshape((tile_size,1)).repeat(tile_size,axis=1)

        tile_row=np.arange(0,self.cached_tiles_size[1]).reshape((1,self.cached_tiles_size[1])).repeat(self.cached_tiles_size[0],axis=0)*tile_size
        tile_col=np.arange(0,self.cached_tiles_size[0]).reshape((self.cached_tiles_size[0],1)).repeat(self.cached_tiles_size[1],axis=1)*tile_size

        row=row.reshape((1,1,tile_size,tile_size,1))
        col=col.reshape((1,1,tile_size,tile_size,1))

        tile_row=tile_row.reshape(self.cached_tiles_size[0],self.cached_tiles_size[1],1,1,1)
        tile_col=tile_col.reshape(self.cached_tiles_size[0],self.cached_tiles_size[1],1,1,1)

        coords_x=tile_row+row
        coords_y=tile_col+col
        coords=np.concatenate((coords_x,coords_y),axis=4)
        self.cached_coords=torch.Tensor(coords).int().cuda().reshape((self.cached_tiles_size[0]*self.cached_tiles_size[1],tile_size,tile_size,2))

        coods_in_tile=np.concatenate((row,col),axis=4)
        self.coords_in_tile=torch.Tensor(coods_in_tile).int().cuda()

        return
    

    def worldpose_2_ndc(self,pos,view_matrix,project_matrix):
        translated_pos=torch.matmul(pos,view_matrix)
        hom_pos=torch.matmul(translated_pos,project_matrix)
        ndc_pos=hom_pos/(hom_pos[:,:,3:4]+1e-6)
        return ndc_pos

    @torch.no_grad()
    def culling_and_sort(self,ndc_pos):
        '''
        todo implement in cuda
        input: ViewMatrix,ProjMatrix
        output: sorted_visible_points,num_of_points
        '''
        culling_result=torch.any(ndc_pos[:,:,0:2]<-1.3,dim=2)|torch.any(ndc_pos[:,:,0:2]>1.3,dim=2)|(ndc_pos[:,:,2]<=0)

        max_visible_points_num=(~culling_result).sum(1).max()

        masked_depth=ndc_pos[:,:,2]*(~culling_result)+100*culling_result
        sorted_masked_depth,visible_point=torch.sort(masked_depth,1)
        point_index_mask=(sorted_masked_depth<100)[:,:max_visible_points_num]
        points_num=point_index_mask.sum(1)
        visible_point=visible_point[:,:max_visible_points_num]*point_index_mask

        return visible_point,points_num
    
    def sample_by_visibility(self,visible_points_for_views,visible_points_num_for_views):
        visible_cov3d=self.cached_cov3d[visible_points_for_views]
        visible_positions=self._xyz[visible_points_for_views]
        visible_opacities=self._opacity[visible_points_for_views]
        visible_sh0=self._features_dc[visible_points_for_views]
        return visible_cov3d,visible_positions,visible_opacities,visible_sh0
    
    def cov2d_after_culling(self,visible_cov3d,visible_positions,
                            view_matrix,camera_focal):
        '''
        output: conv2d tensor  
            tensor size: ViewsNum x PointsNum(max num: 8M) x 2 x 2 
            memory size: 32 x 8M x 2 x 2 x sizeof(float)    ->  4G(backward +4G)

        '''
        cov2d=GaussianSplattingModel.__calc_cov2d(visible_cov3d,visible_positions,view_matrix,camera_focal)

        return cov2d

    
    @torch.no_grad
    def tile_raster(self,ndc:torch.Tensor,cov2d:torch.Tensor,valid_points_num:torch.Tensor):
        '''
        todo implement in cuda

        output: 
            tensor size:    ViewsNum x TilesNum x MaxPointsNumInTile    
            memory size:    32 x [2k,8k] x 4k x sizeof(int) 1G-4G
        '''
        tilesX=self.cached_tiles_size[0]
        tilesY=self.cached_tiles_size[1]
        tiles_num=tilesX*tilesY
        tile_size=self.cached_tile_size
        image_size=self.cached_image_size


        coordX=(ndc[:,:,0]+1.0)*0.5*image_size[0]
        coordY=(ndc[:,:,1]+1.0)*0.5*image_size[1]

        det=cov2d[:,:,0,0]*cov2d[:,:,1,1]-cov2d[:,:,0,1]*cov2d[:,:,0,1]
        mid=0.5*(cov2d[:,:,0,0]+cov2d[:,:,1,1])
        temp=(mid*mid-det).clamp_min(0.1).sqrt()
        pixel_radius=3*(torch.max(mid+temp,mid-temp).sqrt())
        
        L=((coordX-pixel_radius)/tile_size).floor().int().clamp(0,tilesX-1)
        U=((coordY-pixel_radius)/tile_size).floor().int().clamp(0,tilesY-1)
        R=((coordX+pixel_radius)/tile_size).ceil().int().clamp(0,tilesX-1)
        D=((coordY+pixel_radius)/tile_size).ceil().int().clamp(0,tilesY-1)

        #calc allocate params
        tiles_touched=(R-L+1)*(D-U+1)
        prefix_sum=tiles_touched.cumsum(1)
        total_tiles_num_batch=prefix_sum.gather(1,valid_points_num.unsqueeze(1)-1)
        allocate_size=total_tiles_num_batch.max()

        
        # allocate table and fill tile_id in it(uint 16)
        # !!!!!implement in cuda!!!!!
        # input LURD,valid_points_num,prefix_sum,allocate_size,TilesSizeX
        # output tileId_table[ViewsNum,AllocateSize] (p.s. fill InvalidTileId(0xffff) in talbe)
        my_table=torch.ops.RasterBinning.duplicateWithKeys(L,U,R,D,valid_points_num,prefix_sum,int(allocate_size.cpu()),int(tilesX))
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
        tile_start_index=torch.ops.RasterBinning.tileRange(sorted_tileId,int(allocate_size.cpu()),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)
            
        return tile_start_index,sorted_pointId.long(),sorted_tileId.long()

    def pixel_raster_in_tile(self,ndc_pos:torch.Tensor,cov2d:torch.Tensor,tile_start_index:torch.Tensor,sorted_pointId:torch.Tensor,sorted_tileId:torch.Tensor):
        view_num=ndc_pos.shape[0]

        cov2d_inv=torch.linalg.inv(cov2d)

        mean2d=(ndc_pos[:,:,0:2]+1.0)*0.5*self.cached_image_size_tensor

        #screen coords to tile local coords
        mean2d=mean2d.gather(1,sorted_pointId.unsqueeze(2).repeat(1,1,2))
        shifted_mean2d=mean2d-sorted_tileId.unsqueeze(2)

        # shifted_meanX=shifted_meanX.reshape(view_num,-1,1,1,1)
        #shifted_meanY=shifted_meanY.reshape(view_num,-1,1,1)
        #dx=self.coords_in_tile-shifted_meanX
        #dy=self.coords_in_tile[:,:,:,:,1]-shifted_meanY





        return
    
    
