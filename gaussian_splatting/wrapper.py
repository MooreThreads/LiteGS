import torch
import typing
import numpy as np
import platform
from util import spherical_harmonics

plat = platform.system().lower()
if plat == 'windows':
    torch.ops.load_library("gaussian_splatting/submodules/gaussian_raster/build/Release/GaussianRaster.dll")
elif plat == 'linux':
    torch.ops.load_library("gaussian_splatting/submodules/gaussian_raster/build/libGaussianRaster.so")

##
## Create Transform Matrix from scale and quaternion
##
class CreateTransformMatrixFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,quaternion:torch.Tensor,scale:torch.Tensor):
        ctx.save_for_backward(quaternion,scale)
        transform_matrix=torch.ops.RasterBinning.createTransformMatrix_forward(quaternion,scale)
        return transform_matrix
    
    @staticmethod
    def backward(ctx,grad_transform_matrix:torch.Tensor):
        (quaternion,scale)=ctx.saved_tensors
        grad_quaternion,grad_scale=torch.ops.RasterBinning.createTransformMatrix_backward(grad_transform_matrix,quaternion,scale)
        return grad_quaternion,grad_scale
    
def create_transform_matrix(scaling_vec:torch.Tensor,rotator_vec:torch.Tensor)->torch.Tensor:

    def create_transform_matrix_internel_v2(scaling_vec:torch.Tensor,rotator_vec:torch.Tensor)->torch.Tensor:
        '''faster'''
        transform_matrix=CreateTransformMatrixFunc.apply(rotator_vec,scaling_vec)
        return transform_matrix

    def create_transform_matrix_internel_v1(scaling_vec:torch.Tensor,rotator_vec:torch.Tensor)->torch.Tensor:
        rotation_matrix=torch.zeros((3,3,rotator_vec.shape[-1]),device='cuda')

        r=rotator_vec[0]
        x=rotator_vec[1]
        y=rotator_vec[2]
        z=rotator_vec[3]


        rotation_matrix[0,0]=1 - 2 * (y * y + z * z)
        rotation_matrix[0,1]=2 * (x * y + r * z)
        rotation_matrix[0,2]=2 * (x * z - r * y)

        rotation_matrix[1,0]=2 * (x * y - r * z)
        rotation_matrix[1,1]=1 - 2 * (x * x + z * z)
        rotation_matrix[1,2]=2 * (y * z + r * x)

        rotation_matrix[2,0]=2 * (x * z + r * y)
        rotation_matrix[2,1]=2 * (y * z - r * x)
        rotation_matrix[2,2]=1 - 2 * (x * x + y * y)

        transform_matrix=rotation_matrix*scaling_vec.unsqueeze(1)
        return transform_matrix
    
    return create_transform_matrix_internel_v2(scaling_vec,rotator_vec)


##
## Create Rayspace Transform Matrix through first-order Taylor expansion.
##
def create_rayspace_transform(point_positions:torch.Tensor,view_matrix:torch.Tensor,camera_focal:torch.Tensor,bTranspose:bool=True)->torch.Tensor:
    #Keep no_grad. Auto gradient will make bad influence of xyz

    @torch.no_grad()
    def create_rayspace_transform_v1(point_positions:torch.Tensor,view_matrix:torch.Tensor,camera_focal:torch.Tensor,bTranspose:bool=True)->torch.Tensor:
        t=torch.matmul(view_matrix.transpose(-1,-2),point_positions)
        J=torch.zeros((t.shape[0],3,3,t.shape[-1]),device=t.device)#view point mat3x3
        camera_focal=camera_focal.unsqueeze(-1)
        tz_square=t[:,2]*t[:,2]
        J[:,0,0]=camera_focal[:,0]/t[:,2]#focal x
        J[:,1,1]=camera_focal[:,1]/t[:,2]#focal y
        if bTranspose:
            J[:,0,2]=-(camera_focal[:,0]*t[:,0])/tz_square
            J[:,1,2]=-(camera_focal[:,1]*t[:,1])/tz_square
        else:
            J[:,2,0]=-(camera_focal[:,0]*t[:,0])/tz_square
            J[:,2,1]=-(camera_focal[:,1]*t[:,1])/tz_square
        return J

    @torch.no_grad()
    def create_rayspace_transform_v2(point_positions:torch.Tensor,view_matrix:torch.Tensor,camera_focal:torch.Tensor,bTranspose:bool=True)->torch.Tensor:
        '''faster'''
        t=torch.matmul(view_matrix.transpose(-1,-2),point_positions)
        J=torch.ops.RasterBinning.jacobianRayspace(t,camera_focal,bTranspose)
        return J
    
    return create_rayspace_transform_v2(point_positions,view_matrix,camera_focal,bTranspose)

##
## Create Covariance matrix through transform matrix
##
class CreateCovarianceMatrixFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,transforms:torch.Tensor):
        ctx.save_for_backward(transforms)
        cov=transforms.transpose(-1,-2).contiguous()@transforms
        return cov
    
    @staticmethod
    def backward(ctx,CovarianceMatrixGradient:torch.Tensor):
        (transforms,)=ctx.saved_tensors
        return (2*transforms@CovarianceMatrixGradient)

def create_cov3d(transform_matrix:torch.Tensor)->torch.Tensor:
    '''
        transform_matrix:[P,3,3]
        trans^t @ trans
    '''
    def create_cov3d_internel_v1(transform_matrix:torch.Tensor)->torch.Tensor:
        cov3d=torch.matmul(transform_matrix.transpose(-1,-2),transform_matrix)
        return cov3d
    
    def create_cov3d_internel_v2(transform_matrix:torch.Tensor)->torch.Tensor:
        '''simplify the calculations in the backward phase.(The grad of Cov3d will be symmetric)'''
        cov3d=CreateCovarianceMatrixFunc.apply(transform_matrix)
        return cov3d

    return create_cov3d_internel_v2(transform_matrix)

###
### world position to ndc position
###
class World2NdcFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,position:torch.Tensor,view_project_matrix:torch.Tensor):
        #todo fix float
        hom_pos=torch.matmul(position.transpose(-1,-2).contiguous(),view_project_matrix).transpose(-1,-2).contiguous()#torch.matmul(view_project_matrix.transpose(-1,-2),position)
        repc_hom_w=1/(hom_pos[:,3:4]+1e-7)
        ndc_pos=hom_pos*repc_hom_w
        ctx.save_for_backward(view_project_matrix,ndc_pos,repc_hom_w)
        return ndc_pos
    
    @staticmethod
    def backward(ctx,grad_ndc_pos:torch.Tensor):
        (view_project_matrix,ndc_pos,repc_hom_w)=ctx.saved_tensors
        position_grad=torch.ops.RasterBinning.world2ndc_backword(view_project_matrix,ndc_pos,repc_hom_w,grad_ndc_pos)
        return (position_grad,None)

def wrold2ndc(position:torch.Tensor,view_project_matrix:torch.Tensor)->torch.Tensor:
    '''
    Override the backward. AutoGrad for world2ndc may lead to floating-point precision issues.
    '''
    return World2NdcFunc.apply(position,view_project_matrix)

###
### project the 3d-cov in world space to screen space
###
class ProjCov3dTo2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cov3d:torch.Tensor,transforms_t:torch.Tensor):
        ctx.save_for_backward(transforms_t)
        cov2d=transforms_t@cov3d@(transforms_t.transpose(-1,-2).contiguous())
        return cov2d
    
    @staticmethod
    def backward(ctx,cov2d_gradient:torch.Tensor):
        (transforms_t,)=ctx.saved_tensors
        N,P=transforms_t.shape[0:2]
        # cov3d_gradient=torch.zeros((N,P,3,3),device=transforms_t.device)
        # for i in range(0,3):
        #     for j in range(0,3):
        #         cov3d_gradient[:,:,i,j]=\
        #             (transforms_t[:,:,0,i]*transforms_t[:,:,0,j])*cov2d_gradient[:,:,0,0]\
        #             + (transforms_t[:,:,0,i]*transforms_t[:,:,1,j])*cov2d_gradient[:,:,0,1]\
        #             + (transforms_t[:,:,1,i]*transforms_t[:,:,0,j])*cov2d_gradient[:,:,1,0]\
        #             + (transforms_t[:,:,1,i]*transforms_t[:,:,1,j])*cov2d_gradient[:,:,1,1]
        temp_matrix_A=transforms_t[:,:,(0,0,1,1),:].transpose(-1,-2).contiguous()
        temp_matrix_B=(transforms_t[:,:,(0,1,0,1),:]*cov2d_gradient.reshape(N,P,-1,1)).contiguous()
        cov3d_gradient=temp_matrix_A@temp_matrix_B

        return cov3d_gradient,None
    
def project_3dcov_to_2d(cov3d:torch.Tensor,transforms_t:torch.Tensor)->torch.Tensor:
    def project_3dcov_to_2d_internel_v1(cov3d:torch.Tensor,transforms_t:torch.Tensor)->torch.Tensor:
        cov2d=(transforms_t@cov3d@transforms_t.transpose(-1,-2))
        return cov2d
    def project_3dcov_to_2d_internel_v2(cov3d:torch.Tensor,transforms_t:torch.Tensor)->torch.Tensor:
        '''simplify the calculations in the backward phase.'''
        return ProjCov3dTo2dFunc.apply(cov3d,transforms_t)
    
    return project_3dcov_to_2d_internel_v2(cov3d,transforms_t)

###
### The fastest version of Create cov2d. 
###
class Cov2dCreateV2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx,J:torch.Tensor,view_matrix:torch.Tensor,transform_matrix:torch.Tensor)->torch.Tensor:
        ctx.save_for_backward(J,view_matrix,transform_matrix)
        cov2d=torch.ops.RasterBinning.createCov2dDirectly_forward(J,view_matrix,transform_matrix)
        return cov2d
    
    @staticmethod
    def backward(ctx,grad_cov2d:torch.Tensor):
        (J,view_matrix,transform_matrix)=ctx.saved_tensors
        transform_matrix_grad=torch.ops.RasterBinning.createCov2dDirectly_backward(grad_cov2d,J,view_matrix,transform_matrix)
        return (None,None,transform_matrix_grad)
    
class Cov2dCreateV1Func(torch.autograd.Function):
    '''
    Used only for debugging and testing purposes
    '''
    @staticmethod
    def forward(ctx,J:torch.Tensor,view_matrix:torch.Tensor,transform_matrix:torch.Tensor)->torch.Tensor:
        view_rayspace_transform=(view_matrix[...,:3,:3].unsqueeze(1)@J[...,:2]).contiguous()
        T=transform_matrix@view_rayspace_transform
        cov2d=T.transpose(-1,-2).contiguous()@T
        ctx.save_for_backward(T,view_rayspace_transform)
        return cov2d
    
    @staticmethod
    def backward(ctx,grad_cov2d:torch.Tensor):
        (T,view_rayspace_transform)=ctx.saved_tensors
        dT=2*T@grad_cov2d
        dTrans=dT@(view_rayspace_transform.transpose(-1,-2).contiguous())
        return (None,None,dTrans)
    
def create_2dcov_directly(J:torch.Tensor,view_matrix:torch.Tensor,transform_matrix:torch.Tensor)->torch.Tensor:
    '''
    A faster function to calculate cov2d

    The usual method contains several matrix multiplications with a large batch number and a small K. Loading and writing these intermediate variables takes a lot of time.
    '''
    cov2d=Cov2dCreateV2Func.apply(J,view_matrix,transform_matrix)
    return cov2d

###
### the rasterization of 2d guassian.
###
class GaussiansRasterFunc(torch.autograd.Function):
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

def rasterize_2d_gaussian(
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
        img_w:int):
    
    return GaussiansRasterFunc.apply(
        sorted_pointId,
        tile_start_index,
        mean2d,
        cov2d_inv,
        color,
        opacities,
        tiles,
        tile_size,
        tiles_num_x,
        tiles_num_y,
        img_h,
        img_w)




###
### the rasterization of 2d guassian.
###
class SphericalHarmonicFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,deg:int, sh_base:torch.Tensor,sh_rest:torch.Tensor, dirs:torch.Tensor):
        ctx.save_for_backward(dirs)
        ctx.degree=deg
        ctx.sh_rest_dim=sh_rest.shape[0]
        rgb=torch.ops.RasterBinning.sh2rgb_forward(deg,sh_base,sh_rest,dirs)
        return rgb
    
    @staticmethod
    def backward(ctx, grad_rgb):
        (dirs,)=ctx.saved_tensors
        degree=ctx.degree
        sh_rest_dim=ctx.sh_rest_dim
        sh_base_grad,sh_reset_grad=torch.ops.RasterBinning.sh2rgb_backward(degree,grad_rgb,sh_rest_dim,dirs)


        return None,sh_base_grad,sh_reset_grad,None

def sh2rgb(deg:int, sh_base:torch.Tensor,sh_rest:torch.Tensor, dirs:torch.Tensor):

    def sh2rgb_internel_v1(deg:int, sh_base:torch.Tensor,sh_rest:torch.Tensor, dirs:torch.Tensor):
        return spherical_harmonics.eval_sh(deg,torch.cat((sh_base,sh_rest),dim=0),dirs).clamp_min(0)
    
    def sh2rgb_internel_v2(deg:int, sh_base:torch.Tensor,sh_rest:torch.Tensor, dirs:torch.Tensor):
        return SphericalHarmonicFunc.apply(deg,sh_base,sh_rest,dirs).clamp_min(0)
    
    return sh2rgb_internel_v2(deg,sh_base,sh_rest,dirs)


###
### eigh[no grad] and inverse[grad] the matrix.
###
class EighAndInverse2x2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input_matrix:torch.Tensor):
        val,vec,inverse_matrix=torch.ops.RasterBinning.eigh_and_inv_2x2matrix_forward(input_matrix)
        ctx.save_for_backward(inverse_matrix)
        return val,vec,inverse_matrix
    
    @staticmethod
    def backward(ctx,val_grad,vec_grad,inverse_matrix_grad):
        (inverse_matrix,)=ctx.saved_tensors
        matrix_grad=torch.ops.RasterBinning.inv_2x2matrix_backward(inverse_matrix,inverse_matrix_grad)
        return matrix_grad
    
def eigh_and_inverse_cov2d(cov2d:torch.Tensor):

    def eigh_and_inverse_cov2d_internel_v1(cov2d:torch.Tensor):
        det=cov2d[:,0,0]*cov2d[:,1,1]-cov2d[:,0,1]*cov2d[:,1,0]
        with torch.no_grad():
            mid=0.5*(cov2d[:,0,0]+cov2d[:,1,1])
            temp=(mid*mid-det).clamp_min(1e-9).sqrt()
            eigen_val=torch.cat(((mid-temp).unsqueeze(1),(mid+temp).unsqueeze(1)),dim=1)
            eigen_vec_y=((eigen_val-cov2d[:,0,0].unsqueeze(1))/cov2d[:,0,1].unsqueeze(1))
            eigen_vec=torch.cat((torch.ones_like(eigen_vec_y).unsqueeze(-2),eigen_vec_y.unsqueeze(-2)),dim=-2)
            eigen_vec=torch.nn.functional.normalize(eigen_vec,dim=-2)
        reci_det=1/(det+1e-7)
        cov2d_inv=torch.zeros_like(cov2d)
        cov2d_inv[:,0,1]=-cov2d[:,0,1]*reci_det
        cov2d_inv[:,1,0]=-cov2d[:,1,0]*reci_det
        cov2d_inv[:,0,0]=cov2d[:,1,1]*reci_det
        cov2d_inv[:,1,1]=cov2d[:,0,0]*reci_det
        return eigen_val,eigen_vec,cov2d_inv
    
    def eigh_and_inverse_cov2d_internel_v2(cov2d:torch.Tensor):
        return EighAndInverse2x2Func.apply(cov2d)

    return eigh_and_inverse_cov2d_internel_v2(cov2d)

###
### compact params
###

class CompactVisibleParamsFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,visible_mask:torch.Tensor,
                position:torch.Tensor,scale:torch.Tensor,rotation:torch.Tensor,
                sh_base:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        visible_mask_cumsum=visible_mask.cumsum(0)
        visible_chunks_num=int(visible_mask_cumsum[-1].cpu())
        compacted_pos,compacted_scale,compacted_rot,compacted_sh_base,compacted_sh_rest,compacted_opacity,reverse_map=torch.ops.RasterBinning.compact_visible_params_forward(
            visible_chunks_num,
            visible_mask,
            visible_mask_cumsum,
            position,
            scale,
            rotation,
            sh_base,
            sh_rest,
            opacity)
        ctx.save_for_backward(reverse_map)
        ctx.chunk_num=position.shape[0]
        ctx.chunk_size=position.shape[-1]
        return compacted_pos,compacted_scale,compacted_rot,compacted_sh_base,compacted_sh_rest,compacted_opacity
    
    @staticmethod
    def backward(ctx,compacted_pos_grad:torch.Tensor,compacted_scale_grad:torch.Tensor,compacted_rot_grad:torch.Tensor,compacted_sh_base_grad:torch.Tensor,compacted_sh_rest_grad:torch.Tensor,compacted_opacity_grad:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        (reverse_map,)=ctx.saved_tensors
        chunk_num=ctx.chunk_num
        chunk_size=ctx.chunk_size
        grads=torch.ops.RasterBinning.compact_visible_params_backward(chunk_num,chunk_size,reverse_map,compacted_pos_grad,compacted_scale_grad,compacted_rot_grad,compacted_sh_base_grad,compacted_sh_rest_grad,compacted_opacity_grad)
        return None,*grads

def compact_visible_params(visible_mask:torch.Tensor,position:torch.Tensor,scale:torch.Tensor,rotation:torch.Tensor,sh_base:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    compacted_pos,compacted_scale,compacted_rot,compacted_sh_base,compacted_sh_rest,compacted_opacity=CompactVisibleParamsFunc.apply(
        visible_mask,
        position,scale,rotation,
        sh_base,sh_rest,opacity)
    
    # compacted_pos=position[visible_mask].transpose(0,1).reshape(4,-1)
    # compacted_scale=scale[visible_mask].transpose(0,1).reshape(3,-1)
    # compacted_rot=rotation[visible_mask].transpose(0,1).reshape(4,-1)
    # compacted_sh_base=sh_base[visible_mask].transpose(0,1).transpose(1,2).reshape(1,3,-1)
    # compacted_sh_rest=sh_rest[visible_mask].transpose(0,1).transpose(1,2).reshape(sh_rest.shape[1],3,compacted_sh_base.shape[-1])
    # compacted_opacity=opacity[visible_mask].transpose(0,1).reshape(1,-1)



    return compacted_pos,compacted_scale,compacted_rot,compacted_sh_base,compacted_sh_rest,compacted_opacity