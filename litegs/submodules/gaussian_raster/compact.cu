#ifndef __CUDACC__
    #define __CUDACC__
    #define __NVCC__
#endif
#include "cuda_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
namespace cg = cooperative_groups;

#include <ATen/core/TensorAccessor.h>

#include "cuda_errchk.h"
#include"compact.h"


__global__ void create_viewproj_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> view_params,    //[views_num,7] 
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> recp_tan_half_fov_x,    //[1]
    int img_h, int img_w, float z_near, float z_far,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,    //[viewsnum,4,4] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> proj_matrix,    //[viewsnum,4,4] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> viewproj_matrix,    //[viewsnum,4,4] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> frustumplane    //[viewsnum,6,4] 
)
{
    int view_id = blockIdx.x*blockDim.x+threadIdx.x;
    if (view_id < view_params.size(0))
    {
        //init view_mat
        float r = view_params[view_id][0];
        float x = view_params[view_id][1];
        float y = view_params[view_id][2];
        float z = view_params[view_id][3];
        float recp = rsqrtf(r * r + x * x + y * y + z * z);
        r *= recp;
        x *= recp;
        y *= recp;
        z *= recp;
        float view[4][4] = {
            {1 - 2 * (y * y + z * z),2 * (x * y + r * z),2 * (x * z - r * y),0},
            {2 * (x * y - r * z),1 - 2 * (x * x + z * z),2 * (y * z + r * x),0},
            {2 * (x * z + r * y),2 * (y * z - r * x),1 - 2 * (x * x + y * y),0},
            {view_params[view_id][4],view_params[view_id][5],view_params[view_id][6],1.0f}
        };
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                view_matrix[view_id][i][j] = view[i][j];
            }
        }
        //init proj_mat (transposed for row-vector convention)
        float proj_00 = recp_tan_half_fov_x[0];
        float proj_11 = proj_00 * img_w / img_h;
        float proj[4][4] = {
            {proj_00,0,0,0},
            {0,proj_11,0,0},
            {0,0,z_far / (z_far - z_near),1},  // Transposed: last row instead of last column
            {0,0,-z_far * z_near / (z_far - z_near),0}  // Transposed: last column instead of last row
        };
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                proj_matrix[view_id][i][j] = proj[i][j];
            }
        }
        //init viewproj
        float viewproj[4][4] = {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        };
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                float temp = 0.0f;
                for (int k = 0; k < 4; k++)
                {
                    temp += view[i][k] * proj[k][j];
                }
                viewproj[i][j] = temp;
                viewproj_matrix[view_id][i][j] = viewproj[i][j];
            }
        }
        //init planes
        frustumplane[view_id][0][0] = viewproj[0][3] + viewproj[0][0];
        frustumplane[view_id][0][1] = viewproj[1][3] + viewproj[1][0];
        frustumplane[view_id][0][2] = viewproj[2][3] + viewproj[2][0];
        frustumplane[view_id][0][3] = viewproj[3][3] + viewproj[3][0];

        frustumplane[view_id][1][0] = viewproj[0][3] - viewproj[0][0];
        frustumplane[view_id][1][1] = viewproj[1][3] - viewproj[1][0];
        frustumplane[view_id][1][2] = viewproj[2][3] - viewproj[2][0];
        frustumplane[view_id][1][3] = viewproj[3][3] - viewproj[3][0];

        frustumplane[view_id][2][0] = viewproj[0][3] + viewproj[0][1];
        frustumplane[view_id][2][1] = viewproj[1][3] + viewproj[1][1];
        frustumplane[view_id][2][2] = viewproj[2][3] + viewproj[2][1];
        frustumplane[view_id][2][3] = viewproj[3][3] + viewproj[3][1];

        frustumplane[view_id][3][0] = viewproj[0][3] - viewproj[0][1];
        frustumplane[view_id][3][1] = viewproj[1][3] - viewproj[1][1];
        frustumplane[view_id][3][2] = viewproj[2][3] - viewproj[2][1];
        frustumplane[view_id][3][3] = viewproj[3][3] - viewproj[3][1];

        frustumplane[view_id][4][0] = viewproj[0][2];
        frustumplane[view_id][4][1] = viewproj[1][2];
        frustumplane[view_id][4][2] = viewproj[2][2];
        frustumplane[view_id][4][3] = viewproj[3][2];

        frustumplane[view_id][5][0] = viewproj[0][3] - viewproj[0][2];
        frustumplane[view_id][5][1] = viewproj[1][3] - viewproj[1][2];
        frustumplane[view_id][5][2] = viewproj[2][3] - viewproj[2][2];
        frustumplane[view_id][5][3] = viewproj[3][3] - viewproj[3][2];
        
    }
}

std::vector<at::Tensor> create_viewproj_forward(at::Tensor view_params, at::Tensor recp_tan_half_fov_x,int img_h,int img_w,float z_near,float z_far)
{
    int views_num = view_params.size(0);
    torch::Tensor view_matrix = torch::empty({ views_num,4,4 }, view_params.options());
    torch::Tensor proj_matrix = torch::empty({ views_num,4,4 }, view_params.options());
    torch::Tensor viewproj_matrix = torch::empty({ views_num,4,4 }, view_params.options());
    torch::Tensor frustumplane = torch::empty({ views_num,6,4 }, view_params.options().requires_grad(false));
    int blocks_num = std::ceil(views_num / 128.0f);
    create_viewproj_forward_kernel<<<views_num,128>>>(
        view_params.packed_accessor32< float, 2, torch::RestrictPtrTraits>(),
        recp_tan_half_fov_x.packed_accessor32< float, 1, torch::RestrictPtrTraits>(),
        img_h, img_w, z_near, z_far,
        view_matrix.packed_accessor32< float, 3, torch::RestrictPtrTraits>(),
        proj_matrix.packed_accessor32< float, 3, torch::RestrictPtrTraits>(),
        viewproj_matrix.packed_accessor32< float, 3, torch::RestrictPtrTraits>(),
        frustumplane.packed_accessor32< float, 3, torch::RestrictPtrTraits>()
    );
    return { view_matrix, proj_matrix, viewproj_matrix, frustumplane };
}



__global__ void compact_visible_params_kernel_forward(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunkid,    //[chunk_num] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[4,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,    //[1,3,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,    //[?,3,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_position,    //[4,p] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_scale,    //[3,p] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_rotation,    //[4,p] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> compacted_sh_base,    //[1,3,p] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> compacted_sh_rest,    //[?,3,p] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_opacity    //[1,p] 
)
{
    if (blockIdx.x < visible_chunkid.size(0))
    {
        int chunksize = position.size(2);
        int chunkid = visible_chunkid[blockIdx.x];
        for (int index = threadIdx.x; index < chunksize; index += blockDim.x)
        {
            //copy
            compacted_position[0][blockIdx.x][index] = position[0][chunkid][index];
            compacted_position[1][blockIdx.x][index] = position[1][chunkid][index];
            compacted_position[2][blockIdx.x][index] = position[2][chunkid][index];
            compacted_scale[0][blockIdx.x][index] = scale[0][chunkid][index];
            compacted_scale[1][blockIdx.x][index] = scale[1][chunkid][index];
            compacted_scale[2][blockIdx.x][index] = scale[2][chunkid][index];
            compacted_rotation[0][blockIdx.x][index] = rotation[0][chunkid][index];
            compacted_rotation[1][blockIdx.x][index] = rotation[1][chunkid][index];
            compacted_rotation[2][blockIdx.x][index] = rotation[2][chunkid][index];
            compacted_rotation[3][blockIdx.x][index] = rotation[3][chunkid][index];
            compacted_sh_base[0][0][blockIdx.x][index] = sh_base[0][0][chunkid][index];
            compacted_sh_base[0][1][blockIdx.x][index] = sh_base[0][1][chunkid][index];
            compacted_sh_base[0][2][blockIdx.x][index] = sh_base[0][2][chunkid][index];
            for (int i = 0; i < sh_rest.size(0); i++)
            {
                compacted_sh_rest[i][0][blockIdx.x][index] = sh_rest[i][0][chunkid][index];
                compacted_sh_rest[i][1][blockIdx.x][index] = sh_rest[i][1][chunkid][index];
                compacted_sh_rest[i][2][blockIdx.x][index] = sh_rest[i][2][chunkid][index];
            }
            compacted_opacity[0][blockIdx.x][index] = opacity[0][chunkid][index];
        }
    }
}


std::vector<at::Tensor> compact_visible_params_forward(at::Tensor visible_chunkid, at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity)
{
    int64_t visible_chunknum = visible_chunkid.size(0);
    int64_t chunksize = position.size(2);

    auto tensor_shape = position.sizes();
    at::Tensor compacted_position = torch::empty({ tensor_shape[0], visible_chunknum, chunksize }, position.options());
    tensor_shape = scale.sizes();
    at::Tensor compacted_scale = torch::empty({ tensor_shape[0], visible_chunknum, chunksize }, scale.options());
    tensor_shape = rotation.sizes();
    at::Tensor compacted_rotation = torch::empty({ tensor_shape[0], visible_chunknum, chunksize }, rotation.options());

    tensor_shape = sh_base.sizes();
    at::Tensor compacted_sh_base = torch::empty({ tensor_shape[0],tensor_shape[1], visible_chunknum, chunksize }, sh_base.options());
    tensor_shape = sh_rest.sizes();
    at::Tensor compacted_sh_rest = torch::empty({ tensor_shape[0],tensor_shape[1], visible_chunknum, chunksize }, sh_rest.options());

    tensor_shape = opacity.sizes();
    at::Tensor compacted_opacity = torch::empty({ tensor_shape[0], visible_chunknum, chunksize }, opacity.options());

    //dim3 Block3d(32, 1, 1);
    compact_visible_params_kernel_forward<<<visible_chunknum, 128 >>>(
        visible_chunkid.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        compacted_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        compacted_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
    );
    CUDA_CHECK_ERRORS;
    return { compacted_position,compacted_scale,compacted_rotation,compacted_sh_base,compacted_sh_rest,compacted_opacity };
}


__global__ void compact_visible_params_kernel_backward(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> reverse_map,     //[visible_chunk_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_position_grad,    //[4,P] 
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_scale_grad,    //[3,P] 
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_rotation_grad,    //[4,P] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_sh_base_grad,    //[1,3,P] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_sh_rest_grad,    //[?,3,P] 
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_opacity_grad,    //[1,P] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position_grad,    //[4,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale_grad,    //[3,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation_grad,    //[4,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base_grad,    //[1,3,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest_grad,    //[?,3,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity_grad    //[1,chunk_num,chunk_size] 
)
{
    //assert blockDim.x==chunksize

    if (blockIdx.x<reverse_map.size(0))
    {
        int chunk_index = reverse_map[blockIdx.x];
        int chunk_size = position_grad.size(2);
        int offset = blockIdx.x * chunk_size;

        //copy
        for (int index = threadIdx.x; index < chunk_size; index += blockDim.x)
        {
            position_grad[0][chunk_index][index] = compacted_position_grad[0][offset+index];
            position_grad[1][chunk_index][index] = compacted_position_grad[1][offset+index];
            position_grad[2][chunk_index][index] = compacted_position_grad[2][offset+index];

            scale_grad[0][chunk_index][index] = compacted_scale_grad[0][offset+index];
            scale_grad[1][chunk_index][index] = compacted_scale_grad[1][offset+index];
            scale_grad[2][chunk_index][index] = compacted_scale_grad[2][offset+index];

            rotation_grad[0][chunk_index][index] = compacted_rotation_grad[0][offset+index];
            rotation_grad[1][chunk_index][index] = compacted_rotation_grad[1][offset+index];
            rotation_grad[2][chunk_index][index] = compacted_rotation_grad[2][offset+index];
            rotation_grad[3][chunk_index][index] = compacted_rotation_grad[3][offset+index];

            sh_base_grad[0][0][chunk_index][index] = compacted_sh_base_grad[0][0][offset+index];
            sh_base_grad[0][1][chunk_index][index] = compacted_sh_base_grad[0][1][offset+index];
            sh_base_grad[0][2][chunk_index][index] = compacted_sh_base_grad[0][2][offset+index];

            for (int i = 0; i < compacted_sh_rest_grad.size(0); i++)
            {
                sh_rest_grad[i][0][chunk_index][index] = compacted_sh_rest_grad[i][0][offset+index];
                sh_rest_grad[i][1][chunk_index][index] = compacted_sh_rest_grad[i][1][offset+index];
                sh_rest_grad[i][2][chunk_index][index] = compacted_sh_rest_grad[i][2][offset+index];
            }

            opacity_grad[0][chunk_index][index] = compacted_opacity_grad[0][offset+index];
        }
    }
}

std::vector<at::Tensor> compact_visible_params_backward(int64_t chunk_num, int64_t chunk_size, at::Tensor reverse_map, 
    at::Tensor compacted_position_grad, at::Tensor compacted_scale_grad, at::Tensor compacted_rotation_grad, 
    at::Tensor compacted_sh_base_grad, at::Tensor compacted_sh_rest_grad, at::Tensor compacted_opacity_grad)
{

    auto tensor_shape = compacted_position_grad.sizes();
    at::Tensor position_grad = torch::zeros({ tensor_shape[0], chunk_num,chunk_size }, compacted_position_grad.options());
    
    tensor_shape = compacted_scale_grad.sizes();
    at::Tensor scale_grad = torch::zeros({  tensor_shape[0],chunk_num,chunk_size }, compacted_scale_grad.options());

    tensor_shape = compacted_rotation_grad.sizes();
    at::Tensor rotation_grad = torch::zeros({  tensor_shape[0],chunk_num,chunk_size }, compacted_rotation_grad.options());

    tensor_shape = compacted_sh_base_grad.sizes();
    at::Tensor sh_base_grad = torch::zeros({  tensor_shape[0],tensor_shape[1],chunk_num,chunk_size }, compacted_sh_base_grad.options());

    tensor_shape = compacted_sh_rest_grad.sizes();
    at::Tensor sh_rest_grad = torch::zeros({  tensor_shape[0],tensor_shape[1],chunk_num,chunk_size }, compacted_sh_rest_grad.options());

    tensor_shape = compacted_opacity_grad.sizes();
    at::Tensor opacity_grad = torch::zeros({  tensor_shape[0],chunk_num,chunk_size }, compacted_opacity_grad.options());

    int visible_chunknum = compacted_position_grad.size(1) / chunk_size;
    compact_visible_params_kernel_backward<<<visible_chunknum,256>>>(
        reverse_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        compacted_position_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_scale_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_rotation_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_sh_base_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_sh_rest_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_opacity_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
        );
    CUDA_CHECK_ERRORS;
    return { position_grad,scale_grad,rotation_grad,sh_base_grad,sh_rest_grad,opacity_grad };
}

__global__ void sparse_chunk_adam_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> param,     //
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad,    //
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> exp_avg,    //
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> exp_avg_sq,    //
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible,
    const float lr,const float b1,const float b2,const float eps
)
{
    
    //if (blockIdx.x < visible.size(0)&&blockIdx.y<param.size(0) && threadIdx.x < param.size(2))
    {
        int chunk_id = visible[blockIdx.x];
        //for (int i = 0; i < param.size(0); i++)
        {
            float Register_param_grad = grad[blockIdx.y][blockIdx.x][threadIdx.x];
            float Register_exp_avg = exp_avg[blockIdx.y][chunk_id][threadIdx.x];
            float Register_exp_avg_sq = exp_avg_sq[blockIdx.y][chunk_id][threadIdx.x];
            Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
            Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
            float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);
            param[blockIdx.y][chunk_id][threadIdx.x] += step;
            exp_avg[blockIdx.y][chunk_id][threadIdx.x] = Register_exp_avg;
            exp_avg_sq[blockIdx.y][chunk_id][threadIdx.x] = Register_exp_avg_sq;
        }
        //param[0][0][0] = -1;
    }
    
}


__global__ void sparse_primitive_adam_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> param,     //
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad,    //
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> exp_avg,    //
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> exp_avg_sq,    //
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible,
    const float lr, const float b1, const float b2, const float eps
)
{
    int primitive_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_id < visible.size(0)&& visible[primitive_id])
    {
        for (int i = 0; i < param.size(0); i++)
        {
            float Register_param_grad = grad[i][primitive_id];
            float Register_exp_avg = exp_avg[i][primitive_id];
            float Register_exp_avg_sq = exp_avg_sq[i][primitive_id];
            Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
            Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
            float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);
            param[i][primitive_id] += step;
            exp_avg[i][primitive_id] = Register_exp_avg;
            exp_avg_sq[i][primitive_id] = Register_exp_avg_sq;
        }
        //param[0][0][0] = -1;
    }

}

void adamUpdate(torch::Tensor &param,torch::Tensor &param_grad,torch::Tensor &exp_avg,torch::Tensor &exp_avg_sq,torch::Tensor &visible,
    const double lr,
	const double b1,
	const double b2,
	const double eps
)
{
    if (param.sizes().size() == 3)//chunk
    {
        dim3 Block3d(visible.size(0), param.size(0), 1);
        sparse_chunk_adam_kernel << <Block3d, param.size(2) >> > (
            param.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            param_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            exp_avg.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            exp_avg_sq.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            visible.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            lr, b1, b2, eps);
    }
    else if(param.sizes().size() == 2)
    {
        int primitive_num=visible.size(0);
        sparse_primitive_adam_kernel<<<int(std::ceil(primitive_num / 256.0f)),256>>> (
            param.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            param_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            exp_avg.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            exp_avg_sq.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            visible.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            lr, b1, b2, eps);
    }
    return;
}

__global__ void create_viewproj_backward_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix_grad,    //[viewsnum,4,4]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> proj_matrix_grad,    //[viewsnum,4,4]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> viewproj_matrix_grad,    //[viewsnum,4,4]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> view_params,    //[views_num,7]
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> recp_tan_half_fov_x,    //[1]
    int img_h, int img_w, float z_near, float z_far,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_view_params,    //[views_num,7]
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grad_recp_tan_half_fov_x    //[1]
)
{
    int view_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (view_id < view_params.size(0))
    {
        float r = view_params[view_id][0];
        float x = view_params[view_id][1];
        float y = view_params[view_id][2];
        float z = view_params[view_id][3];
        float recp = rsqrtf(r * r + x * x + y * y + z * z);
        r *= recp;
        x *= recp;
        y *= recp;
        z *= recp;
        
        // Compute view matrix elements (needed for viewproj gradient computation)
        float view[4][4] = {
            {1 - 2 * (y * y + z * z),2 * (x * y + r * z),2 * (x * z - r * y),0},
            {2 * (x * y - r * z),1 - 2 * (x * x + z * z),2 * (y * z + r * x),0},
            {2 * (x * z + r * y),2 * (y * z - r * x),1 - 2 * (x * x + y * y),0},
            {view_params[view_id][4],view_params[view_id][5],view_params[view_id][6],1.0f}
        };

        // Compute proj matrix elements
        float proj_00 = recp_tan_half_fov_x[0];
        float proj_11 = proj_00 * img_w / img_h;
        float proj[4][4] = {
            {proj_00,0,0,0},
            {0,proj_11,0,0},
            {0,0,z_far / (z_far - z_near),1},
            {0,0,-z_far * z_near / (z_far - z_near),0}
        };

        // Initialize local gradients for view_matrix and proj_matrix
        float local_view_grad[4][4] = {{0}};
        float local_proj_grad[4][4] = {{0}};
        
        // Chain rule: dL/d(view) = dL/d(viewproj) * d(viewproj)/d(view) = dL/d(viewproj) * proj^T
        // Chain rule: dL/d(proj) = dL/d(viewproj) * d(viewproj)/d(proj) = view^T * dL/d(viewproj)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float viewproj_grad = viewproj_matrix_grad[view_id][i][j];
                for (int k = 0; k < 4; k++) {
                    // d(viewproj[i][j])/d(view[i][k]) = proj[k][j]
                    local_view_grad[i][k] += viewproj_grad * proj[k][j];
                    
                    // d(viewproj[i][j])/d(proj[k][j]) = view[i][k]
                    local_proj_grad[k][j] += viewproj_grad * view[i][k];
                }
            }
        }
        
        // Accumulate local gradients with the passed-in gradients
        float accumulated_view_grad[4][4];
        float accumulated_proj_grad[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                accumulated_view_grad[i][j] = view_matrix_grad[view_id][i][j] + local_view_grad[i][j];
                accumulated_proj_grad[i][j] = proj_matrix_grad[view_id][i][j] + local_proj_grad[i][j];
            }
        }

        // Initialize gradients for quaternion parameters
        float grad_r = 0, grad_x = 0, grad_y = 0, grad_z = 0;
        
        // Direct gradient computation from view matrix elements (including chain rule contributions)
        // Row 0
        float grad = accumulated_view_grad[0][0]; // (1 - 2y² - 2z²)
        grad_y += grad * (-4*y);
        grad_z += grad * (-4*z);

        grad = accumulated_view_grad[0][1]; // 2(xy + rz)
        grad_x += grad * (2*y);
        grad_y += grad * (2*x);
        grad_r += grad * (2*z);
        grad_z += grad * (2*r);

        grad = accumulated_view_grad[0][2]; // 2(xz - ry)
        grad_x += grad * (2*z);
        grad_z += grad * (2*x);
        grad_r += grad * (-2*y);
        grad_y += grad * (-2*r);

        // Row 1
        grad = accumulated_view_grad[1][0]; // 2(xy - rz)
        grad_x += grad * (2*y);
        grad_y += grad * (2*x);
        grad_r += grad * (-2*z);
        grad_z += grad * (-2*r);

        grad = accumulated_view_grad[1][1]; // (1 - 2x² - 2z²)
        grad_x += grad * (-4*x);
        grad_z += grad * (-4*z);

        grad = accumulated_view_grad[1][2]; // 2(yz + rx)
        grad_y += grad * (2*z);
        grad_z += grad * (2*y);
        grad_r += grad * (2*x);
        grad_x += grad * (2*r);

        // Row 2
        grad = accumulated_view_grad[2][0]; // 2(xz + ry)
        grad_x += grad * (2*z);
        grad_z += grad * (2*x);
        grad_r += grad * (2*y);
        grad_y += grad * (2*r);

        grad = accumulated_view_grad[2][1]; // 2(yz - rx)
        grad_y += grad * (2*z);
        grad_z += grad * (2*y);
        grad_r += grad * (-2*x);
        grad_x += grad * (-2*r);

        grad = accumulated_view_grad[2][2]; // (1 - 2x² - 2y²)
        grad_x += grad * (-4*x);
        grad_y += grad * (-4*y);

        // Translation gradients (grad_view_params[:, 4:])
        grad_view_params[view_id][4] = accumulated_view_grad[3][0];
        grad_view_params[view_id][5] = accumulated_view_grad[3][1];
        grad_view_params[view_id][6] = accumulated_view_grad[3][2];

        // Compute recp_tan_half_fov_x gradient
        grad_recp_tan_half_fov_x[0] += accumulated_proj_grad[0][0]; // grad w.r.t. proj_00
        grad_recp_tan_half_fov_x[0] += accumulated_proj_grad[1][1] * (img_w / img_h); // grad w.r.t. proj_11

        // Apply quaternion normalization and unit constraint
        float norm = sqrtf(r*r + x*x + y*y + z*z);
        float dot = (r * grad_r + x * grad_x + y * grad_y + z * grad_z) / (norm * norm);
        
        grad_view_params[view_id][0] = grad_r / norm - r * dot;
        grad_view_params[view_id][1] = grad_x / norm - x * dot;
        grad_view_params[view_id][2] = grad_y / norm - y * dot;
        grad_view_params[view_id][3] = grad_z / norm - z * dot;
    }
}

std::vector<at::Tensor> create_viewproj_backward(
    at::Tensor view_matrix_grad,
    at::Tensor proj_matrix_grad, 
    at::Tensor viewproj_matrix_grad,
    at::Tensor view_params,
    at::Tensor recp_tan_half_fov_x,
    int img_h, int img_w,
    float z_near, float z_far)
{
    int views_num = view_params.size(0);
    torch::Tensor grad_view_params = torch::zeros_like(view_params);
    torch::Tensor grad_recp_tan_half_fov_x = torch::zeros_like(recp_tan_half_fov_x);

    int blocks_num = std::ceil(views_num / 128.0f);
    create_viewproj_backward_kernel<<<blocks_num, 128>>>(
        view_matrix_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        proj_matrix_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        viewproj_matrix_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        view_params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        recp_tan_half_fov_x.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        img_h, img_w, z_near, z_far,
        grad_view_params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_recp_tan_half_fov_x.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
    );
    CUDA_CHECK_ERRORS;
    // Return both gradients as a vector
    return {grad_view_params, grad_recp_tan_half_fov_x};
}

__global__ void frustum_culling_aabb_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> frustumplane,  // [N, 6, 4]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> aabb_origin,   // [3, M]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> aabb_ext,      // [3, M]
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> visibility,           // [M]
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_num,           // [M]
    torch::PackedTensorAccessor32<signed long long, 1, torch::RestrictPtrTraits> visible_chunkid           // [M]
) 
{
    __shared__ int visible_num_in_block;
    if (threadIdx.x == 0)
    {
        visible_num_in_block = 0;
    }
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    bool global_visible = false;
    

    if (m < aabb_origin.size(1))
    {
        // Check all 6 frustum planes
        for (int n = 0; n < frustumplane.size(0); n++)
        {
            bool is_visible = true;
            for (int plane_idx = 0; plane_idx < 6; plane_idx++) {
                // Get plane normal and distance
                float plane_normal_x = frustumplane[n][plane_idx][0];
                float plane_normal_y = frustumplane[n][plane_idx][1];
                float plane_normal_z = frustumplane[n][plane_idx][2];
                float plane_distance = frustumplane[n][plane_idx][3];

                // Get AABB origin and extent
                float origin_x = aabb_origin[0][m];
                float origin_y = aabb_origin[1][m];
                float origin_z = aabb_origin[2][m];

                float ext_x = aabb_ext[0][m];
                float ext_y = aabb_ext[1][m];
                float ext_z = aabb_ext[2][m];

                // Project origin to plane normal
                float dist_origin = plane_normal_x * origin_x + plane_normal_y * origin_y + plane_normal_z * origin_z + plane_distance;

                // Project extent to plane normal (using absolute values)
                float dist_ext = fabsf(plane_normal_x) * ext_x + fabsf(plane_normal_y) * ext_y + fabsf(plane_normal_z) * ext_z;

                // Push out the origin
                float pushed_origin_dist = dist_origin + dist_ext;

                // If completely outside any plane, it's not visible
                is_visible &= (pushed_origin_dist >= 0);
            }
            global_visible |= is_visible;
        }
    }
    visibility[m] = global_visible;
    __syncthreads();

    // reduce lane
    unsigned warp_mask = __ballot_sync(0xffffffff, global_visible);
    int lane_id = threadIdx.x & 0x1f;
    int lane_offset = __popc(warp_mask & ((1u << lane_id) - 1u));
    //reduce warp
    int warp_offset = 0;
    if (lane_id == 0)
    {
        warp_offset=atomicAdd(&visible_num_in_block, __popc(warp_mask));
    }
    warp_offset = __shfl_sync(0xffffffff, warp_offset, 0);
    __syncthreads();
    //reduce block
    int block_offset = 0;
    if (threadIdx.x == 0)
    {
        block_offset = atomicAdd(&visible_num[0], visible_num_in_block);
    }
    block_offset = __shfl_sync(0xffffffff, block_offset, 0);
    if (global_visible)
    {
        visible_chunkid[lane_offset + warp_offset + block_offset] = m;
    }
}

std::vector<at::Tensor> frustum_culling_aabb_cuda(at::Tensor aabb_origin,at::Tensor aabb_ext,at::Tensor frustumplane) 
{
    // Get dimensions
    int N = frustumplane.size(0);
    int M = aabb_origin.size(1);
    
    // Create output tensor
    torch::Tensor visibility = torch::empty({M}, torch::dtype(torch::kBool).device(frustumplane.device()));
    torch::Tensor visible_chunks_num = torch::zeros({ 1 }, torch::dtype(torch::kInt32).device(frustumplane.device()));
    torch::Tensor visible_chunkid = torch::empty({ M }, torch::dtype(torch::kInt64).device(frustumplane.device()));
    
    // Launch kernel
    frustum_culling_aabb_kernel<<<(M + 255) / 256, 256 >>>(
        frustumplane.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        aabb_origin.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        aabb_ext.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        visibility.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        visible_chunkid.packed_accessor32<signed long long, 1, torch::RestrictPtrTraits>()
    );
    
    // Check for errors
    CUDA_CHECK_ERRORS;


    
    // Return visibility tensor
    return {visibility,visible_chunks_num,visible_chunkid };
}
