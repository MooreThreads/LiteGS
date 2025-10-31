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
#include <ATen/cuda/CUDAContext.h>
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
        float recp = rsqrtf(r * r + x * x + y * y + z * z + 1e-12f);
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
        float recp = rsqrtf(r * r + x * x + y * y + z * z + 1e-12f);
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
        float local_view_grad[4][4] = { {0} };
        float local_proj_grad[4][4] = { {0} };

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
        grad_y += grad * (-4 * y);
        grad_z += grad * (-4 * z);

        grad = accumulated_view_grad[0][1]; // 2(xy + rz)
        grad_x += grad * (2 * y);
        grad_y += grad * (2 * x);
        grad_r += grad * (2 * z);
        grad_z += grad * (2 * r);

        grad = accumulated_view_grad[0][2]; // 2(xz - ry)
        grad_x += grad * (2 * z);
        grad_z += grad * (2 * x);
        grad_r += grad * (-2 * y);
        grad_y += grad * (-2 * r);

        // Row 1
        grad = accumulated_view_grad[1][0]; // 2(xy - rz)
        grad_x += grad * (2 * y);
        grad_y += grad * (2 * x);
        grad_r += grad * (-2 * z);
        grad_z += grad * (-2 * r);

        grad = accumulated_view_grad[1][1]; // (1 - 2x² - 2z²)
        grad_x += grad * (-4 * x);
        grad_z += grad * (-4 * z);

        grad = accumulated_view_grad[1][2]; // 2(yz + rx)
        grad_y += grad * (2 * z);
        grad_z += grad * (2 * y);
        grad_r += grad * (2 * x);
        grad_x += grad * (2 * r);

        // Row 2
        grad = accumulated_view_grad[2][0]; // 2(xz + ry)
        grad_x += grad * (2 * z);
        grad_z += grad * (2 * x);
        grad_r += grad * (2 * y);
        grad_y += grad * (2 * r);

        grad = accumulated_view_grad[2][1]; // 2(yz - rx)
        grad_y += grad * (2 * z);
        grad_z += grad * (2 * y);
        grad_r += grad * (-2 * x);
        grad_x += grad * (-2 * r);

        grad = accumulated_view_grad[2][2]; // (1 - 2x² - 2y²)
        grad_x += grad * (-4 * x);
        grad_y += grad * (-4 * y);

        // Translation gradients (grad_view_params[:, 4:])
        grad_view_params[view_id][4] = accumulated_view_grad[3][0];
        grad_view_params[view_id][5] = accumulated_view_grad[3][1];
        grad_view_params[view_id][6] = accumulated_view_grad[3][2];

        // Compute recp_tan_half_fov_x gradient
        grad_recp_tan_half_fov_x[0] += accumulated_proj_grad[0][0]; // grad w.r.t. proj_00
        grad_recp_tan_half_fov_x[0] += accumulated_proj_grad[1][1] * (img_w / img_h); // grad w.r.t. proj_11

        // Apply quaternion normalization and unit constraint
        float norm = sqrtf(r * r + x * x + y * y + z * z);
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
    create_viewproj_backward_kernel << <blocks_num, 128 >> > (
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
    return { grad_view_params, grad_recp_tan_half_fov_x };
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
        visible_num_in_block = atomicAdd(&visible_num[0], visible_num_in_block);
    }
    __syncthreads();
    block_offset = visible_num_in_block;
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

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

template <int degree>
__device__ void sh2rgb_forward_kernel(
    int view_id,int chunkid,int index,float3 dir,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> SH_base,    //[1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> SH_rest,    //[(deg + 1) ** 2-1,3,chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> rgb         //[batch,3,chunks_num,chunk_size]
)
{
    float3 result;
    result.x = SH_C0 * SH_base[0][0][chunkid][index];
    result.y = SH_C0 * SH_base[0][1][chunkid][index];
    result.z = SH_C0 * SH_base[0][2][chunkid][index];
    if (degree > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result.x = result.x - SH_C1 * y * SH_rest[0][0][chunkid][index] + SH_C1 * z * SH_rest[1][0][chunkid][index] - SH_C1 * x * SH_rest[2][0][chunkid][index];
        result.y = result.y - SH_C1 * y * SH_rest[0][1][chunkid][index] + SH_C1 * z * SH_rest[1][1][chunkid][index] - SH_C1 * x * SH_rest[2][1][chunkid][index];
        result.z = result.z - SH_C1 * y * SH_rest[0][2][chunkid][index] + SH_C1 * z * SH_rest[1][2][chunkid][index] - SH_C1 * x * SH_rest[2][2][chunkid][index];

        if (degree > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result.x = result.x +
                SH_C2[0] * xy * SH_rest[3][0][chunkid][index] +
                SH_C2[1] * yz * SH_rest[4][0][chunkid][index] +
                SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][0][chunkid][index] +
                SH_C2[3] * xz * SH_rest[6][0][chunkid][index] +
                SH_C2[4] * (xx - yy) * SH_rest[7][0][chunkid][index];
            result.y = result.y +
                SH_C2[0] * xy * SH_rest[3][1][chunkid][index] +
                SH_C2[1] * yz * SH_rest[4][1][chunkid][index] +
                SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][1][chunkid][index] +
                SH_C2[3] * xz * SH_rest[6][1][chunkid][index] +
                SH_C2[4] * (xx - yy) * SH_rest[7][1][chunkid][index];
            result.z = result.z +
                SH_C2[0] * xy * SH_rest[3][2][chunkid][index] +
                SH_C2[1] * yz * SH_rest[4][2][chunkid][index] +
                SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][2][chunkid][index] +
                SH_C2[3] * xz * SH_rest[6][2][chunkid][index] +
                SH_C2[4] * (xx - yy) * SH_rest[7][2][chunkid][index];

            if (degree > 2)
            {
                result.x = result.x +
                    SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][0][chunkid][index] +
                    SH_C3[1] * xy * z * SH_rest[9][0][chunkid][index] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][0][chunkid][index] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][0][chunkid][index] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][0][chunkid][index] +
                    SH_C3[5] * z * (xx - yy) * SH_rest[13][0][chunkid][index] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][0][chunkid][index];
                result.y = result.y +
                    SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][1][chunkid][index] +
                    SH_C3[1] * xy * z * SH_rest[9][1][chunkid][index] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][1][chunkid][index] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][1][chunkid][index] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][1][chunkid][index] +
                    SH_C3[5] * z * (xx - yy) * SH_rest[13][1][chunkid][index] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][1][chunkid][index];
                result.z = result.z +
                    SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][2][chunkid][index] +
                    SH_C3[1] * xy * z * SH_rest[9][2][chunkid][index] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][2][chunkid][index] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][2][chunkid][index] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][2][chunkid][index] +
                    SH_C3[5] * z * (xx - yy) * SH_rest[13][2][chunkid][index] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][2][chunkid][index];
            }
        }

    }
    result.x += 0.5f;
    result.y += 0.5f;
    result.z += 0.5f;
    rgb[view_id][0][chunkid][index] = result.x;
    rgb[view_id][1][chunkid][index] = result.y;
    rgb[view_id][2][chunkid][index] = result.z;
}

template <int degree,bool bInit>
__device__ void sh2rgb_backward_kernel(
    int chunkid,int source_chunkid, int index, float3 dir,float3 dL_dRGB,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> SH_base,    //[1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> SH_rest,    //[(deg + 1) ** 2-1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> rgb_grad,         //[batch,3,visible_chunks_num,chunk_size]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> SH_base_grad,   //[1,3,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> SH_rest_grad   //[(deg + 1) ** 2-1,3,visible_chunks_num,chunk_size] 
)
{

    float dRGBdsh0 = SH_C0;
    if (bInit)
    {
        SH_base_grad[0][0][chunkid][index] = dRGBdsh0 * dL_dRGB.x;
        SH_base_grad[0][1][chunkid][index] = dRGBdsh0 * dL_dRGB.y;
        SH_base_grad[0][2][chunkid][index] = dRGBdsh0 * dL_dRGB.z;
    }
    else
    {
        SH_base_grad[0][0][chunkid][index] += dRGBdsh0 * dL_dRGB.x;
        SH_base_grad[0][1][chunkid][index] += dRGBdsh0 * dL_dRGB.y;
        SH_base_grad[0][2][chunkid][index] += dRGBdsh0 * dL_dRGB.z;
    }
    
    if (degree > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;

        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        if (bInit)
        {
            SH_rest_grad[0][0][chunkid][index] = dRGBdsh1 * dL_dRGB.x;
            SH_rest_grad[1][0][chunkid][index] = dRGBdsh2 * dL_dRGB.x;
            SH_rest_grad[2][0][chunkid][index] = dRGBdsh3 * dL_dRGB.x;
            SH_rest_grad[0][1][chunkid][index] = dRGBdsh1 * dL_dRGB.y;
            SH_rest_grad[1][1][chunkid][index] = dRGBdsh2 * dL_dRGB.y;
            SH_rest_grad[2][1][chunkid][index] = dRGBdsh3 * dL_dRGB.y;
            SH_rest_grad[0][2][chunkid][index] = dRGBdsh1 * dL_dRGB.z;
            SH_rest_grad[1][2][chunkid][index] = dRGBdsh2 * dL_dRGB.z;
            SH_rest_grad[2][2][chunkid][index] = dRGBdsh3 * dL_dRGB.z;
        }
        else
        {
            SH_rest_grad[0][0][chunkid][index] += dRGBdsh1 * dL_dRGB.x;
            SH_rest_grad[1][0][chunkid][index] += dRGBdsh2 * dL_dRGB.x;
            SH_rest_grad[2][0][chunkid][index] += dRGBdsh3 * dL_dRGB.x;
            SH_rest_grad[0][1][chunkid][index] += dRGBdsh1 * dL_dRGB.y;
            SH_rest_grad[1][1][chunkid][index] += dRGBdsh2 * dL_dRGB.y;
            SH_rest_grad[2][1][chunkid][index] += dRGBdsh3 * dL_dRGB.y;
            SH_rest_grad[0][2][chunkid][index] += dRGBdsh1 * dL_dRGB.z;
            SH_rest_grad[1][2][chunkid][index] += dRGBdsh2 * dL_dRGB.z;
            SH_rest_grad[2][2][chunkid][index] += dRGBdsh3 * dL_dRGB.z;
        }

        if (degree > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            if (bInit)
            {
                SH_rest_grad[3][0][chunkid][index] = dRGBdsh4 * dL_dRGB.x;
                SH_rest_grad[4][0][chunkid][index] = dRGBdsh5 * dL_dRGB.x;
                SH_rest_grad[5][0][chunkid][index] = dRGBdsh6 * dL_dRGB.x;
                SH_rest_grad[6][0][chunkid][index] = dRGBdsh7 * dL_dRGB.x;
                SH_rest_grad[7][0][chunkid][index] = dRGBdsh8 * dL_dRGB.x;
                SH_rest_grad[3][1][chunkid][index] = dRGBdsh4 * dL_dRGB.y;
                SH_rest_grad[4][1][chunkid][index] = dRGBdsh5 * dL_dRGB.y;
                SH_rest_grad[5][1][chunkid][index] = dRGBdsh6 * dL_dRGB.y;
                SH_rest_grad[6][1][chunkid][index] = dRGBdsh7 * dL_dRGB.y;
                SH_rest_grad[7][1][chunkid][index] = dRGBdsh8 * dL_dRGB.y;
                SH_rest_grad[3][2][chunkid][index] = dRGBdsh4 * dL_dRGB.z;
                SH_rest_grad[4][2][chunkid][index] = dRGBdsh5 * dL_dRGB.z;
                SH_rest_grad[5][2][chunkid][index] = dRGBdsh6 * dL_dRGB.z;
                SH_rest_grad[6][2][chunkid][index] = dRGBdsh7 * dL_dRGB.z;
                SH_rest_grad[7][2][chunkid][index] = dRGBdsh8 * dL_dRGB.z;
            }
            else
            {
                SH_rest_grad[3][0][chunkid][index] += dRGBdsh4 * dL_dRGB.x;
                SH_rest_grad[4][0][chunkid][index] += dRGBdsh5 * dL_dRGB.x;
                SH_rest_grad[5][0][chunkid][index] += dRGBdsh6 * dL_dRGB.x;
                SH_rest_grad[6][0][chunkid][index] += dRGBdsh7 * dL_dRGB.x;
                SH_rest_grad[7][0][chunkid][index] += dRGBdsh8 * dL_dRGB.x;
                SH_rest_grad[3][1][chunkid][index] += dRGBdsh4 * dL_dRGB.y;
                SH_rest_grad[4][1][chunkid][index] += dRGBdsh5 * dL_dRGB.y;
                SH_rest_grad[5][1][chunkid][index] += dRGBdsh6 * dL_dRGB.y;
                SH_rest_grad[6][1][chunkid][index] += dRGBdsh7 * dL_dRGB.y;
                SH_rest_grad[7][1][chunkid][index] += dRGBdsh8 * dL_dRGB.y;
                SH_rest_grad[3][2][chunkid][index] += dRGBdsh4 * dL_dRGB.z;
                SH_rest_grad[4][2][chunkid][index] += dRGBdsh5 * dL_dRGB.z;
                SH_rest_grad[5][2][chunkid][index] += dRGBdsh6 * dL_dRGB.z;
                SH_rest_grad[6][2][chunkid][index] += dRGBdsh7 * dL_dRGB.z;
                SH_rest_grad[7][2][chunkid][index] += dRGBdsh8 * dL_dRGB.z;
            }

            if (degree > 2)
            {
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                if (bInit)
                {
                    SH_rest_grad[8][0][chunkid][index] = dRGBdsh9 * dL_dRGB.x;
                    SH_rest_grad[9][0][chunkid][index] = dRGBdsh10 * dL_dRGB.x;
                    SH_rest_grad[10][0][chunkid][index] = dRGBdsh11 * dL_dRGB.x;
                    SH_rest_grad[11][0][chunkid][index] = dRGBdsh12 * dL_dRGB.x;
                    SH_rest_grad[12][0][chunkid][index] = dRGBdsh13 * dL_dRGB.x;
                    SH_rest_grad[13][0][chunkid][index] = dRGBdsh14 * dL_dRGB.x;
                    SH_rest_grad[14][0][chunkid][index] = dRGBdsh15 * dL_dRGB.x;
                    SH_rest_grad[8][1][chunkid][index] = dRGBdsh9 * dL_dRGB.y;
                    SH_rest_grad[9][1][chunkid][index] = dRGBdsh10 * dL_dRGB.y;
                    SH_rest_grad[10][1][chunkid][index] = dRGBdsh11 * dL_dRGB.y;
                    SH_rest_grad[11][1][chunkid][index] = dRGBdsh12 * dL_dRGB.y;
                    SH_rest_grad[12][1][chunkid][index] = dRGBdsh13 * dL_dRGB.y;
                    SH_rest_grad[13][1][chunkid][index] = dRGBdsh14 * dL_dRGB.y;
                    SH_rest_grad[14][1][chunkid][index] = dRGBdsh15 * dL_dRGB.y;
                    SH_rest_grad[8][2][chunkid][index] = dRGBdsh9 * dL_dRGB.z;
                    SH_rest_grad[9][2][chunkid][index] = dRGBdsh10 * dL_dRGB.z;
                    SH_rest_grad[10][2][chunkid][index] = dRGBdsh11 * dL_dRGB.z;
                    SH_rest_grad[11][2][chunkid][index] = dRGBdsh12 * dL_dRGB.z;
                    SH_rest_grad[12][2][chunkid][index] = dRGBdsh13 * dL_dRGB.z;
                    SH_rest_grad[13][2][chunkid][index] = dRGBdsh14 * dL_dRGB.z;
                    SH_rest_grad[14][2][chunkid][index] = dRGBdsh15 * dL_dRGB.z;
                }
                else
                {
                    SH_rest_grad[8][0][chunkid][index] += dRGBdsh9 * dL_dRGB.x;
                    SH_rest_grad[9][0][chunkid][index] += dRGBdsh10 * dL_dRGB.x;
                    SH_rest_grad[10][0][chunkid][index] += dRGBdsh11 * dL_dRGB.x;
                    SH_rest_grad[11][0][chunkid][index] += dRGBdsh12 * dL_dRGB.x;
                    SH_rest_grad[12][0][chunkid][index] += dRGBdsh13 * dL_dRGB.x;
                    SH_rest_grad[13][0][chunkid][index] += dRGBdsh14 * dL_dRGB.x;
                    SH_rest_grad[14][0][chunkid][index] += dRGBdsh15 * dL_dRGB.x;
                    SH_rest_grad[8][1][chunkid][index] += dRGBdsh9 * dL_dRGB.y;
                    SH_rest_grad[9][1][chunkid][index] += dRGBdsh10 * dL_dRGB.y;
                    SH_rest_grad[10][1][chunkid][index] += dRGBdsh11 * dL_dRGB.y;
                    SH_rest_grad[11][1][chunkid][index] += dRGBdsh12 * dL_dRGB.y;
                    SH_rest_grad[12][1][chunkid][index] += dRGBdsh13 * dL_dRGB.y;
                    SH_rest_grad[13][1][chunkid][index] += dRGBdsh14 * dL_dRGB.y;
                    SH_rest_grad[14][1][chunkid][index] += dRGBdsh15 * dL_dRGB.y;
                    SH_rest_grad[8][2][chunkid][index] += dRGBdsh9 * dL_dRGB.z;
                    SH_rest_grad[9][2][chunkid][index] += dRGBdsh10 * dL_dRGB.z;
                    SH_rest_grad[10][2][chunkid][index] += dRGBdsh11 * dL_dRGB.z;
                    SH_rest_grad[11][2][chunkid][index] += dRGBdsh12 * dL_dRGB.z;
                    SH_rest_grad[12][2][chunkid][index] += dRGBdsh13 * dL_dRGB.z;
                    SH_rest_grad[13][2][chunkid][index] += dRGBdsh14 * dL_dRGB.z;
                    SH_rest_grad[14][2][chunkid][index] += dRGBdsh15 * dL_dRGB.z;
                }
            }
        }

    }
    
}

template <int degree>
__global__ void activate_forward_kernel(
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> visible_mask,    //[chunks_num] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,    //[views_num,4,4] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,    //[1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,    //[?,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunks_num,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_position,    //[4,chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_scale,    //[3,chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_rotation,    //[4,chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> color,    //[views_num,3,chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_opacity    //[1,chunks_num,chunk_size]
)
{
    int index = threadIdx.x;
    int chunkid = blockIdx.x;
    bool visible = visible_mask[chunkid];
    if (visible)
    {
        float3 pos{ position[0][chunkid][index],position[1][chunkid][index],position[2][chunkid][index] };
        activated_position[0][chunkid][index] = pos.x;
        activated_position[1][chunkid][index] = pos.y;
        activated_position[2][chunkid][index] = pos.z;
        activated_position[3][chunkid][index] = 1.0f;

        activated_scale[0][chunkid][index] = __expf(scale[0][chunkid][index]);
        activated_scale[1][chunkid][index] = __expf(scale[1][chunkid][index]);
        activated_scale[2][chunkid][index] = __expf(scale[2][chunkid][index]);

        float w = rotation[0][chunkid][index];
        float x = rotation[1][chunkid][index];
        float y = rotation[2][chunkid][index];
        float z = rotation[3][chunkid][index];

        float recp_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
        activated_rotation[0][chunkid][index] = w * recp_norm;
        activated_rotation[1][chunkid][index] = x * recp_norm;
        activated_rotation[2][chunkid][index] = y * recp_norm;
        activated_rotation[3][chunkid][index] = z * recp_norm;

        activated_opacity[0][chunkid][index]= 1.0f / (1.0f + __expf(-opacity[0][chunkid][index]));

        //sh
        for (int view_id = 0; view_id < view_matrix.size(0); view_id++)
        {
            //-t @ rot.trans()
            float3 inv_trans{ -view_matrix[view_id][3][0],-view_matrix[view_id][3][1],-view_matrix[view_id][3][2] };
            float3 camera_center;
            camera_center.x = inv_trans.x * view_matrix[view_id][0][0] + inv_trans.y * view_matrix[view_id][0][1] + inv_trans.z * view_matrix[view_id][0][2];
            camera_center.y = inv_trans.x * view_matrix[view_id][1][0] + inv_trans.y * view_matrix[view_id][1][1] + inv_trans.z * view_matrix[view_id][1][2];
            camera_center.z = inv_trans.x * view_matrix[view_id][2][0] + inv_trans.y * view_matrix[view_id][2][1] + inv_trans.z * view_matrix[view_id][2][2];
            float3 dir{ pos.x - camera_center.x,pos.y - camera_center.y,pos.z - camera_center.z };
            float norm_recp = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-12f);
            dir.x *= norm_recp;
            dir.y *= norm_recp;
            dir.z *= norm_recp;
            sh2rgb_forward_kernel<degree>(view_id,chunkid, index, dir, sh_base, sh_rest, color);
        }
    }

}

template <int degree>
__global__ void activate_backward_kernel(
    const torch::PackedTensorAccessor32<signed long long, 1, torch::RestrictPtrTraits> visible_chunkid,    //[visible_chunks_num] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,    //[views_num,4,4] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,    //[1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,    //[?,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_position_grad,    //[4,visible_chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_scale_grad,    //[3,visible_chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_rotation_grad,    //[4,visible_chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> color_grad,    //[views_num,3,visible_chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_opacity_grad,    //[1,visible_chunks_num,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position_grad,    //[3,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale_grad,    //[3,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation_grad,    //[4,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base_grad,    //[1,3,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest_grad,    //[?,3,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity_grad    //[1,visible_chunks_num,chunk_size]
)
{
    int index = threadIdx.x;
    int chunkid = blockIdx.x;
    int source_chunkid = visible_chunkid[chunkid];

    position_grad[0][chunkid][index] = activated_position_grad[0][chunkid][index];
    position_grad[1][chunkid][index] = activated_position_grad[1][chunkid][index];
    position_grad[2][chunkid][index] = activated_position_grad[2][chunkid][index];

    scale_grad[0][chunkid][index] = __expf(scale[0][source_chunkid][index]) * activated_scale_grad[0][chunkid][index];
    scale_grad[1][chunkid][index] = __expf(scale[1][source_chunkid][index]) * activated_scale_grad[1][chunkid][index];
    scale_grad[2][chunkid][index] = __expf(scale[2][source_chunkid][index]) * activated_scale_grad[2][chunkid][index];

    float w = rotation[0][source_chunkid][index];
    float x = rotation[1][source_chunkid][index];
    float y = rotation[2][source_chunkid][index];
    float z = rotation[3][source_chunkid][index];

    float recp_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
    rotation_grad[0][chunkid][index] = activated_rotation_grad[0][chunkid][index] * recp_norm;
    rotation_grad[1][chunkid][index] = activated_rotation_grad[1][chunkid][index] * recp_norm;
    rotation_grad[2][chunkid][index] = activated_rotation_grad[2][chunkid][index] * recp_norm;
    rotation_grad[3][chunkid][index] = activated_rotation_grad[3][chunkid][index] * recp_norm;

    opacity_grad[0][chunkid][index] = activated_opacity_grad[0][chunkid][index] * (1.0f - 1.0f / (1.0f + __expf(opacity[0][source_chunkid][index])));

    //sh
    for (int view_id = 0; view_id < view_matrix.size(0); view_id++)
    {
        //-t @ rot.trans()
        float3 inv_trans{ -view_matrix[view_id][3][0],-view_matrix[view_id][3][1],-view_matrix[view_id][3][2] };
        float3 camera_center;
        camera_center.x = inv_trans.x * view_matrix[view_id][0][0] + inv_trans.y * view_matrix[view_id][0][1] + inv_trans.z * view_matrix[view_id][0][2];
        camera_center.y = inv_trans.x * view_matrix[view_id][1][0] + inv_trans.y * view_matrix[view_id][1][1] + inv_trans.z * view_matrix[view_id][1][2];
        camera_center.z = inv_trans.x * view_matrix[view_id][2][0] + inv_trans.y * view_matrix[view_id][2][1] + inv_trans.z * view_matrix[view_id][2][2];
        float3 dir{ position[0][source_chunkid][index] - camera_center.x,position[1][source_chunkid][index] - camera_center.y,position[2][source_chunkid][index] - camera_center.z};
        float norm_recp = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-12f);
        dir.x *= norm_recp;
        dir.y *= norm_recp;
        dir.z *= norm_recp;
        float3 dL_dRGB{ color_grad[view_id][0][chunkid][index], color_grad[view_id][1][chunkid][index], color_grad[view_id][2][chunkid][index] };
        if (view_id == 0)
        {
            sh2rgb_backward_kernel<degree, true>(chunkid, source_chunkid, index, dir, dL_dRGB, sh_base, sh_rest, color_grad, sh_base_grad, sh_rest_grad);
        }
        else
        {
            sh2rgb_backward_kernel<degree, false>(chunkid, source_chunkid, index, dir, dL_dRGB, sh_base, sh_rest, color_grad, sh_base_grad, sh_rest_grad);
        }
    }
}

__global__ void compact_visible_params_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunkid,    //[chunks_num] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[4,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> color,    //[1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_position,    //[4,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_scale,    //[3,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_rotation,    //[4,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> compacted_color,    //[1,3,visible_chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_opacity    //[1,visible_chunks_num,chunk_size] 
)
{
    int index = threadIdx.x;
    int chunkid = visible_chunkid[blockIdx.x];
    //copy
    compacted_position[0][blockIdx.x][index] = position[0][chunkid][index];
    compacted_position[1][blockIdx.x][index] = position[1][chunkid][index];
    compacted_position[2][blockIdx.x][index] = position[2][chunkid][index];
    compacted_position[3][blockIdx.x][index] = position[3][chunkid][index];
    compacted_scale[0][blockIdx.x][index] = scale[0][chunkid][index];
    compacted_scale[1][blockIdx.x][index] = scale[1][chunkid][index];
    compacted_scale[2][blockIdx.x][index] = scale[2][chunkid][index];
    compacted_rotation[0][blockIdx.x][index] = rotation[0][chunkid][index];
    compacted_rotation[1][blockIdx.x][index] = rotation[1][chunkid][index];
    compacted_rotation[2][blockIdx.x][index] = rotation[2][chunkid][index];
    compacted_rotation[3][blockIdx.x][index] = rotation[3][chunkid][index];
    compacted_color[0][0][blockIdx.x][index] = color[0][0][chunkid][index];
    compacted_color[0][1][blockIdx.x][index] = color[0][1][chunkid][index];
    compacted_color[0][2][blockIdx.x][index] = color[0][2][chunkid][index];
    compacted_opacity[0][blockIdx.x][index] = opacity[0][chunkid][index];
}

std::vector<at::Tensor> cull_compact_activate(at::Tensor aabb_origin, at::Tensor aabb_ext, at::Tensor frustumplane, at::Tensor view_matrix,int sh_degree,
    at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity)
{

    // Create Stream to cover MemcpyDevice2Host latency
    cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t cpy_event;
    cudaEventCreate(&cpy_event);

    // Get dimensions
    int views_num = frustumplane.size(0);
    int chunks_num = aabb_origin.size(1);

    // Create output tensor
    torch::Tensor visibility = torch::empty({ chunks_num }, torch::dtype(torch::kBool).device(frustumplane.device()));
    torch::Tensor visible_chunks_num = torch::zeros({ 1 }, torch::dtype(torch::kInt32).device(frustumplane.device()));
    torch::Tensor visible_chunkid = torch::empty({ chunks_num }, torch::dtype(torch::kInt64).device(frustumplane.device()));

    // Launch kernel
    frustum_culling_aabb_kernel << <(chunks_num + 255) / 256, 256,0, torch_stream >> > (
        frustumplane.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        aabb_origin.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        aabb_ext.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        visibility.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        visible_chunkid.packed_accessor32<signed long long, 1, torch::RestrictPtrTraits>()
        );
    CUDA_CHECK_ERRORS;

    //visible_chunks_num to host
    int* device_data=visible_chunks_num.data_ptr<int>();
    int visible_chunks_num_host = 0;
    cudaMemcpyAsync(&visible_chunks_num_host, device_data, sizeof(int), cudaMemcpyDeviceToHost, torch_stream);
    cudaEventRecord(cpy_event, torch_stream);


    //activate
    int chunksize = position.size(2);
    auto tensor_shape = position.sizes();
    at::Tensor actived_position = torch::empty({ 4, chunks_num, chunksize }, position.options());
    tensor_shape = scale.sizes();
    at::Tensor actived_scale = torch::empty({ tensor_shape[0], chunks_num, chunksize }, scale.options());
    tensor_shape = rotation.sizes();
    at::Tensor actived_rotation = torch::empty({ tensor_shape[0], chunks_num, chunksize }, rotation.options());
    tensor_shape = sh_base.sizes();
    at::Tensor color = torch::empty({ 1,3, chunks_num, chunksize }, sh_base.options());
    tensor_shape = opacity.sizes();
    at::Tensor actived_opacity = torch::empty({ tensor_shape[0], chunks_num, chunksize }, opacity.options());

    //todo sh_degree
    switch (sh_degree)
    {
    case 0:
        activate_forward_kernel<0> << <chunks_num, chunksize,0, torch_stream >> > (
            visibility.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 1:
        activate_forward_kernel<1> << <chunks_num, chunksize, 0, torch_stream >> > (
            visibility.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 2:
        activate_forward_kernel<2> << <chunks_num, chunksize, 0, torch_stream >> > (
            visibility.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 3:
        activate_forward_kernel<3> << <chunks_num, chunksize, 0, torch_stream >> > (
            visibility.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            actived_rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    default:
        assert(false);
    }

    //compact
    cudaEventSynchronize(cpy_event);
    cudaEventDestroy(cpy_event);
    visible_chunkid=visible_chunkid.slice(0, 0, visible_chunks_num_host);
    tensor_shape = actived_position.sizes();
    at::Tensor compacted_position = torch::empty({ tensor_shape[0], visible_chunks_num_host, chunksize }, position.options());
    tensor_shape = actived_scale.sizes();
    at::Tensor compacted_scale = torch::empty({ tensor_shape[0], visible_chunks_num_host, chunksize }, scale.options());
    tensor_shape = actived_rotation.sizes();
    at::Tensor compacted_rotation = torch::empty({ tensor_shape[0], visible_chunks_num_host, chunksize }, rotation.options());
    tensor_shape = color.sizes();
    at::Tensor compacted_color = torch::empty({ tensor_shape[0],tensor_shape[1], visible_chunks_num_host, chunksize }, sh_base.options());
    tensor_shape = actived_opacity.sizes();
    at::Tensor compacted_opacity = torch::empty({ tensor_shape[0], visible_chunks_num_host, chunksize }, opacity.options());

    //dim3 Block3d(32, 1, 1);
    compact_visible_params_kernel << <visible_chunks_num_host, chunksize, 0, torch_stream >> > (
        visible_chunkid.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        actived_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        actived_scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        actived_rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        color.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_color.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        compacted_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
        );
    CUDA_CHECK_ERRORS;
    return { visible_chunkid, compacted_position,compacted_scale,compacted_rotation,compacted_color,compacted_opacity };
}

std::vector<at::Tensor> activate_backward(at::Tensor visible_chunkid, at::Tensor view_matrix, int sh_degree,
    at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity,
    at::Tensor activated_position_grad, at::Tensor activated_scale_grad, at::Tensor activated_rotation_grad, at::Tensor color_grad, at::Tensor activated_opacity_grad)
{
    int visible_chunks_num = visible_chunkid.size(0);
    int chunksize = position.size(2);

    auto tensor_shape = position.sizes();
    at::Tensor compacted_position_grad = torch::empty({ tensor_shape[0], visible_chunks_num, chunksize }, position.options());
    tensor_shape = scale.sizes();
    at::Tensor compacted_scale_grad = torch::empty({ tensor_shape[0], visible_chunks_num, chunksize }, scale.options());
    tensor_shape = rotation.sizes();
    at::Tensor compacted_rotation_grad = torch::empty({ tensor_shape[0], visible_chunks_num, chunksize }, rotation.options());
    tensor_shape = sh_base.sizes();
    at::Tensor compacted_sh_base_grad = torch::empty({ tensor_shape[0],tensor_shape[1], visible_chunks_num, chunksize }, sh_base.options());
    tensor_shape = sh_rest.sizes();
    at::Tensor compacted_sh_rest_grad = torch::zeros({ tensor_shape[0],tensor_shape[1], visible_chunks_num, chunksize }, sh_rest.options());
    tensor_shape = opacity.sizes();
    at::Tensor compacted_opacity_grad = torch::empty({ tensor_shape[0], visible_chunks_num, chunksize }, opacity.options());

    switch (sh_degree)
    {
    case 0:
        activate_backward_kernel<0> << <visible_chunks_num, chunksize >> > (
            visible_chunkid.packed_accessor32<signed long long, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            activated_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            compacted_sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            compacted_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 1:
        activate_backward_kernel<1> << <visible_chunks_num, chunksize >> > (
            visible_chunkid.packed_accessor32<signed long long, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            activated_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            compacted_sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            compacted_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 2:
        activate_backward_kernel<2> << <visible_chunks_num, chunksize >> > (
            visible_chunkid.packed_accessor32<signed long long, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            activated_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            compacted_sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            compacted_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 3:
        activate_backward_kernel<3> << <visible_chunks_num, chunksize >> > (
            visible_chunkid.packed_accessor32<signed long long, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            activated_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            activated_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            compacted_sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            compacted_sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            compacted_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    default:
        assert(false);
    }

    

    return { compacted_position_grad ,compacted_scale_grad ,compacted_rotation_grad ,compacted_sh_base_grad ,compacted_sh_rest_grad ,compacted_opacity_grad };
}