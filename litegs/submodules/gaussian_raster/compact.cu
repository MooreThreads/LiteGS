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
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> valid_index,
    int* valid_legnth,
    const float lr,const float b1,const float b2,const float eps
)
{
    
    if (valid_legnth == nullptr || blockIdx.x < *valid_legnth)
    {
        int chunk_id = valid_index[blockIdx.x];
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
    
}


__global__ void sparse_primitive_adam_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> param,     //
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad,    //
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> exp_avg,    //
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> exp_avg_sq,    //
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> valid_index,
    const float lr, const float b1, const float b2, const float eps
)
{
    int primitive_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_id < valid_index.size(0)&& valid_index[primitive_id])
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

void adamUpdate(
    torch::Tensor param, torch::Tensor param_grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor valid_index, std::optional<torch::Tensor> valid_length,
    const double lr, const double b1, const double b2, const double eps
)
{
    if (param.sizes().size() == 3)//chunk
    {
        dim3 Block3d(valid_index.size(0), param.size(0), 1);
        int* valid_length_ptr = nullptr;
        if (valid_length.has_value())
        {
            valid_length_ptr = (*valid_length).data_ptr<int>();
        }
        sparse_chunk_adam_kernel << <Block3d, param.size(2) >> > (
            param.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            param_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            exp_avg.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            exp_avg_sq.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            valid_index.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            valid_length_ptr,
            lr, b1, b2, eps);
    }
    else if(param.sizes().size() == 2)
    {
        int primitive_num= valid_index.size(0);
        sparse_primitive_adam_kernel<<<int(std::ceil(primitive_num / 256.0f)),256>>> (
            param.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            param_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            exp_avg.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            exp_avg_sq.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            valid_index.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            lr, b1, b2, eps);
    }
    else
    {
        assert(false);
    }
    return;
}

__global__ void frustum_culling_aabb_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> frustumplane,  // [N, 6, 4]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> aabb_origin,   // [3, M]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> aabb_ext,      // [3, M]
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> visibility,           // [M]
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_num,           // [M]
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunk_id           // [M]
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
        visible_chunk_id[lane_offset + warp_offset + block_offset] = m;
    }
}

std::vector<at::Tensor> frustum_culling_aabb(at::Tensor aabb_origin, at::Tensor aabb_ext, at::Tensor frustumplane, std::optional<at::Tensor> feedback_buffer_arg, std::optional<at::Tensor> data_idx_arg)
{
    // Get dimensions
    int N = frustumplane.size(0);
    int M = aabb_origin.size(1);
    
    // Create output tensor
    torch::Tensor visibility = torch::empty({M}, torch::dtype(torch::kBool).device(frustumplane.device()));
    torch::Tensor visible_chunks_num = torch::zeros({ 1 }, torch::dtype(torch::kInt32).device(frustumplane.device()));
    torch::Tensor visible_chunk_id = torch::arange(M, torch::dtype(torch::kInt64).device(frustumplane.device()));
    
    // Launch kernel
    frustum_culling_aabb_kernel<<<(M + 255) / 256, 256 >>>(
        frustumplane.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        aabb_origin.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        aabb_ext.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        visibility.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>()
    );
    
    // Check for errors
    CUDA_CHECK_ERRORS;

    int pred_visible_chunks_num = 0;
    if (feedback_buffer_arg.has_value() && data_idx_arg.has_value())
    {
        int* feedback_buffer = (*feedback_buffer_arg).data_ptr<int>();
        for (int i = 0; i < (*data_idx_arg).size(0); i++)
        {
            int idx=(*data_idx_arg)[i].item().toInt();
            if (feedback_buffer[idx] > pred_visible_chunks_num)
            {
                pred_visible_chunks_num = feedback_buffer[idx];
            }
            cudaMemcpyAsync(&feedback_buffer[idx], visible_chunks_num.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    pred_visible_chunks_num = 1.2f * pred_visible_chunks_num;

    if (pred_visible_chunks_num <= 0)
    {
        cudaMemcpy(&pred_visible_chunks_num, visible_chunks_num.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Return visibility tensor
    visible_chunk_id = visible_chunk_id.slice(0, 0, pred_visible_chunks_num);
    return { visibility,visible_chunks_num,visible_chunk_id };
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
    int view_id,int compacted_chunk_id, int chunk_id,int index,float3 dir,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> SH_base,    //[1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> SH_rest,    //[(deg + 1) ** 2-1,3,chunks_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> rgb         //[batch,3,chunks_num,chunk_size]
)
{
    float3 result;
    result.x = SH_C0 * SH_base[0][0][chunk_id][index];
    result.y = SH_C0 * SH_base[0][1][chunk_id][index];
    result.z = SH_C0 * SH_base[0][2][chunk_id][index];
    if (degree > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result.x = result.x - SH_C1 * y * SH_rest[0][0][chunk_id][index] + SH_C1 * z * SH_rest[1][0][chunk_id][index] - SH_C1 * x * SH_rest[2][0][chunk_id][index];
        result.y = result.y - SH_C1 * y * SH_rest[0][1][chunk_id][index] + SH_C1 * z * SH_rest[1][1][chunk_id][index] - SH_C1 * x * SH_rest[2][1][chunk_id][index];
        result.z = result.z - SH_C1 * y * SH_rest[0][2][chunk_id][index] + SH_C1 * z * SH_rest[1][2][chunk_id][index] - SH_C1 * x * SH_rest[2][2][chunk_id][index];

        if (degree > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result.x = result.x +
                SH_C2[0] * xy * SH_rest[3][0][chunk_id][index] +
                SH_C2[1] * yz * SH_rest[4][0][chunk_id][index] +
                SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][0][chunk_id][index] +
                SH_C2[3] * xz * SH_rest[6][0][chunk_id][index] +
                SH_C2[4] * (xx - yy) * SH_rest[7][0][chunk_id][index];
            result.y = result.y +
                SH_C2[0] * xy * SH_rest[3][1][chunk_id][index] +
                SH_C2[1] * yz * SH_rest[4][1][chunk_id][index] +
                SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][1][chunk_id][index] +
                SH_C2[3] * xz * SH_rest[6][1][chunk_id][index] +
                SH_C2[4] * (xx - yy) * SH_rest[7][1][chunk_id][index];
            result.z = result.z +
                SH_C2[0] * xy * SH_rest[3][2][chunk_id][index] +
                SH_C2[1] * yz * SH_rest[4][2][chunk_id][index] +
                SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][2][chunk_id][index] +
                SH_C2[3] * xz * SH_rest[6][2][chunk_id][index] +
                SH_C2[4] * (xx - yy) * SH_rest[7][2][chunk_id][index];

            if (degree > 2)
            {
                result.x = result.x +
                    SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][0][chunk_id][index] +
                    SH_C3[1] * xy * z * SH_rest[9][0][chunk_id][index] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][0][chunk_id][index] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][0][chunk_id][index] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][0][chunk_id][index] +
                    SH_C3[5] * z * (xx - yy) * SH_rest[13][0][chunk_id][index] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][0][chunk_id][index];
                result.y = result.y +
                    SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][1][chunk_id][index] +
                    SH_C3[1] * xy * z * SH_rest[9][1][chunk_id][index] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][1][chunk_id][index] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][1][chunk_id][index] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][1][chunk_id][index] +
                    SH_C3[5] * z * (xx - yy) * SH_rest[13][1][chunk_id][index] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][1][chunk_id][index];
                result.z = result.z +
                    SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][2][chunk_id][index] +
                    SH_C3[1] * xy * z * SH_rest[9][2][chunk_id][index] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][2][chunk_id][index] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][2][chunk_id][index] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][2][chunk_id][index] +
                    SH_C3[5] * z * (xx - yy) * SH_rest[13][2][chunk_id][index] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][2][chunk_id][index];
            }
        }

    }
    result.x += 0.5f;
    result.y += 0.5f;
    result.z += 0.5f;
    rgb[view_id][0][compacted_chunk_id][index] = result.x;
    rgb[view_id][1][compacted_chunk_id][index] = result.y;
    rgb[view_id][2][compacted_chunk_id][index] = result.z;
}

template <int degree,bool bInit>
__device__ void sh2rgb_backward_kernel(
    int chunk_id,int source_chunk_id, int index, float3 dir,float3 dL_dRGB,
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
        SH_base_grad[0][0][chunk_id][index] = dRGBdsh0 * dL_dRGB.x;
        SH_base_grad[0][1][chunk_id][index] = dRGBdsh0 * dL_dRGB.y;
        SH_base_grad[0][2][chunk_id][index] = dRGBdsh0 * dL_dRGB.z;
    }
    else
    {
        SH_base_grad[0][0][chunk_id][index] += dRGBdsh0 * dL_dRGB.x;
        SH_base_grad[0][1][chunk_id][index] += dRGBdsh0 * dL_dRGB.y;
        SH_base_grad[0][2][chunk_id][index] += dRGBdsh0 * dL_dRGB.z;
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
            SH_rest_grad[0][0][chunk_id][index] = dRGBdsh1 * dL_dRGB.x;
            SH_rest_grad[1][0][chunk_id][index] = dRGBdsh2 * dL_dRGB.x;
            SH_rest_grad[2][0][chunk_id][index] = dRGBdsh3 * dL_dRGB.x;
            SH_rest_grad[0][1][chunk_id][index] = dRGBdsh1 * dL_dRGB.y;
            SH_rest_grad[1][1][chunk_id][index] = dRGBdsh2 * dL_dRGB.y;
            SH_rest_grad[2][1][chunk_id][index] = dRGBdsh3 * dL_dRGB.y;
            SH_rest_grad[0][2][chunk_id][index] = dRGBdsh1 * dL_dRGB.z;
            SH_rest_grad[1][2][chunk_id][index] = dRGBdsh2 * dL_dRGB.z;
            SH_rest_grad[2][2][chunk_id][index] = dRGBdsh3 * dL_dRGB.z;
        }
        else
        {
            SH_rest_grad[0][0][chunk_id][index] += dRGBdsh1 * dL_dRGB.x;
            SH_rest_grad[1][0][chunk_id][index] += dRGBdsh2 * dL_dRGB.x;
            SH_rest_grad[2][0][chunk_id][index] += dRGBdsh3 * dL_dRGB.x;
            SH_rest_grad[0][1][chunk_id][index] += dRGBdsh1 * dL_dRGB.y;
            SH_rest_grad[1][1][chunk_id][index] += dRGBdsh2 * dL_dRGB.y;
            SH_rest_grad[2][1][chunk_id][index] += dRGBdsh3 * dL_dRGB.y;
            SH_rest_grad[0][2][chunk_id][index] += dRGBdsh1 * dL_dRGB.z;
            SH_rest_grad[1][2][chunk_id][index] += dRGBdsh2 * dL_dRGB.z;
            SH_rest_grad[2][2][chunk_id][index] += dRGBdsh3 * dL_dRGB.z;
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
                SH_rest_grad[3][0][chunk_id][index] = dRGBdsh4 * dL_dRGB.x;
                SH_rest_grad[4][0][chunk_id][index] = dRGBdsh5 * dL_dRGB.x;
                SH_rest_grad[5][0][chunk_id][index] = dRGBdsh6 * dL_dRGB.x;
                SH_rest_grad[6][0][chunk_id][index] = dRGBdsh7 * dL_dRGB.x;
                SH_rest_grad[7][0][chunk_id][index] = dRGBdsh8 * dL_dRGB.x;
                SH_rest_grad[3][1][chunk_id][index] = dRGBdsh4 * dL_dRGB.y;
                SH_rest_grad[4][1][chunk_id][index] = dRGBdsh5 * dL_dRGB.y;
                SH_rest_grad[5][1][chunk_id][index] = dRGBdsh6 * dL_dRGB.y;
                SH_rest_grad[6][1][chunk_id][index] = dRGBdsh7 * dL_dRGB.y;
                SH_rest_grad[7][1][chunk_id][index] = dRGBdsh8 * dL_dRGB.y;
                SH_rest_grad[3][2][chunk_id][index] = dRGBdsh4 * dL_dRGB.z;
                SH_rest_grad[4][2][chunk_id][index] = dRGBdsh5 * dL_dRGB.z;
                SH_rest_grad[5][2][chunk_id][index] = dRGBdsh6 * dL_dRGB.z;
                SH_rest_grad[6][2][chunk_id][index] = dRGBdsh7 * dL_dRGB.z;
                SH_rest_grad[7][2][chunk_id][index] = dRGBdsh8 * dL_dRGB.z;
            }
            else
            {
                SH_rest_grad[3][0][chunk_id][index] += dRGBdsh4 * dL_dRGB.x;
                SH_rest_grad[4][0][chunk_id][index] += dRGBdsh5 * dL_dRGB.x;
                SH_rest_grad[5][0][chunk_id][index] += dRGBdsh6 * dL_dRGB.x;
                SH_rest_grad[6][0][chunk_id][index] += dRGBdsh7 * dL_dRGB.x;
                SH_rest_grad[7][0][chunk_id][index] += dRGBdsh8 * dL_dRGB.x;
                SH_rest_grad[3][1][chunk_id][index] += dRGBdsh4 * dL_dRGB.y;
                SH_rest_grad[4][1][chunk_id][index] += dRGBdsh5 * dL_dRGB.y;
                SH_rest_grad[5][1][chunk_id][index] += dRGBdsh6 * dL_dRGB.y;
                SH_rest_grad[6][1][chunk_id][index] += dRGBdsh7 * dL_dRGB.y;
                SH_rest_grad[7][1][chunk_id][index] += dRGBdsh8 * dL_dRGB.y;
                SH_rest_grad[3][2][chunk_id][index] += dRGBdsh4 * dL_dRGB.z;
                SH_rest_grad[4][2][chunk_id][index] += dRGBdsh5 * dL_dRGB.z;
                SH_rest_grad[5][2][chunk_id][index] += dRGBdsh6 * dL_dRGB.z;
                SH_rest_grad[6][2][chunk_id][index] += dRGBdsh7 * dL_dRGB.z;
                SH_rest_grad[7][2][chunk_id][index] += dRGBdsh8 * dL_dRGB.z;
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
                    SH_rest_grad[8][0][chunk_id][index] = dRGBdsh9 * dL_dRGB.x;
                    SH_rest_grad[9][0][chunk_id][index] = dRGBdsh10 * dL_dRGB.x;
                    SH_rest_grad[10][0][chunk_id][index] = dRGBdsh11 * dL_dRGB.x;
                    SH_rest_grad[11][0][chunk_id][index] = dRGBdsh12 * dL_dRGB.x;
                    SH_rest_grad[12][0][chunk_id][index] = dRGBdsh13 * dL_dRGB.x;
                    SH_rest_grad[13][0][chunk_id][index] = dRGBdsh14 * dL_dRGB.x;
                    SH_rest_grad[14][0][chunk_id][index] = dRGBdsh15 * dL_dRGB.x;
                    SH_rest_grad[8][1][chunk_id][index] = dRGBdsh9 * dL_dRGB.y;
                    SH_rest_grad[9][1][chunk_id][index] = dRGBdsh10 * dL_dRGB.y;
                    SH_rest_grad[10][1][chunk_id][index] = dRGBdsh11 * dL_dRGB.y;
                    SH_rest_grad[11][1][chunk_id][index] = dRGBdsh12 * dL_dRGB.y;
                    SH_rest_grad[12][1][chunk_id][index] = dRGBdsh13 * dL_dRGB.y;
                    SH_rest_grad[13][1][chunk_id][index] = dRGBdsh14 * dL_dRGB.y;
                    SH_rest_grad[14][1][chunk_id][index] = dRGBdsh15 * dL_dRGB.y;
                    SH_rest_grad[8][2][chunk_id][index] = dRGBdsh9 * dL_dRGB.z;
                    SH_rest_grad[9][2][chunk_id][index] = dRGBdsh10 * dL_dRGB.z;
                    SH_rest_grad[10][2][chunk_id][index] = dRGBdsh11 * dL_dRGB.z;
                    SH_rest_grad[11][2][chunk_id][index] = dRGBdsh12 * dL_dRGB.z;
                    SH_rest_grad[12][2][chunk_id][index] = dRGBdsh13 * dL_dRGB.z;
                    SH_rest_grad[13][2][chunk_id][index] = dRGBdsh14 * dL_dRGB.z;
                    SH_rest_grad[14][2][chunk_id][index] = dRGBdsh15 * dL_dRGB.z;
                }
                else
                {
                    SH_rest_grad[8][0][chunk_id][index] += dRGBdsh9 * dL_dRGB.x;
                    SH_rest_grad[9][0][chunk_id][index] += dRGBdsh10 * dL_dRGB.x;
                    SH_rest_grad[10][0][chunk_id][index] += dRGBdsh11 * dL_dRGB.x;
                    SH_rest_grad[11][0][chunk_id][index] += dRGBdsh12 * dL_dRGB.x;
                    SH_rest_grad[12][0][chunk_id][index] += dRGBdsh13 * dL_dRGB.x;
                    SH_rest_grad[13][0][chunk_id][index] += dRGBdsh14 * dL_dRGB.x;
                    SH_rest_grad[14][0][chunk_id][index] += dRGBdsh15 * dL_dRGB.x;
                    SH_rest_grad[8][1][chunk_id][index] += dRGBdsh9 * dL_dRGB.y;
                    SH_rest_grad[9][1][chunk_id][index] += dRGBdsh10 * dL_dRGB.y;
                    SH_rest_grad[10][1][chunk_id][index] += dRGBdsh11 * dL_dRGB.y;
                    SH_rest_grad[11][1][chunk_id][index] += dRGBdsh12 * dL_dRGB.y;
                    SH_rest_grad[12][1][chunk_id][index] += dRGBdsh13 * dL_dRGB.y;
                    SH_rest_grad[13][1][chunk_id][index] += dRGBdsh14 * dL_dRGB.y;
                    SH_rest_grad[14][1][chunk_id][index] += dRGBdsh15 * dL_dRGB.y;
                    SH_rest_grad[8][2][chunk_id][index] += dRGBdsh9 * dL_dRGB.z;
                    SH_rest_grad[9][2][chunk_id][index] += dRGBdsh10 * dL_dRGB.z;
                    SH_rest_grad[10][2][chunk_id][index] += dRGBdsh11 * dL_dRGB.z;
                    SH_rest_grad[11][2][chunk_id][index] += dRGBdsh12 * dL_dRGB.z;
                    SH_rest_grad[12][2][chunk_id][index] += dRGBdsh13 * dL_dRGB.z;
                    SH_rest_grad[13][2][chunk_id][index] += dRGBdsh14 * dL_dRGB.z;
                    SH_rest_grad[14][2][chunk_id][index] += dRGBdsh15 * dL_dRGB.z;
                }
            }
        }

    }
    
}

template <int degree>
__global__ void activate_forward_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunk_id,    //[allocate_size] 
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_chunks_num,    //[1] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,    //[views_num,4,4] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,    //[1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,    //[?,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunks_num,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_position,    //[4,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_scale,    //[3,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_rotation,    //[4,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> color,    //[views_num,3,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_opacity    //[1,allocate_size,chunk_size]
)
{
    int index = threadIdx.x;
    int compacted_chunk_id = blockIdx.x;
    int chunk_id = visible_chunk_id[blockIdx.x];
    if (compacted_chunk_id <visible_chunks_num[0])
    {
        float3 pos{ position[0][chunk_id][index],position[1][chunk_id][index],position[2][chunk_id][index] };
        activated_position[0][compacted_chunk_id][index] = pos.x;
        activated_position[1][compacted_chunk_id][index] = pos.y;
        activated_position[2][compacted_chunk_id][index] = pos.z;
        activated_position[3][compacted_chunk_id][index] = 1.0f;

        activated_scale[0][compacted_chunk_id][index] = __expf(scale[0][chunk_id][index]);
        activated_scale[1][compacted_chunk_id][index] = __expf(scale[1][chunk_id][index]);
        activated_scale[2][compacted_chunk_id][index] = __expf(scale[2][chunk_id][index]);

        float w = rotation[0][chunk_id][index];
        float x = rotation[1][chunk_id][index];
        float y = rotation[2][chunk_id][index];
        float z = rotation[3][chunk_id][index];

        float recp_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
        activated_rotation[0][compacted_chunk_id][index] = w * recp_norm;
        activated_rotation[1][compacted_chunk_id][index] = x * recp_norm;
        activated_rotation[2][compacted_chunk_id][index] = y * recp_norm;
        activated_rotation[3][compacted_chunk_id][index] = z * recp_norm;

        activated_opacity[0][compacted_chunk_id][index]= 1.0f / (1.0f + __expf(-opacity[0][chunk_id][index]));

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
            sh2rgb_forward_kernel<degree>(view_id, compacted_chunk_id, chunk_id, index, dir, sh_base, sh_rest, color);
        }
    }
    else
    {
        activated_opacity[0][compacted_chunk_id][threadIdx.x] = 0.0f;
    }

}

template <int degree>
__global__ void activate_backward_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunk_id,    //[allocate_size] 
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_chunks_num,    //[1] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,    //[views_num,4,4] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,    //[1,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,    //[?,3,chunks_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_position_grad,    //[4,allocate_size,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_scale_grad,    //[3,allocate_size,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_rotation_grad,    //[4,allocate_size,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> color_grad,    //[views_num,3,allocate_size,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_opacity_grad,    //[1,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position_grad,    //[3,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale_grad,    //[3,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation_grad,    //[4,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base_grad,    //[1,3,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest_grad,    //[?,3,allocate_size,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity_grad    //[1,allocate_size,chunk_size]
)
{
    int index = threadIdx.x;
    int chunk_id = blockIdx.x;
    int source_chunk_id = visible_chunk_id[chunk_id];

    if (chunk_id < visible_chunks_num[0])
    {
        position_grad[0][chunk_id][index] = activated_position_grad[0][chunk_id][index];
        position_grad[1][chunk_id][index] = activated_position_grad[1][chunk_id][index];
        position_grad[2][chunk_id][index] = activated_position_grad[2][chunk_id][index];

        scale_grad[0][chunk_id][index] = __expf(scale[0][source_chunk_id][index]) * activated_scale_grad[0][chunk_id][index];
        scale_grad[1][chunk_id][index] = __expf(scale[1][source_chunk_id][index]) * activated_scale_grad[1][chunk_id][index];
        scale_grad[2][chunk_id][index] = __expf(scale[2][source_chunk_id][index]) * activated_scale_grad[2][chunk_id][index];

        float w = rotation[0][source_chunk_id][index];
        float x = rotation[1][source_chunk_id][index];
        float y = rotation[2][source_chunk_id][index];
        float z = rotation[3][source_chunk_id][index];
        float recp_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
        float out_w = w * recp_norm;
        float out_x = x * recp_norm;
        float out_y = y * recp_norm;
        float out_z = z * recp_norm;
        float dot = activated_rotation_grad[0][chunk_id][index] * out_w
            + activated_rotation_grad[1][chunk_id][index] * out_x
            + activated_rotation_grad[2][chunk_id][index] * out_y
            + activated_rotation_grad[3][chunk_id][index] * out_z;

        rotation_grad[0][chunk_id][index] = recp_norm * (activated_rotation_grad[0][chunk_id][index] - dot * out_w);
        rotation_grad[1][chunk_id][index] = recp_norm * (activated_rotation_grad[1][chunk_id][index] - dot * out_x);
        rotation_grad[2][chunk_id][index] = recp_norm * (activated_rotation_grad[2][chunk_id][index] - dot * out_y);
        rotation_grad[3][chunk_id][index] = recp_norm * (activated_rotation_grad[3][chunk_id][index] - dot * out_z);

        opacity_grad[0][chunk_id][index] = activated_opacity_grad[0][chunk_id][index] * (1.0f - 1.0f / (1.0f + __expf(opacity[0][source_chunk_id][index])));

        //sh
        for (int view_id = 0; view_id < view_matrix.size(0); view_id++)
        {
            //-t @ rot.trans()
            float3 inv_trans{ -view_matrix[view_id][3][0],-view_matrix[view_id][3][1],-view_matrix[view_id][3][2] };
            float3 camera_center;
            camera_center.x = inv_trans.x * view_matrix[view_id][0][0] + inv_trans.y * view_matrix[view_id][0][1] + inv_trans.z * view_matrix[view_id][0][2];
            camera_center.y = inv_trans.x * view_matrix[view_id][1][0] + inv_trans.y * view_matrix[view_id][1][1] + inv_trans.z * view_matrix[view_id][1][2];
            camera_center.z = inv_trans.x * view_matrix[view_id][2][0] + inv_trans.y * view_matrix[view_id][2][1] + inv_trans.z * view_matrix[view_id][2][2];
            float3 dir{ position[0][source_chunk_id][index] - camera_center.x,position[1][source_chunk_id][index] - camera_center.y,position[2][source_chunk_id][index] - camera_center.z };
            float norm_recp = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-12f);
            dir.x *= norm_recp;
            dir.y *= norm_recp;
            dir.z *= norm_recp;
            float3 dL_dRGB{ color_grad[view_id][0][chunk_id][index], color_grad[view_id][1][chunk_id][index], color_grad[view_id][2][chunk_id][index] };
            if (view_id == 0)
            {
                sh2rgb_backward_kernel<degree, true>(chunk_id, source_chunk_id, index, dir, dL_dRGB, sh_base, sh_rest, color_grad, sh_base_grad, sh_rest_grad);
            }
            else
            {
                sh2rgb_backward_kernel<degree, false>(chunk_id, source_chunk_id, index, dir, dL_dRGB, sh_base, sh_rest, color_grad, sh_base_grad, sh_rest_grad);
            }
        }
    }
    
}


std::vector<at::Tensor> cull_compact_activate(
    int sh_degree,
    at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
    at::Tensor view_matrix,
    at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity
)
{

    //activate
    int chunksize = position.size(2);
    int allocate_chunks_num = visible_chunk_id.size(0);
    auto tensor_shape = position.sizes();
    at::Tensor actived_position = torch::empty({ 4, allocate_chunks_num, chunksize }, position.options());
    tensor_shape = scale.sizes();
    at::Tensor actived_scale = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, scale.options());
    tensor_shape = rotation.sizes();
    at::Tensor actived_rotation = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, rotation.options());
    tensor_shape = sh_base.sizes();
    at::Tensor color = torch::empty({ 1,3, allocate_chunks_num, chunksize }, sh_base.options());
    tensor_shape = opacity.sizes();
    at::Tensor actived_opacity = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, opacity.options());

    //todo sh_degree
    switch (sh_degree)
    {
    case 0:
        activate_forward_kernel<0> <<<allocate_chunks_num, chunksize,0 >>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
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
            actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
            );
        break;
    case 1:
        activate_forward_kernel<1> << <allocate_chunks_num, chunksize, 0 >> > (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
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
            actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
            );
        break;
    case 2:
        activate_forward_kernel<2> << <allocate_chunks_num, chunksize, 0 >> > (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
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
            actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
            );
        break;
    case 3:
        activate_forward_kernel<3> << <allocate_chunks_num, chunksize, 0 >> > (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
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
            actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
            );
        break;
    default:
        assert(false);
    }

    return { actived_position,actived_scale,actived_rotation,color,actived_opacity };
}

std::vector<at::Tensor> activate_backward(
    int sh_degree,
    at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
    at::Tensor view_matrix,
    at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity,
    at::Tensor activated_position_grad, at::Tensor activated_scale_grad, at::Tensor activated_rotation_grad, at::Tensor color_grad, at::Tensor activated_opacity_grad
)
{
    int allocate_chunks_num = visible_chunk_id.size(0);
    int chunksize = position.size(2);

    auto tensor_shape = position.sizes();
    at::Tensor compacted_position_grad = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, position.options());
    tensor_shape = scale.sizes();
    at::Tensor compacted_scale_grad = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, scale.options());
    tensor_shape = rotation.sizes();
    at::Tensor compacted_rotation_grad = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, rotation.options());
    tensor_shape = sh_base.sizes();
    at::Tensor compacted_sh_base_grad = torch::empty({ tensor_shape[0],tensor_shape[1], allocate_chunks_num, chunksize }, sh_base.options());
    tensor_shape = sh_rest.sizes();
    at::Tensor compacted_sh_rest_grad = torch::zeros({ tensor_shape[0],tensor_shape[1], allocate_chunks_num, chunksize }, sh_rest.options());
    tensor_shape = opacity.sizes();
    at::Tensor compacted_opacity_grad = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, opacity.options());

    switch (sh_degree)
    {
    case 0:
        activate_backward_kernel<0> << <allocate_chunks_num, chunksize >> > (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
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
        activate_backward_kernel<1> << <allocate_chunks_num, chunksize >> > (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
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
        activate_backward_kernel<2> << <allocate_chunks_num, chunksize >> > (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
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
        activate_backward_kernel<3> << <allocate_chunks_num, chunksize >> > (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
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


enum OpType {
    OP_ADD = 0,
    OP_MIN = 1,
    OP_MAX = 2
};

template <typename scalar_t, int OP>
__global__ void sparse_scatter_kernel(
    scalar_t* __restrict__ A,                // [ele_num, chunks_num, chunk_size]
    const scalar_t* __restrict__ B,          // [ele_num, allocated_chunks_num, chunk_size]
    const int64_t* __restrict__ chunk_ids,       // [allocated_chunks_num]
    const int* __restrict__ valid_count_ptr, // [1]
    const int chunks_num,
    const int allocated_chunks_num
) 
{
    const int source_chunk = blockIdx.x;  // allocated_chunk_idx
    const int ele_index = blockIdx.y;

    if (source_chunk >= *valid_count_ptr) return;

    // 3. 获取目标 Chunk ID
    const int target_chunk = chunk_ids[source_chunk];

    const int64_t offset_B = ele_index * allocated_chunks_num * blockDim.x + source_chunk * blockDim.x + threadIdx.x;

    const int64_t offset_A = ele_index * chunks_num * blockDim.x + target_chunk * blockDim.x + threadIdx.x;

    // 5. 执行计算
    scalar_t val_b = B[offset_B];

    if constexpr (OP == OP_ADD) {
        A[offset_A] += val_b;
    }
    else if constexpr (OP == OP_MIN) {
        A[offset_A] = min(A[offset_A], val_b);
    }
    else if constexpr (OP == OP_MAX) {
        A[offset_A] = max(A[offset_A], val_b);
    }
}

void gpu_driven_pipeline_sparse_op(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor visible_chunk_ids,
    torch::Tensor visible_count,
    std::string op_name // 0:add, 1:min, 2:max
) 
{
    // check input
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(visible_chunk_ids.is_cuda(), "visible_chunk_ids must be a CUDA tensor");
    TORCH_CHECK(visible_count.is_cuda(), "visible_count must be a CUDA tensor");
    OpType op_type;
    if (op_name == "add" || op_name == "sum") 
    {
        op_type = OP_ADD;
    }
    else if (op_name == "min") 
    {
        op_type = OP_MIN;
    }
    else if (op_name == "max") 
    {
        op_type = OP_MAX;
    }
    else 
    {
        TORCH_CHECK(false, "Unsupported op: ", op_name, ". Expected: add, min, max");
    }

    // A: [ele_num, chunks_num, chunk_size]
    // B: [ele_num, allocated_chunks_num, chunk_size]
    const int ele_num = A.size(0);
    const int chunks_num = A.size(1);
    const int chunk_size = A.size(2);
    const int allocated_chunks_num = B.size(1);
    TORCH_CHECK(chunk_size <= 1024, "chunk_size exceeds max threads per block");

    // 配置 Kernel Launch 参数
    // grid.x -> allocated_chunks (dim 1)
    // grid.y -> ele_num (dim 0)
    dim3 grid(allocated_chunks_num, ele_num, 1);
    dim3 block(chunk_size, 1, 1);

    // 调度 Kernel
    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "sparse_scatter_kernel",
        [&] {
            switch (op_type) {
            case OP_ADD:
                sparse_scatter_kernel<scalar_t, OP_ADD> << <grid, block >> > (
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    visible_chunk_ids.data_ptr<int64_t>(),
                    visible_count.data_ptr<int>(),
                    chunks_num, allocated_chunks_num
                    );
                break;
            case OP_MIN:
                sparse_scatter_kernel<scalar_t, OP_MIN> << <grid, block >> > (
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    visible_chunk_ids.data_ptr<int64_t>(),
                    visible_count.data_ptr<int>(),
                    chunks_num, allocated_chunks_num
                    );
                break;
            case OP_MAX:
                sparse_scatter_kernel<scalar_t, OP_MAX> << <grid, block >> > (
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    visible_chunk_ids.data_ptr<int64_t>(),
                    visible_count.data_ptr<int>(),
                    chunks_num, allocated_chunks_num
                    );
                break;
            }
        }
    );
}

// ========================================================================
// New: Compact + Activate WITHOUT SH (only exp, normalize, sigmoid)
// ========================================================================

__global__ void activate_forward_kernel_nosh(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunk_id,    //[allocate_size]
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_chunks_num,    //[1]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[3,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunks_num,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_position,    //[4,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_scale,    //[3,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_rotation,    //[4,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_opacity    //[1,allocate_size,chunk_size]
)
{
    int index = threadIdx.x;
    int compacted_chunk_id = blockIdx.x;
    int chunk_id = visible_chunk_id[blockIdx.x];

    if (compacted_chunk_id < visible_chunks_num[0])
    {
        // Position: add homogeneous coordinate w=1.0 (with compact)
        activated_position[0][compacted_chunk_id][index] = position[0][chunk_id][index];
        activated_position[1][compacted_chunk_id][index] = position[1][chunk_id][index];
        activated_position[2][compacted_chunk_id][index] = position[2][chunk_id][index];
        activated_position[3][compacted_chunk_id][index] = 1.0f;

        // Scale: exp (with compact)
        activated_scale[0][compacted_chunk_id][index] = __expf(scale[0][chunk_id][index]);
        activated_scale[1][compacted_chunk_id][index] = __expf(scale[1][chunk_id][index]);
        activated_scale[2][compacted_chunk_id][index] = __expf(scale[2][chunk_id][index]);

        // Rotation: normalize quaternion (with compact)
        float w = rotation[0][chunk_id][index];
        float x = rotation[1][chunk_id][index];
        float y = rotation[2][chunk_id][index];
        float z = rotation[3][chunk_id][index];
        float recp_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
        activated_rotation[0][compacted_chunk_id][index] = w * recp_norm;
        activated_rotation[1][compacted_chunk_id][index] = x * recp_norm;
        activated_rotation[2][compacted_chunk_id][index] = y * recp_norm;
        activated_rotation[3][compacted_chunk_id][index] = z * recp_norm;

        // Opacity: sigmoid (with compact)
        activated_opacity[0][compacted_chunk_id][index] = 1.0f / (1.0f + __expf(-opacity[0][chunk_id][index]));
    }
    else
    {
        activated_opacity[0][compacted_chunk_id][threadIdx.x] = 0.0f;
    }
}

__global__ void activate_backward_kernel_nosh(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunk_id,    //[allocate_size]
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_chunks_num,    //[1]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_position_grad,    //[4,allocate_size,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_scale_grad,    //[3,allocate_size,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_rotation_grad,    //[4,allocate_size,chunk_size]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> activated_opacity_grad,    //[1,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position_grad,    //[3,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale_grad,    //[3,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation_grad,    //[4,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity_grad    //[1,allocate_size,chunk_size]
)
{
    int index = threadIdx.x;
    int chunk_id = blockIdx.x;
    int source_chunk_id = visible_chunk_id[blockIdx.x];

    if (chunk_id < visible_chunks_num[0])
    {
        // Position gradient (with compact)
        position_grad[0][chunk_id][index] = activated_position_grad[0][chunk_id][index];
        position_grad[1][chunk_id][index] = activated_position_grad[1][chunk_id][index];
        position_grad[2][chunk_id][index] = activated_position_grad[2][chunk_id][index];

        // Scale gradient: d(exp(x))/dx = exp(x) (with compact)
        scale_grad[0][chunk_id][index] = __expf(scale[0][source_chunk_id][index]) * activated_scale_grad[0][chunk_id][index];
        scale_grad[1][chunk_id][index] = __expf(scale[1][source_chunk_id][index]) * activated_scale_grad[1][chunk_id][index];
        scale_grad[2][chunk_id][index] = __expf(scale[2][source_chunk_id][index]) * activated_scale_grad[2][chunk_id][index];

        // Rotation gradient (with compact)
        float w = rotation[0][source_chunk_id][index];
        float x = rotation[1][source_chunk_id][index];
        float y = rotation[2][source_chunk_id][index];
        float z = rotation[3][source_chunk_id][index];
        float recp_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
        float out_w = w * recp_norm;
        float out_x = x * recp_norm;
        float out_y = y * recp_norm;
        float out_z = z * recp_norm;
        float dot = activated_rotation_grad[0][chunk_id][index] * out_w
            + activated_rotation_grad[1][chunk_id][index] * out_x
            + activated_rotation_grad[2][chunk_id][index] * out_y
            + activated_rotation_grad[3][chunk_id][index] * out_z;

        rotation_grad[0][chunk_id][index] = recp_norm * (activated_rotation_grad[0][chunk_id][index] - dot * out_w);
        rotation_grad[1][chunk_id][index] = recp_norm * (activated_rotation_grad[1][chunk_id][index] - dot * out_x);
        rotation_grad[2][chunk_id][index] = recp_norm * (activated_rotation_grad[2][chunk_id][index] - dot * out_y);
        rotation_grad[3][chunk_id][index] = recp_norm * (activated_rotation_grad[3][chunk_id][index] - dot * out_z);

        // Opacity gradient (with compact)
        opacity_grad[0][chunk_id][index] = activated_opacity_grad[0][chunk_id][index] * (1.0f - 1.0f / (1.0f + __expf(opacity[0][source_chunk_id][index])));
    }
}

std::vector<at::Tensor> cull_compact_activate_nosh(
    at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
    at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor opacity
)
{
    int chunksize = position.size(2);
    int allocate_chunks_num = visible_chunk_id.size(0);

    auto tensor_shape = position.sizes();
    at::Tensor actived_position = torch::empty({ 4, allocate_chunks_num, chunksize }, position.options());
    tensor_shape = scale.sizes();
    at::Tensor actived_scale = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, scale.options());
    tensor_shape = rotation.sizes();
    at::Tensor actived_rotation = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, rotation.options());
    tensor_shape = opacity.sizes();
    at::Tensor actived_opacity = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, opacity.options());

    activate_forward_kernel_nosh <<<allocate_chunks_num, chunksize, 0>>> (
        visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        actived_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        actived_scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        actived_rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        actived_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
    );

    return { actived_position, actived_scale, actived_rotation, actived_opacity };
}

std::vector<at::Tensor> activate_backward_nosh(
    at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
    at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor opacity,
    at::Tensor activated_position_grad, at::Tensor activated_scale_grad, at::Tensor activated_rotation_grad, at::Tensor activated_opacity_grad
)
{
    int chunksize = position.size(2);
    int allocate_chunks_num = visible_chunk_id.size(0);

    auto tensor_shape = position.sizes();
    at::Tensor position_grad = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, position.options());
    tensor_shape = scale.sizes();
    at::Tensor scale_grad = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, scale.options());
    tensor_shape = rotation.sizes();
    at::Tensor rotation_grad = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, rotation.options());
    tensor_shape = opacity.sizes();
    at::Tensor opacity_grad = torch::empty({ tensor_shape[0], allocate_chunks_num, chunksize }, opacity.options());

    activate_backward_kernel_nosh <<<allocate_chunks_num, chunksize, 0>>> (
        visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        activated_position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        activated_scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        activated_rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        activated_opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
    );

    return { position_grad, scale_grad, rotation_grad, opacity_grad };
}

// ========================================================================
// New: Compact + SH ONLY (no activate, just compact and compute SH to RGB)
// ========================================================================

template <int degree>
__global__ void compact_sh_forward_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunk_id,    //[allocate_size]
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_chunks_num,    //[1]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,    //[views_num,4,4]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[3,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,    //[1,3,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,    //[?,3,chunks_num,chunk_size]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> color    //[views_num,3,allocate_size,chunk_size]
)
{
    int index = threadIdx.x;
    int compacted_chunk_id = blockIdx.x;
    int chunk_id = visible_chunk_id[blockIdx.x];

    if (compacted_chunk_id < visible_chunks_num[0])
    {
        // Get position for view direction calculation (with compact)
        float3 pos{ position[0][chunk_id][index], position[1][chunk_id][index], position[2][chunk_id][index] };

        // Compute SH to RGB for each view (with compact)
        for (int view_id = 0; view_id < view_matrix.size(0); view_id++)
        {
            // Calculate camera center from view matrix
            float3 inv_trans{ -view_matrix[view_id][3][0], -view_matrix[view_id][3][1], -view_matrix[view_id][3][2] };
            float3 camera_center;
            camera_center.x = inv_trans.x * view_matrix[view_id][0][0] + inv_trans.y * view_matrix[view_id][0][1] + inv_trans.z * view_matrix[view_id][0][2];
            camera_center.y = inv_trans.x * view_matrix[view_id][1][0] + inv_trans.y * view_matrix[view_id][1][1] + inv_trans.z * view_matrix[view_id][1][2];
            camera_center.z = inv_trans.x * view_matrix[view_id][2][0] + inv_trans.y * view_matrix[view_id][2][1] + inv_trans.z * view_matrix[view_id][2][2];

            // Direction from camera to gaussian (with compact)
            float3 dir{ pos.x - camera_center.x, pos.y - camera_center.y, pos.z - camera_center.z };
            float norm_recp = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-12f);
            dir.x *= norm_recp;
            dir.y *= norm_recp;
            dir.z *= norm_recp;

            // SH to RGB (with compact using compacted_chunk_id)
            sh2rgb_forward_kernel<degree>(view_id, compacted_chunk_id, chunk_id, index, dir, sh_base, sh_rest, color);
        }
    }
}

template <int degree>
__global__ void compact_sh_backward_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunk_id,    //[allocate_size]
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_chunks_num,    //[1]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,    //[views_num,4,4]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[3,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,    //[1,3,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,    //[?,3,chunks_num,chunk_size]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> color_grad,    //[views_num,3,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base_grad,    //[1,3,allocate_size,chunk_size]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest_grad    //[?,3,allocate_size,chunk_size]
)
{
    int index = threadIdx.x;
    int chunk_id = blockIdx.x;
    int source_chunk_id = visible_chunk_id[blockIdx.x];

    if (chunk_id < visible_chunks_num[0])
    {
        // Compute SH backward for each view (with compact)
        for (int view_id = 0; view_id < view_matrix.size(0); view_id++)
        {
            // Calculate camera center from view matrix
            float3 inv_trans{ -view_matrix[view_id][3][0], -view_matrix[view_id][3][1], -view_matrix[view_id][3][2] };
            float3 camera_center;
            camera_center.x = inv_trans.x * view_matrix[view_id][0][0] + inv_trans.y * view_matrix[view_id][0][1] + inv_trans.z * view_matrix[view_id][0][2];
            camera_center.y = inv_trans.x * view_matrix[view_id][1][0] + inv_trans.y * view_matrix[view_id][1][1] + inv_trans.z * view_matrix[view_id][1][2];
            camera_center.z = inv_trans.x * view_matrix[view_id][2][0] + inv_trans.y * view_matrix[view_id][2][1] + inv_trans.z * view_matrix[view_id][2][2];

            // Direction from camera to gaussian (with compact)
            float3 dir{ position[0][source_chunk_id][index] - camera_center.x,
                        position[1][source_chunk_id][index] - camera_center.y,
                        position[2][source_chunk_id][index] - camera_center.z };
            float norm_recp = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-12f);
            dir.x *= norm_recp;
            dir.y *= norm_recp;
            dir.z *= norm_recp;

            float3 dL_dRGB{ color_grad[view_id][0][chunk_id][index], color_grad[view_id][1][chunk_id][index], color_grad[view_id][2][chunk_id][index] };

            // Only accumulate sh_base_grad for view_id == 0
            if (view_id == 0)
            {
                sh2rgb_backward_kernel<degree, true>(chunk_id, source_chunk_id, index, dir, dL_dRGB, sh_base, sh_rest, color_grad, sh_base_grad, sh_rest_grad);
            }
            else
            {
                sh2rgb_backward_kernel<degree, false>(chunk_id, source_chunk_id, index, dir, dL_dRGB, sh_base, sh_rest, color_grad, sh_base_grad, sh_rest_grad);
            }
        }
    }
}

at::Tensor compact_sh_forward(
    int sh_degree,
    at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
    at::Tensor view_matrix,
    at::Tensor position, at::Tensor sh_base, at::Tensor sh_rest
)
{
    int chunksize = position.size(2);
    int allocate_chunks_num = visible_chunk_id.size(0);
    int views_num = view_matrix.size(0);

    at::Tensor color = torch::empty({ views_num, 3, allocate_chunks_num, chunksize }, sh_base.options());

    switch (sh_degree)
    {
    case 0:
        compact_sh_forward_kernel<0> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
        );
        break;
    case 1:
        compact_sh_forward_kernel<1> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
        );
        break;
    case 2:
        compact_sh_forward_kernel<2> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
        );
        break;
    case 3:
        compact_sh_forward_kernel<3> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
        );
        break;
    default:
        AT_ERROR("Unsupported sh_degree: ", sh_degree);
    }
    CUDA_CHECK_ERRORS;
    return color;
}

std::vector<at::Tensor> compact_sh_backward(
    int sh_degree,
    at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
    at::Tensor view_matrix,
    at::Tensor position, at::Tensor sh_base, at::Tensor sh_rest,
    at::Tensor color_grad
)
{
    int chunksize = position.size(2);
    int allocate_chunks_num = visible_chunk_id.size(0);

    auto tensor_shape = sh_base.sizes();
    at::Tensor sh_base_grad = torch::empty({ tensor_shape[0],tensor_shape[1], allocate_chunks_num, chunksize }, sh_base.options());
    tensor_shape = sh_rest.sizes();
    at::Tensor sh_rest_grad = torch::zeros({ tensor_shape[0],tensor_shape[1], allocate_chunks_num, chunksize }, sh_rest.options());

    switch (sh_degree)
    {
    case 0:
        compact_sh_backward_kernel<0> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
        );
        break;
    case 1:
        compact_sh_backward_kernel<1> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
        );
        break;
    case 2:
        compact_sh_backward_kernel<2> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
        );
        break;
    case 3:
        compact_sh_backward_kernel<3> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
        );
        break;
    default:
        AT_ERROR("Unsupported sh_degree: ", sh_degree);
    }
    CUDA_CHECK_ERRORS;
    return { sh_base_grad, sh_rest_grad };
}

// ============================================================================
// Fused compact SH backward + Adam update kernel
// ============================================================================
template <int degree>
__global__ void compact_sh_backward_adam_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_chunk_id,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> visible_chunks_num,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> color_grad,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> exp_avg_sh_base,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> exp_avg_sq_sh_base,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> exp_avg_sh_rest,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> exp_avg_sq_sh_rest,
    const float sh_base_lr, const float sh_rest_lr, const float b1, const float b2, const float eps
)
{
    int index = threadIdx.x;
    int chunk_id = blockIdx.x;

    // 【安全修复】：原代码在这里直接取 visible_chunk_id[chunk_id]，如果越界会触发非法内存访问
    // 必须先做边界检查，再读取！
    if (chunk_id >= visible_chunks_num[0]) return;

    int source_chunk_id = visible_chunk_id[chunk_id];
    int views_count = view_matrix.size(0);

    // 【访存优化】：提前把 position 读进寄存器缓存，避免后面各个 degree 循环里重复去 Global Memory 取数据
    float3 pos = {
        position[0][source_chunk_id][index],
        position[1][source_chunk_id][index],
        position[2][source_chunk_id][index]
    };

    // ==========================================
    // Degree 0 (SH Base)
    // ==========================================
    {
        float3 dL_dsh_base_accum{ 0.0f, 0.0f, 0.0f };
        float dRGBdsh0 = SH_C0;

        for (int view_id = 0; view_id < views_count; view_id++)
        {
            float3 dL_dRGB{
                color_grad[view_id][0][chunk_id][index],
                color_grad[view_id][1][chunk_id][index],
                color_grad[view_id][2][chunk_id][index]
            };

            dL_dsh_base_accum.x += dRGBdsh0 * dL_dRGB.x;
            dL_dsh_base_accum.y += dRGBdsh0 * dL_dRGB.y;
            dL_dsh_base_accum.z += dRGBdsh0 * dL_dRGB.z;
        }

        // 算完马上执行 Adam 并清空逻辑
        for (int rgb_idx = 0; rgb_idx < 3; rgb_idx++)
        {
            float grad = (rgb_idx == 0) ? dL_dsh_base_accum.x :
                (rgb_idx == 1) ? dL_dsh_base_accum.y : dL_dsh_base_accum.z;

            float& exp_avg = exp_avg_sh_base[0][rgb_idx][source_chunk_id][index];
            float& exp_avg_sq = exp_avg_sq_sh_base[0][rgb_idx][source_chunk_id][index];
            float& param = sh_base[0][rgb_idx][source_chunk_id][index];

            exp_avg = b1 * exp_avg + (1.0f - b1) * grad;
            exp_avg_sq = b2 * exp_avg_sq + (1.0f - b2) * grad * grad;
            param += -sh_base_lr * exp_avg / (sqrtf(exp_avg_sq) + eps);
        }
    } // <-- 离开作用域，dL_dsh_base_accum 占用的寄存器被立即释放！

    // ==========================================
    // Degree 1 (SH Rest indices 0~2)
    // ==========================================
    if (degree > 0)
    {
        // 局部累加数组，仅占用 9 个 float 寄存器
        float dL_dsh_rest_accum[3][3] = { 0.0f };

        for (int view_id = 0; view_id < views_count; view_id++)
        {
            float3 inv_trans{ -view_matrix[view_id][3][0], -view_matrix[view_id][3][1], -view_matrix[view_id][3][2] };
            float3 camera_center{
                inv_trans.x * view_matrix[view_id][0][0] + inv_trans.y * view_matrix[view_id][0][1] + inv_trans.z * view_matrix[view_id][0][2],
                inv_trans.x * view_matrix[view_id][1][0] + inv_trans.y * view_matrix[view_id][1][1] + inv_trans.z * view_matrix[view_id][1][2],
                inv_trans.x * view_matrix[view_id][2][0] + inv_trans.y * view_matrix[view_id][2][1] + inv_trans.z * view_matrix[view_id][2][2]
            };

            float3 dir{ pos.x - camera_center.x, pos.y - camera_center.y, pos.z - camera_center.z };
            float norm_recp = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-12f);
            float x = dir.x * norm_recp;
            float y = dir.y * norm_recp;
            float z = dir.z * norm_recp;

            float3 dL_dRGB{
                color_grad[view_id][0][chunk_id][index],
                color_grad[view_id][1][chunk_id][index],
                color_grad[view_id][2][chunk_id][index]
            };

            float dRGBdsh1 = -SH_C1 * y;
            float dRGBdsh2 = SH_C1 * z;
            float dRGBdsh3 = -SH_C1 * x;

            dL_dsh_rest_accum[0][0] += dRGBdsh1 * dL_dRGB.x; dL_dsh_rest_accum[0][1] += dRGBdsh1 * dL_dRGB.y; dL_dsh_rest_accum[0][2] += dRGBdsh1 * dL_dRGB.z;
            dL_dsh_rest_accum[1][0] += dRGBdsh2 * dL_dRGB.x; dL_dsh_rest_accum[1][1] += dRGBdsh2 * dL_dRGB.y; dL_dsh_rest_accum[1][2] += dRGBdsh2 * dL_dRGB.z;
            dL_dsh_rest_accum[2][0] += dRGBdsh3 * dL_dRGB.x; dL_dsh_rest_accum[2][1] += dRGBdsh3 * dL_dRGB.y; dL_dsh_rest_accum[2][2] += dRGBdsh3 * dL_dRGB.z;
        }

        // 立即执行 Adam 更新
        for (int local_idx = 0; local_idx < 3; local_idx++)
        {
            int sh_idx = 0 + local_idx; // Degree 1 的绝对索引为 0~2
            for (int rgb_idx = 0; rgb_idx < 3; rgb_idx++)
            {
                float grad = dL_dsh_rest_accum[local_idx][rgb_idx];
                float& exp_avg = exp_avg_sh_rest[sh_idx][rgb_idx][source_chunk_id][index];
                float& exp_avg_sq = exp_avg_sq_sh_rest[sh_idx][rgb_idx][source_chunk_id][index];
                float& param = sh_rest[sh_idx][rgb_idx][source_chunk_id][index];

                exp_avg = b1 * exp_avg + (1.0f - b1) * grad;
                exp_avg_sq = b2 * exp_avg_sq + (1.0f - b2) * grad * grad;
                param += -sh_rest_lr * exp_avg / (sqrtf(exp_avg_sq) + eps);
            }
        }
    } // <-- Degree 1 的 9 个寄存器释放

    // ==========================================
    // Degree 2 (SH Rest indices 3~7)
    // ==========================================
    if (degree > 1)
    {
        // 局部累加数组，占用 15 个 float 寄存器 (与上面的 Degree 1 物理复用)
        float dL_dsh_rest_accum[5][3] = { 0.0f };

        for (int view_id = 0; view_id < views_count; view_id++)
        {
            float3 inv_trans{ -view_matrix[view_id][3][0], -view_matrix[view_id][3][1], -view_matrix[view_id][3][2] };
            float3 camera_center{
                inv_trans.x * view_matrix[view_id][0][0] + inv_trans.y * view_matrix[view_id][0][1] + inv_trans.z * view_matrix[view_id][0][2],
                inv_trans.x * view_matrix[view_id][1][0] + inv_trans.y * view_matrix[view_id][1][1] + inv_trans.z * view_matrix[view_id][1][2],
                inv_trans.x * view_matrix[view_id][2][0] + inv_trans.y * view_matrix[view_id][2][1] + inv_trans.z * view_matrix[view_id][2][2]
            };

            float3 dir{ pos.x - camera_center.x, pos.y - camera_center.y, pos.z - camera_center.z };
            float norm_recp = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-12f);
            float x = dir.x * norm_recp, y = dir.y * norm_recp, z = dir.z * norm_recp;

            float3 dL_dRGB{ color_grad[view_id][0][chunk_id][index], color_grad[view_id][1][chunk_id][index], color_grad[view_id][2][chunk_id][index] };

            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);

            dL_dsh_rest_accum[0][0] += dRGBdsh4 * dL_dRGB.x; dL_dsh_rest_accum[0][1] += dRGBdsh4 * dL_dRGB.y; dL_dsh_rest_accum[0][2] += dRGBdsh4 * dL_dRGB.z;
            dL_dsh_rest_accum[1][0] += dRGBdsh5 * dL_dRGB.x; dL_dsh_rest_accum[1][1] += dRGBdsh5 * dL_dRGB.y; dL_dsh_rest_accum[1][2] += dRGBdsh5 * dL_dRGB.z;
            dL_dsh_rest_accum[2][0] += dRGBdsh6 * dL_dRGB.x; dL_dsh_rest_accum[2][1] += dRGBdsh6 * dL_dRGB.y; dL_dsh_rest_accum[2][2] += dRGBdsh6 * dL_dRGB.z;
            dL_dsh_rest_accum[3][0] += dRGBdsh7 * dL_dRGB.x; dL_dsh_rest_accum[3][1] += dRGBdsh7 * dL_dRGB.y; dL_dsh_rest_accum[3][2] += dRGBdsh7 * dL_dRGB.z;
            dL_dsh_rest_accum[4][0] += dRGBdsh8 * dL_dRGB.x; dL_dsh_rest_accum[4][1] += dRGBdsh8 * dL_dRGB.y; dL_dsh_rest_accum[4][2] += dRGBdsh8 * dL_dRGB.z;
        }

        // 立即执行 Adam 更新
        for (int local_idx = 0; local_idx < 5; local_idx++)
        {
            int sh_idx = 3 + local_idx; // Degree 2 的绝对索引为 3~7
            for (int rgb_idx = 0; rgb_idx < 3; rgb_idx++)
            {
                float grad = dL_dsh_rest_accum[local_idx][rgb_idx];
                float& exp_avg = exp_avg_sh_rest[sh_idx][rgb_idx][source_chunk_id][index];
                float& exp_avg_sq = exp_avg_sq_sh_rest[sh_idx][rgb_idx][source_chunk_id][index];
                float& param = sh_rest[sh_idx][rgb_idx][source_chunk_id][index];

                exp_avg = b1 * exp_avg + (1.0f - b1) * grad;
                exp_avg_sq = b2 * exp_avg_sq + (1.0f - b2) * grad * grad;
                param += -sh_rest_lr * exp_avg / (sqrtf(exp_avg_sq) + eps);
            }
        }
    } // <-- Degree 2 寄存器释放

    // ==========================================
    // Degree 3 (SH Rest indices 8~14)
    // ==========================================
    if (degree > 2)
    {
        // 局部累加数组，占用 21 个 float 寄存器
        float dL_dsh_rest_accum[7][3] = { 0.0f };

        for (int view_id = 0; view_id < views_count; view_id++)
        {
            float3 inv_trans{ -view_matrix[view_id][3][0], -view_matrix[view_id][3][1], -view_matrix[view_id][3][2] };
            float3 camera_center{
                inv_trans.x * view_matrix[view_id][0][0] + inv_trans.y * view_matrix[view_id][0][1] + inv_trans.z * view_matrix[view_id][0][2],
                inv_trans.x * view_matrix[view_id][1][0] + inv_trans.y * view_matrix[view_id][1][1] + inv_trans.z * view_matrix[view_id][1][2],
                inv_trans.x * view_matrix[view_id][2][0] + inv_trans.y * view_matrix[view_id][2][1] + inv_trans.z * view_matrix[view_id][2][2]
            };

            float3 dir{ pos.x - camera_center.x, pos.y - camera_center.y, pos.z - camera_center.z };
            float norm_recp = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-12f);
            float x = dir.x * norm_recp, y = dir.y * norm_recp, z = dir.z * norm_recp;

            float3 dL_dRGB{ color_grad[view_id][0][chunk_id][index], color_grad[view_id][1][chunk_id][index], color_grad[view_id][2][chunk_id][index] };

            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y;

            float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
            float dRGBdsh10 = SH_C3[1] * xy * z;
            float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
            float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
            float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
            float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
            float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);

            dL_dsh_rest_accum[0][0] += dRGBdsh9 * dL_dRGB.x; dL_dsh_rest_accum[0][1] += dRGBdsh9 * dL_dRGB.y; dL_dsh_rest_accum[0][2] += dRGBdsh9 * dL_dRGB.z;
            dL_dsh_rest_accum[1][0] += dRGBdsh10 * dL_dRGB.x; dL_dsh_rest_accum[1][1] += dRGBdsh10 * dL_dRGB.y; dL_dsh_rest_accum[1][2] += dRGBdsh10 * dL_dRGB.z;
            dL_dsh_rest_accum[2][0] += dRGBdsh11 * dL_dRGB.x; dL_dsh_rest_accum[2][1] += dRGBdsh11 * dL_dRGB.y; dL_dsh_rest_accum[2][2] += dRGBdsh11 * dL_dRGB.z;
            dL_dsh_rest_accum[3][0] += dRGBdsh12 * dL_dRGB.x; dL_dsh_rest_accum[3][1] += dRGBdsh12 * dL_dRGB.y; dL_dsh_rest_accum[3][2] += dRGBdsh12 * dL_dRGB.z;
            dL_dsh_rest_accum[4][0] += dRGBdsh13 * dL_dRGB.x; dL_dsh_rest_accum[4][1] += dRGBdsh13 * dL_dRGB.y; dL_dsh_rest_accum[4][2] += dRGBdsh13 * dL_dRGB.z;
            dL_dsh_rest_accum[5][0] += dRGBdsh14 * dL_dRGB.x; dL_dsh_rest_accum[5][1] += dRGBdsh14 * dL_dRGB.y; dL_dsh_rest_accum[5][2] += dRGBdsh14 * dL_dRGB.z;
            dL_dsh_rest_accum[6][0] += dRGBdsh15 * dL_dRGB.x; dL_dsh_rest_accum[6][1] += dRGBdsh15 * dL_dRGB.y; dL_dsh_rest_accum[6][2] += dRGBdsh15 * dL_dRGB.z;
        }

        // 立即执行 Adam 更新
        for (int local_idx = 0; local_idx < 7; local_idx++)
        {
            int sh_idx = 8 + local_idx; // Degree 3 的绝对索引为 8~14
            for (int rgb_idx = 0; rgb_idx < 3; rgb_idx++)
            {
                float grad = dL_dsh_rest_accum[local_idx][rgb_idx];
                float& exp_avg = exp_avg_sh_rest[sh_idx][rgb_idx][source_chunk_id][index];
                float& exp_avg_sq = exp_avg_sq_sh_rest[sh_idx][rgb_idx][source_chunk_id][index];
                float& param = sh_rest[sh_idx][rgb_idx][source_chunk_id][index];

                exp_avg = b1 * exp_avg + (1.0f - b1) * grad;
                exp_avg_sq = b2 * exp_avg_sq + (1.0f - b2) * grad * grad;
                param += -sh_rest_lr * exp_avg / (sqrtf(exp_avg_sq) + eps);
            }
        }
    }
}

std::vector<at::Tensor> compact_sh_backward_adam(
    int sh_degree,
    at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
    at::Tensor view_matrix,
    at::Tensor position,
    at::Tensor sh_base, at::Tensor sh_rest,
    at::Tensor color_grad,
    at::Tensor exp_avg_sh_base, at::Tensor exp_avg_sq_sh_base,
    at::Tensor exp_avg_sh_rest, at::Tensor exp_avg_sq_sh_rest,
    float lr_sh_base,float lr_sh_rest, float b1, float b2, float eps
)
{
    int chunksize = position.size(2);
    int allocate_chunks_num = visible_chunk_id.size(0);

    switch (sh_degree)
    {
    case 0:
        compact_sh_backward_adam_kernel<0> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sq_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sq_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            lr_sh_base, lr_sh_rest, b1, b2, eps
        );
        break;
    case 1:
        compact_sh_backward_adam_kernel<1> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sq_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sq_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            lr_sh_base, lr_sh_rest, b1, b2, eps
        );
        break;
    case 2:
        compact_sh_backward_adam_kernel<2> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sq_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sq_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            lr_sh_base, lr_sh_rest, b1, b2, eps
        );
        break;
    case 3:
        compact_sh_backward_adam_kernel<3> <<<allocate_chunks_num, chunksize, 0>>> (
            visible_chunk_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            visible_chunks_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sq_sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            exp_avg_sq_sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            lr_sh_base, lr_sh_rest, b1, b2, eps
        );
        break;
    default:
        AT_ERROR("Unsupported sh_degree: ", sh_degree);
    }
    CUDA_CHECK_ERRORS;
    return { sh_base, sh_rest };
}