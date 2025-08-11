/*
Portions of this code are derived from the project "speedy-splat"
(https://github.com/j-alex-hanson/speedy-splat), which is based on
"gaussian-splatting" developed by Inria and the Max Planck Institute for Informatik (MPII).

Original work © Inria and MPII.
Licensed under the Gaussian-Splatting License.
You may use, reproduce, and distribute this work and its derivatives for
**non-commercial research and evaluation purposes only**, subject to the terms
and conditions of the Gaussian-Splatting License.

A copy of the Gaussian-Splatting License is provided in the LICENSE file.
*/

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
#include "binning.h"
#include "speedy_splat.cuh"

 __global__ void duplicate_with_keys_kernel(
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> LU,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> RD,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> prefix_sum,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> depth_sorted_pointid,//viewnum,pointnum
    int TilesNumX,
    torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_pointId
    )
{
    int view_id = blockIdx.y;
    

    if (blockIdx.x * blockDim.x + threadIdx.x < prefix_sum.size(1))
    {
        int point_id = depth_sorted_pointid[view_id][blockIdx.x * blockDim.x + threadIdx.x];
        int end = prefix_sum[view_id][blockIdx.x * blockDim.x + threadIdx.x];

        //int end = prefix_sum[view_id][point_id+1];
        int l = LU[view_id][0][point_id];
        int u = LU[view_id][1][point_id];
        int r = RD[view_id][0][point_id];
        int d = RD[view_id][1][point_id];
        int count = 0;
        //if ((r - l) * (d - u) < 32)
        {
            for (int i = u; i < d; i++)
            {
                for (int j = l; j < r; j++)
                {
                    int tile_id = i * TilesNumX + j;
                    table_tileId[view_id][end - 1 - count] = tile_id + 1;// tile_id 0 means invalid!
                    table_pointId[view_id][end - 1 - count] = point_id;
                    count++;
                }
            }
        }
    }
}

 std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU, at::Tensor RD, at::Tensor prefix_sum, at::Tensor depth_sorted_pointid,
     int64_t allocate_size, int64_t img_h, int64_t img_w, int64_t tilesize_h,int64_t tilesize_w)
{
    at::DeviceGuard guard(LU.device());
    int64_t view_num = LU.sizes()[0];
    int64_t points_num = LU.sizes()[2];

    std::vector<int64_t> output_shape{ view_num, allocate_size };

    auto opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_tileId = torch::zeros(output_shape, opt);
    opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_pointId= torch::zeros(output_shape, opt);

    dim3 Block3d(std::ceil(points_num/512.0f), view_num, 1);
    

    duplicate_with_keys_kernel<<<Block3d ,512>>>(
        LU.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        RD.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        prefix_sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        depth_sorted_pointid.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        int(std::ceil(img_w/(float)tilesize_w)),
        table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        table_pointId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return { table_tileId ,table_pointId };
    
}

__global__ void tile_range_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> table_tileId,//viewnum,pointnum
    int table_length,
    int max_tileId,
    torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> tile_range
)
{
    int view_id = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // head
    if (index == 0)
    {
        int tile_id=table_tileId[view_id][index];
        tile_range[view_id][tile_id] = index;
    }
    
    //tail
    if (index == table_length - 1)
    {
        tile_range[view_id][max_tileId + 1] = table_length;
    }
    
    if (index < table_length-1)
    {
        int cur_tile = table_tileId[view_id][index];
        int next_tile= table_tileId[view_id][index+1];
        if (cur_tile!=next_tile)
        {
            if (cur_tile + 1 < next_tile)
            {
                tile_range[view_id][cur_tile + 1] = index + 1;
            }
            tile_range[view_id][next_tile] = index + 1;
        }
    }
}

at::Tensor tileRange(at::Tensor table_tileId, int64_t table_length, int64_t max_tileId)
{
    at::DeviceGuard guard(table_tileId.device());

    int64_t view_num = table_tileId.sizes()[0];
    std::vector<int64_t> output_shape{ view_num,max_tileId + 1 + 1 };//+1 for tail
    //printf("\ntensor shape in tileRange:%ld,%ld\n", view_num, max_tileId+1-1);
    auto opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(table_tileId.device()).requires_grad(false);
    auto out = torch::ones(output_shape, opt)*-1;

    dim3 Block3d(std::ceil(table_length / 512.0f), view_num, 1);

    tile_range_kernel<<<Block3d, 512 >>>
        (table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(), table_length, max_tileId, out.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return out;
}

__global__ void create_ROI_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> tensor_ndc,        //viewnum,4,pointnum
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> view_space_z,        //viewnum,pointnum
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> tensor_inv_cov2d,  //viewnum,2,2,pointnum
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> tensor_opacity,  //viewnum,pointnum
    int img_h,int img_w,
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_left_up,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_right_down//viewnum,2,pointnum
)
{
    //speedy splat https://github.com/j-alex-hanson/speedy-splat

    int view_id = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < tensor_ndc.size(2))
    {
        float4 ndc{ tensor_ndc[view_id][0][index],tensor_ndc[view_id][1][index],
            tensor_ndc[view_id][2][index] ,tensor_ndc[view_id][3][index] };
        float opacity = max(tensor_opacity[view_id][index], 1.0f / 255);
        float4 con_o{ tensor_inv_cov2d[view_id][0][0][index],tensor_inv_cov2d[view_id][0][1][index],tensor_inv_cov2d[view_id][1][1][index],opacity };
        float disc = con_o.y * con_o.y - con_o.x * con_o.z;
        float2 screen_uv{ ndc.x * 0.5f + 0.5f,ndc.y * 0.5f + 0.5f };
        float2 p{ screen_uv.x * img_w - 0.5f,screen_uv.y * img_h - 0.5f };

        bool bVisible = !((ndc.x < -1.3f) || (ndc.x > 1.3f) || (ndc.y < -1.3f) || (ndc.y > 1.3f) || (view_space_z[view_id][index] <= 0.2f));
        bVisible &= ((con_o.x > 0)& (con_o.z > 0)& (disc < 0));

        if (bVisible)
        {
            float t = 2.0f * log(con_o.w * 255.0f);
            float x_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.x));
            x_term = (con_o.y < 0) ? x_term : -x_term;
            float y_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.z));
            y_term = (con_o.y < 0) ? y_term : -y_term;

            float2 bbox_argmin = { p.y - y_term, p.x - x_term };
            float2 bbox_argmax = { p.y + y_term, p.x + x_term };
            
            float2 bbox_min = {
                computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmin.x).x,
                computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmin.y).x
            };
            float2 bbox_max = {
                computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmax.x).y,
                computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmax.y).y
            };


            tensor_left_up[view_id][0][index] = std::ceil(bbox_min.x);
            tensor_left_up[view_id][1][index] = std::ceil(bbox_min.y);
            tensor_right_down[view_id][0][index] = std::floor(bbox_max.x);
            tensor_right_down[view_id][1][index] = std::floor(bbox_max.y);
              
        }
        else
        {
            tensor_left_up[view_id][0][index] = -1;
            tensor_left_up[view_id][1][index] = -1;
            tensor_right_down[view_id][0][index] = -1;
            tensor_right_down[view_id][1][index] = -1;
        }
    }
}

std::vector<at::Tensor> create_2d_gaussian_ROI(at::Tensor ndc, at::Tensor view_space_z, at::Tensor inv_cov2d, at::Tensor opacity,int64_t height,int64_t width)
{
    at::DeviceGuard guard(ndc.device());

    int views_num = ndc.size(0);
    int points_num = ndc.size(2);
    at::Tensor left_up = torch::empty({ views_num,2,points_num }, ndc.options().dtype(torch::kInt32));
    at::Tensor right_down = torch::empty({ views_num,2,points_num }, ndc.options().dtype(torch::kInt32));

    dim3 Block3d(std::ceil(points_num / 256.0f), views_num, 1);
    create_ROI_kernel<<<Block3d,256>>>(ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        view_space_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        inv_cov2d.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        height, width,
        left_up.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        right_down.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return { left_up ,right_down };
}

template<int tileH,int tileW>
__global__ void get_allocate_size_kernel(
    const torch::PackedTensorAccessor32< int32_t, 3, torch::RestrictPtrTraits> tensor_pixel_left_up,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32< int32_t, 3, torch::RestrictPtrTraits> tensor_pixel_right_down,//viewnum,2,pointnum
    int max_tileid_y,int max_tileid_x,
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_tile_left_up,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_tile_right_down,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> tensor_allocate//viewnum,pointnum
)
{
    int view_id = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < tensor_pixel_left_up.size(2))
    {
        int pixel_left = tensor_pixel_left_up[view_id][0][index];
        int pixel_up = tensor_pixel_left_up[view_id][1][index];
        int pixel_right = tensor_pixel_right_down[view_id][0][index];
        int pixel_down = tensor_pixel_right_down[view_id][1][index];

        int tile_left = min(max(0, pixel_left / tileW),max_tileid_x);
        int tile_right = min(max(0, (pixel_right + tileW - 1) / tileW), max_tileid_x);
        int tile_up = min(max(0, pixel_up / tileH), max_tileid_y);
        int tile_down = min(max(0, (pixel_down + tileH - 1) / tileH), max_tileid_y);

        tensor_tile_left_up[view_id][0][index] = tile_left;
        tensor_tile_left_up[view_id][1][index] = tile_up;
        tensor_tile_right_down[view_id][0][index] = tile_right;
        tensor_tile_right_down[view_id][1][index] = tile_down;

        int allocate_size = (tile_right - tile_left) * (tile_down - tile_up);
        tensor_allocate[view_id][index] = allocate_size;
    }
}

std::vector<at::Tensor> get_allocate_size(at::Tensor pixel_left_up, at::Tensor pixel_right_down, int64_t tilesize_h, int64_t tilesize_w,
    int64_t max_tileid_y, int64_t max_tileid_x)
{
    at::DeviceGuard guard(pixel_left_up.device());
    assert((tilesize_h == 8) && (tilesize_w == 16));

    int views_num = pixel_left_up.size(0);
    int points_num = pixel_left_up.size(2);
    at::Tensor tile_left_up = torch::empty({ views_num,2,points_num }, pixel_left_up.options());
    at::Tensor tile_right_down = torch::empty({ views_num,2,points_num }, pixel_left_up.options());
    at::Tensor allocate_size = torch::empty({ views_num,points_num }, pixel_left_up.options());

    dim3 Block3d(std::ceil(points_num / 256.0f), views_num, 1);
    get_allocate_size_kernel<8,16> << <Block3d, 256 >> > (
        pixel_left_up.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        pixel_right_down.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        max_tileid_y, max_tileid_x,
        tile_left_up.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        tile_right_down.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        allocate_size.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
    return { tile_left_up,tile_right_down,allocate_size };
}