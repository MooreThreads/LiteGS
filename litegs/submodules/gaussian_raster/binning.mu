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

#ifndef __MUSACC__
    #define __MUSACC__
    #define __NVCC__
#endif
#include "musa_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <musa/atomic>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
namespace cg = cooperative_groups;

#include <ATen/core/TensorAccessor.h>

#include "cuda_errchk.h"
#include "binning.h"
#include "speedy_splat.cuh"

template<int TileSizeY, int TileSizeX>
 __global__ void duplicate_with_keys_kernel(
     const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> tensor_ndc,        //viewnum,4,pointnum
     const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> tensor_inv_cov2d,  //viewnum,2,2,pointnum
     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> tensor_opacity,  //viewnum,pointnum
     const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tensor_offset,        //viewnum,pointnum+1
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> depth_sorted_point_id,        //viewnum,pointnum
     int img_h, int img_w, unsigned int tile_num_h, unsigned int tile_num_w,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> tensor_key,//viewnum,pointnum
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> tensor_value//viewnum,pointnum
    )
{
     int view_id = blockIdx.y;
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     if (index < tensor_ndc.size(2))
     {
         int buffer_offset = index == 0 ? 0 : tensor_offset[view_id][index - 1];
         int allocated_size = tensor_offset[view_id][index] - buffer_offset;
         index = depth_sorted_point_id[view_id][index];

         float4 ndc{ tensor_ndc[view_id][0][index],tensor_ndc[view_id][1][index],
             tensor_ndc[view_id][2][index] ,tensor_ndc[view_id][3][index] };
         float opacity = max(tensor_opacity[view_id][index], 1.0f / 255);
         float4 con_o{ tensor_inv_cov2d[view_id][0][0][index],tensor_inv_cov2d[view_id][0][1][index],tensor_inv_cov2d[view_id][1][1][index],opacity };
         float disc = con_o.y * con_o.y - con_o.x * con_o.z;
         float2 screen_uv{ ndc.x * 0.5f + 0.5f,ndc.y * 0.5f + 0.5f };
         float2 p{ screen_uv.x * img_w - 0.5f,screen_uv.y * img_h - 0.5f };
         const dim3 grid{ tile_num_w,tile_num_h,0 };


         if (allocated_size>0)
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

             // Rectangular tile extent of ellipse
             int2 rect_min = {
                 max(0, min((int)grid.x, (int)(bbox_min.x / TileSizeX))),
                 max(0, min((int)grid.y, (int)(bbox_min.y / TileSizeY)))
             };
             int2 rect_max = {
                 max(0, min((int)grid.x, (int)((bbox_max.x + TileSizeX - 1) / TileSizeX))),
                 max(0, min((int)grid.y, (int)((bbox_max.y + TileSizeY - 1) / TileSizeY)))
             };

             int y_span = rect_max.y - rect_min.y;
             int x_span = rect_max.x - rect_min.x;
             if (y_span * x_span > 0)
             {
                 bool isY = y_span < x_span;
                 processTiles<TileSizeY, TileSizeX>(
                     con_o, disc, t, p,
                     bbox_min, bbox_max,
                     bbox_argmin, bbox_argmax,
                     rect_min, rect_max,
                     grid, isY,
                     index, buffer_offset,
                     &tensor_key[view_id][0],
                     &tensor_value[view_id][0]);
             }
         }
     }
}

 std::vector<at::Tensor> create_table(at::Tensor ndc, at::Tensor inv_cov2d, at::Tensor opacity, at::Tensor offset, at::Tensor depth_sorted_pointid,
     int64_t allocate_size, int64_t height, int64_t width, int64_t tile_size_h, int64_t tile_size_w)
{
    assert(tile_size_h == 8 && tile_size_w == 16);
    int tiles_num_h = (height + tile_size_h - 1) / tile_size_h;
    int tiles_num_w = (width + tile_size_w - 1) / tile_size_w;


    at::DeviceGuard guard(inv_cov2d.device());
    int64_t view_num = ndc.sizes()[0];
    int64_t points_num = ndc.sizes()[2];

    std::vector<int64_t> output_shape{ view_num, allocate_size };

    auto opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(ndc.device()).requires_grad(false);
    auto table_tileId = torch::empty(output_shape, opt);
    //auto table_tileId_sorted = torch::empty(output_shape, opt);
    opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(ndc.device()).requires_grad(false);
    auto table_pointId= torch::empty(output_shape, opt);
    //auto table_pointId_sorted = torch::empty(output_shape, opt);

    dim3 Block3d(std::ceil(points_num/256.0f), view_num, 1);
    

    duplicate_with_keys_kernel<8,16><<<Block3d,256>>>(
        ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        inv_cov2d.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        offset.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        depth_sorted_pointid.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        height,width, tiles_num_h, tiles_num_w,
        table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        table_pointId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return { table_tileId ,table_pointId };

    /*unsigned int bit = 0;
    unsigned int max_tiles = tiles_num_h * tiles_num_w;
    while (max_tiles >>= 1) bit++;
    bit++;

    size_t sort_tmp_buffer_size;
    cub::DeviceRadixSort::SortPairs<int,int>(
        nullptr,
        sort_tmp_buffer_size,
        (int*)table_tileId.data_ptr(), (int*)table_tileId_sorted.data_ptr(),
        (int*)table_pointId.data_ptr(), (int*)table_pointId_sorted.data_ptr(),
        allocate_size, 0, bit);
    auto temp_buffer_tensor=torch::empty({ ((int)sort_tmp_buffer_size + 4 - 1) / 4 }, opt);
    
    for (int view_id = 0; view_id < view_num; view_id++)
    {
        cub::DeviceRadixSort::SortPairs(
            temp_buffer_tensor.data_ptr(),
            sort_tmp_buffer_size,
            (int*)table_tileId.data_ptr(), (int*)table_tileId_sorted.data_ptr(),
            (int*)table_pointId.data_ptr(), (int*)table_pointId_sorted.data_ptr(),
            allocate_size, 0, bit);
    }

    return { table_tileId_sorted ,table_pointId_sorted };*/
    
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

template<int TileSizeY, int TileSizeX>
__global__ void get_allocate_size_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> tensor_ndc,        //viewnum,4,pointnum
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> view_space_z,        //viewnum,pointnum
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> tensor_inv_cov2d,  //viewnum,2,2,pointnum
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> tensor_opacity,  //viewnum,pointnum
    int img_h, int img_w, unsigned int tile_num_h, unsigned int tile_num_w,
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_left_up,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_right_down,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> tensor_allocated_size//viewnum,pointnum
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
        const dim3 grid{ tile_num_w,tile_num_h,0 };

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

            // Rectangular tile extent of ellipse
            int2 rect_min = {
                max(0, min((int)grid.x, (int)(bbox_min.x / TileSizeX))),
                max(0, min((int)grid.y, (int)(bbox_min.y / TileSizeY)))
            };
            int2 rect_max = {
                max(0, min((int)grid.x, (int)((bbox_max.x + TileSizeX - 1) / TileSizeX))),
                max(0, min((int)grid.y, (int)((bbox_max.y + TileSizeY - 1) / TileSizeY)))
            };

            int y_span = rect_max.y - rect_min.y;
            int x_span = rect_max.x - rect_min.x;
            int allocated_size = 0;
            if (y_span * x_span > 0)
            {
                bool isY = y_span < x_span;
                allocated_size = processTiles<TileSizeY, TileSizeX>(
                    con_o, disc, t, p,
                    bbox_min, bbox_max,
                    bbox_argmin, bbox_argmax,
                    rect_min, rect_max,
                    grid, isY,
                    index, 0,
                    nullptr,
                    nullptr);
            }
            tensor_allocated_size[view_id][index] = allocated_size;
              
        }
        else
        {
            tensor_left_up[view_id][0][index] = -1;
            tensor_left_up[view_id][1][index] = -1;
            tensor_right_down[view_id][0][index] = -1;
            tensor_right_down[view_id][1][index] = -1;
            tensor_allocated_size[view_id][index] = 0;
        }
    }
}

std::vector<at::Tensor> get_allocate_size(at::Tensor ndc, at::Tensor view_space_z, at::Tensor inv_cov2d, at::Tensor opacity,
    int64_t height,int64_t width, int64_t tile_size_h, int64_t tile_size_w)
{
    at::DeviceGuard guard(ndc.device());

    assert(tile_size_h == 8 && tile_size_w == 16);
    int tiles_num_h = (height + tile_size_h - 1) / tile_size_h;
    int tiles_num_w = (width + tile_size_w - 1) / tile_size_w;

    int views_num = ndc.size(0);
    int points_num = ndc.size(2);
    at::Tensor left_up = torch::empty({ views_num,2,points_num }, ndc.options().dtype(torch::kInt32));
    at::Tensor right_down = torch::empty({ views_num,2,points_num }, ndc.options().dtype(torch::kInt32));
    at::Tensor allocated_size = torch::empty({ views_num,points_num }, ndc.options().dtype(torch::kInt32));

    dim3 Block3d(std::ceil(points_num / 256.0f), views_num, 1);
    get_allocate_size_kernel<8,16><<<Block3d,256>>>(ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        view_space_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        inv_cov2d.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        height, width,tiles_num_h, tiles_num_w,
        left_up.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        right_down.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        allocated_size.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return { left_up ,right_down,allocated_size };
}
