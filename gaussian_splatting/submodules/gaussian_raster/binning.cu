#ifndef __CUDACC__
    #define __CUDACC__
    #define __NVCC__
#endif
#include "cuda_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
namespace cg = cooperative_groups;

#include <c10/cuda/CUDAException.h>
#include <ATen/core/TensorAccessor.h>

#include "cuda_errchk.h"
#include "binning.h"

 __global__ void duplicate_with_keys_kernel(
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> LU,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> RD,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> prefix_sum,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> depth_sorted_pointid,//viewnum,pointnum
    int TileSizeX,
    torch::PackedTensorAccessor32 < int16_t, 2, torch::RestrictPtrTraits> table_tileId,
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
        if ((r - l) * (d - u) < 32)
        {
            for (int i = u; i < d; i++)
            {
                for (int j = l; j < r; j++)
                {
                    int tile_id = i * TileSizeX + j;
                    table_tileId[view_id][end - 1 - count] = tile_id + 1;// tile_id 0 means invalid!
                    table_pointId[view_id][end - 1 - count] = point_id;
                    count++;
                }
            }
        }
    }
}

 __global__ void large_points_duplicate_with_keys_kernel(
     const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> LU,//viewnum,2,pointnum
     const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> RD,//viewnum,2,pointnum
     const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> prefix_sum,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> depth_sorted_pointid,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> large_index,//tasknum,2
     int TileSizeX,
     torch::PackedTensorAccessor32 < int16_t, 2, torch::RestrictPtrTraits> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_pointId
 )
 {
     auto block = cg::this_thread_block();
     auto warp = cg::tiled_partition<32>(block);
     int task_index = warp.meta_group_size() * block.group_index().x + warp.meta_group_rank();

     if (task_index < large_index.size(0))
     {
         int view_id = large_index[task_index][0];
         int point_index = large_index[task_index][1];
         int point_id = depth_sorted_pointid[view_id][point_index];
         int end = prefix_sum[view_id][point_index];

         //int end = prefix_sum[view_id][point_id+1];
         int l = LU[view_id][0][point_id];
         int u = LU[view_id][1][point_id];
         int width = RD[view_id][0][point_id]-l;
         int height = RD[view_id][1][point_id]-u;
         for (int i = warp.thread_rank(); i < width * height; i+=warp.num_threads())
         {
             int col = l + (i % width);
             int row = u + (i / width);
             int tile_id = row * TileSizeX + col;
             table_tileId[view_id][end - 1 - i] = tile_id + 1;// tile_id 0 means invalid!
             table_pointId[view_id][end - 1 - i] = point_id;
         }
     }
 }

std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU, at::Tensor RD, at::Tensor prefix_sum, at::Tensor depth_sorted_pointid,
    at::Tensor large_index,int64_t allocate_size, int64_t TilesSizeX)
{
    at::DeviceGuard guard(LU.device());
    int64_t view_num = LU.sizes()[0];
    int64_t points_num = LU.sizes()[2];

    std::vector<int64_t> output_shape{ view_num, allocate_size };

    auto opt = torch::TensorOptions().dtype(torch::kInt16).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_tileId = torch::zeros(output_shape, opt);
    opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_pointId= torch::zeros(output_shape, opt);

    dim3 Block3d(std::ceil(points_num/1024.0f), view_num, 1);
    

    duplicate_with_keys_kernel<<<Block3d ,1024>>>(
        LU.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        RD.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        prefix_sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        depth_sorted_pointid.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        TilesSizeX,
        table_tileId.packed_accessor32<int16_t, 2, torch::RestrictPtrTraits>(),
        table_pointId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    
    int large_points_num = large_index.size(0);
    int blocksnum = std::ceil((large_points_num * 32) / 1024.0f);
    large_points_duplicate_with_keys_kernel << <blocksnum, 1024 >> > (
        LU.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        RD.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        prefix_sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        depth_sorted_pointid.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        large_index.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        TilesSizeX,
        table_tileId.packed_accessor32<int16_t, 2, torch::RestrictPtrTraits>(),
        table_pointId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return { table_tileId ,table_pointId };
    
}

__global__ void tile_range_kernel(
    const torch::PackedTensorAccessor32<int16_t, 2,torch::RestrictPtrTraits> table_tileId,//viewnum,pointnum
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

    dim3 Block3d(std::ceil(table_length / 1024.0f), view_num, 1);

    tile_range_kernel<<<Block3d, 1024 >>>
        (table_tileId.packed_accessor32<int16_t, 2, torch::RestrictPtrTraits>(), table_length, max_tileId, out.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return out;
}


