#include "raster_binning.h"
#include <c10/cuda/CUDAException.h>
#include "cuda_runtime.h"

#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

 __global__ void duplicate_with_keys_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2/*torch::RestrictPtrTraits*/> L,//viewnum,pointnum
    const torch::PackedTensorAccessor32<int32_t, 2/*torch::RestrictPtrTraits*/> U,
    const torch::PackedTensorAccessor32<int32_t, 2/*torch::RestrictPtrTraits*/> R,
    const torch::PackedTensorAccessor32<int32_t, 2/*torch::RestrictPtrTraits*/> D,
    const torch::PackedTensorAccessor32<int64_t, 1/*torch::RestrictPtrTraits*/> valid_points_num,//view
    const torch::PackedTensorAccessor32<int64_t, 2/*torch::RestrictPtrTraits*/> prefix_sum,//view,pointnum
    int TileSizeX,
    torch::PackedTensorAccessor32 < int16_t, 2> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2> table_pointId
    )
{
    int view_id = blockIdx.y;
    int point_id = blockIdx.x*blockDim.x + threadIdx.x;

    int valid_points_num_in_view = valid_points_num[view_id];
    if (point_id < valid_points_num_in_view)
    {
        int end = prefix_sum[view_id][point_id];
        //int end = prefix_sum[view_id][point_id+1];
        int l = L[view_id][point_id];
        int u = U[view_id][point_id];
        int r = R[view_id][point_id];
        int d = D[view_id][point_id];
        int count = 0;

        for (int i = u; i <= d; i++)
        {
            for (int j = l; j <= r; j++)
            {
                int tile_id = i * TileSizeX + j;
                table_tileId[view_id][end - 1 - count] = tile_id+1;// tile_id 0 means invalid!
                table_pointId[view_id][end - 1 - count] = point_id;
                count++;
            }
        }
    }


}



std::vector<Tensor> duplicateWithKeys(Tensor L,Tensor U,Tensor R,Tensor D,Tensor ValidPointNum,Tensor prefix_sum, int64_t allocate_size, int64_t TilesSizeX)
{
    int64_t view_num = L.sizes()[0];
    int64_t points_num = L.sizes()[1];
    at::IntArrayRef output_shape(std::vector<int64_t>{ view_num, allocate_size });
    //printf("\ntensor shape in duplicateWithKeys:%ld,%ld\n", view_num, allocate_size);
    auto opt = torch::TensorOptions().dtype(torch::kInt16).layout(torch::kStrided).device(L.device()).requires_grad(false);
    auto table_tileId = torch::zeros(output_shape, opt);
    opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(L.device()).requires_grad(false);
    auto table_pointId= torch::zeros(output_shape, opt);

    dim3 BlocksPerGrid(std::ceil(points_num/1024.0f), view_num, 1);
    

    duplicate_with_keys_kernel<<<BlocksPerGrid ,1024>>>(
        L.packed_accessor32<int32_t, 2>(),
        U.packed_accessor32<int32_t, 2>(),
        R.packed_accessor32<int32_t, 2>(),
        D.packed_accessor32<int32_t, 2>(),
        ValidPointNum.packed_accessor32<int64_t, 1>(),
        prefix_sum.packed_accessor32<int64_t, 2>(),
        TilesSizeX,
        table_tileId.packed_accessor32<int16_t, 2>(),
        table_pointId.packed_accessor32<int32_t, 2>());
    CUDA_CHECK_ERRORS;
    

    return { table_tileId ,table_pointId };
    
}

__global__ void tile_range_kernel(
    const torch::PackedTensorAccessor32<int16_t, 2/*torch::RestrictPtrTraits*/> table_tileId,//viewnum,pointnum
    int table_length,
    int max_tileId,
    torch::PackedTensorAccessor32 < int32_t, 2> tile_range
)
{
    int view_id = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // skip head because tileid 0 is invalid
    /*if (index == 0)
    {
        tile_range[view_id][0] = index;
    }*/
    
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
            tile_range[view_id][next_tile] = index + 1;
        }
    }
}

Tensor tileRange(Tensor table_tileId, int64_t table_length, int64_t max_tileId)
{
    int64_t view_num = table_tileId.sizes()[0];
    at::IntArrayRef output_shape(std::vector<int64_t> { view_num,max_tileId+1+1 });//+1 for tail
    //printf("\ntensor shape in tileRange:%ld,%ld\n", view_num, max_tileId+1-1);
    auto opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(table_tileId.device()).requires_grad(false);
    auto out = torch::zeros(output_shape, opt);

    dim3 BlocksPerGrid(std::ceil(table_length / 1024.0f), view_num, 1);

    tile_range_kernel<<<BlocksPerGrid, 1024 >>>
        (table_tileId.packed_accessor32<int16_t, 2>(), table_length, max_tileId, out.packed_accessor32<int32_t, 2>());
    CUDA_CHECK_ERRORS;

    return out;
}