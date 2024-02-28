#ifndef __CUDACC__
    #define __CUDACC__
#endif
#include "cuda_runtime.h"
#include <c10/cuda/CUDAException.h>
#include "raster_binning.h"
#include <ATen/core/TensorAccessor.h>


#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

 __global__ void duplicate_with_keys_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> L,//viewnum,pointnum
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> U,
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> R,
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> D,
    const torch::PackedTensorAccessor32<int64_t, 1,torch::RestrictPtrTraits> valid_points_num,//view
    const torch::PackedTensorAccessor32<int64_t, 2,torch::RestrictPtrTraits> prefix_sum,//view,pointnum
    int TileSizeX,
    torch::PackedTensorAccessor32 < int16_t, 2, torch::RestrictPtrTraits> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_pointId
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

        for (int i = u; i < d; i++)
        {
            for (int j = l; j < r; j++)
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
    DeviceGuard guard(L.device());
    int64_t view_num = L.sizes()[0];
    int64_t points_num = L.sizes()[1];

    std::vector<int64_t> output_shape{ view_num, allocate_size };

    auto opt = torch::TensorOptions().dtype(torch::kInt16).layout(torch::kStrided).device(L.device()).requires_grad(false);
    auto table_tileId = torch::zeros(output_shape, opt);
    opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(L.device()).requires_grad(false);
    auto table_pointId= torch::zeros(output_shape, opt);

    dim3 Block3d(std::ceil(points_num/1024.0f), view_num, 1);
    

    duplicate_with_keys_kernel<<<Block3d ,1024>>>(
        L.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        U.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        R.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        D.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        ValidPointNum.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        prefix_sum.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
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
    DeviceGuard guard(table_tileId.device());

    int64_t view_num = table_tileId.sizes()[0];
    std::vector<int64_t> output_shape{ view_num,max_tileId + 1 + 1 };//+1 for tail
    //printf("\ntensor shape in tileRange:%ld,%ld\n", view_num, max_tileId+1-1);
    auto opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(table_tileId.device()).requires_grad(false);
    auto out = torch::zeros(output_shape, opt);

    dim3 Block3d(std::ceil(table_length / 1024.0f), view_num, 1);

    tile_range_kernel<<<Block3d, 1024 >>>
        (table_tileId.packed_accessor32<int16_t, 2, torch::RestrictPtrTraits>(), table_length, max_tileId, out.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return out;
}



template <int tilesize>
__global__ void rasterize_gathered_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> mean2d,         //[batch,point_num,2]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,point_num,2,2]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,point_num,3]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,          //[batch,point_num,1]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,tile,tilesize,tilesize,3]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output_transmitance,    //[batch,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor    //[batch,tile,tilesize,tilesize]
)
{


    __shared__ float2 collected_xy[tilesize* tilesize];
    __shared__ float collected_opacity[tilesize * tilesize];
    __shared__ float3 collected_cov2d_inv[tilesize* tilesize];
    __shared__ float3 collected_color[tilesize * tilesize];

    const int batch_id = blockIdx.y;
    const int tile_id = blockIdx.x+1;

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    if (tile_id < start_index.size(1)-1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        float transmittance = 1.0f;
        bool done = false;
        float3 final_color{ 0,0,0 };
        int last_contributor = 0;
        for (int offset = start_index_in_tile; offset < end_index_in_tile; offset += tilesize* tilesize)
        {
            int num_done = __syncthreads_count(done);
            if (num_done == blockDim.x*blockDim.y)
                break;

            int valid_num = min(tilesize * tilesize, end_index_in_tile - offset);
            //load to shared memory
            for (int i = 0; i < valid_num; i++)
            {
                int index = offset + i;
                collected_xy[i].x = mean2d[batch_id][index][0];
                collected_xy[i].y = mean2d[batch_id][index][1];
                collected_cov2d_inv[i].x = cov2d_inv[batch_id][index][0][0];
                collected_cov2d_inv[i].y = cov2d_inv[batch_id][index][0][1];
                collected_cov2d_inv[i].z = cov2d_inv[batch_id][index][1][1];

                collected_color[i].x = color[batch_id][index][0];
                collected_color[i].y = color[batch_id][index][1];
                collected_color[i].z = color[batch_id][index][2];
                collected_opacity[i]=opacity[batch_id][index][0];
            }
            __syncthreads();

            //process
            for (int i = 0; i < valid_num && done==false; i++)
            {

                float2 xy = collected_xy[i];
                float2 d = { xy.x - x_in_tile,xy.y - y_in_tile };
                float3 cur_color = collected_color[i];
                float cur_opacity = collected_opacity[i];
                float3 cur_cov2d_inv = collected_cov2d_inv[i];

                float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                if (power > 0.0f)
                    continue;

                float alpha = min(0.99f, cur_opacity * exp(power));
                if (alpha < 1.0f / 255.0f)
                    continue;

                final_color.x += cur_color.x * alpha * transmittance;
                final_color.y += cur_color.y * alpha * transmittance;
                final_color.z += cur_color.z * alpha * transmittance;
                transmittance *= (1 - alpha);
                last_contributor = offset + i;
                if (transmittance < 1e-4f)
                {
                    done = true;
                }

            }
            __syncthreads();
        }

        output_img[batch_id][tile_id][y_in_tile][x_in_tile][0] = final_color.x;
        output_img[batch_id][tile_id][y_in_tile][x_in_tile][1] = final_color.y;
        output_img[batch_id][tile_id][y_in_tile][x_in_tile][2] = final_color.z;

        output_last_contributor[batch_id][tile_id][y_in_tile][x_in_tile] = last_contributor;
        output_transmitance[batch_id][tile_id][y_in_tile][x_in_tile] = transmittance;
    }
}

std::vector<Tensor> rasterize_gathered_forward(
    Tensor start_index,
    Tensor gathered_mean2d,// 
    Tensor gathered_cov2d_inv,
    Tensor gathered_color,
    Tensor gathered_opacity,
    int64_t tilesize,
    int64_t tilesnum
)
{
    DeviceGuard guard(gathered_mean2d.device());

    int64_t viewsnum = start_index.sizes()[0];

    std::vector<int64_t> shape_img{ viewsnum, tilesnum+1,tilesize,tilesize,3 };
    torch::TensorOptions opt_img=torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    Tensor output_img = torch::zeros(shape_img, opt_img);

    std::vector<int64_t> shape_t{ viewsnum, tilesnum + 1 , tilesize, tilesize };
    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    Tensor output_transmitance = torch::zeros(shape_t, opt_t);

    std::vector<int64_t> shape_c{ viewsnum, tilesnum + 1, tilesize, tilesize };
    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(start_index.device()).requires_grad(false);
    Tensor output_last_contributor = torch::zeros(shape_c, opt_c);



    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);
    if (tilesize == 16)
    {
        rasterize_gathered_forward_kernel<16> << <Block3d, Thread3d >> >
            (
                start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                gathered_mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                gathered_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                gathered_color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                gathered_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>()
            );
    }
    else if(tilesize == 32)
    {
        rasterize_gathered_forward_kernel<32> << <Block3d, Thread3d >> >
            (
                start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                gathered_mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                gathered_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                gathered_color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                gathered_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>()
                );
    }


    return { output_img ,output_transmitance ,output_last_contributor };
}

std::vector<Tensor> rasterize_gathered_backward(Tensor arg)
{
    DeviceGuard guard(arg.device());
    return {};
}




template <int tilesize>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> mean2d,         //[batch,point_num,2]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,point_num,2,2]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,point_num,3]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,          //[batch,point_num,1]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,tile,tilesize,tilesize,3]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output_transmitance,    //[batch,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    int tiles_num_x
)
{


    __shared__ float2 collected_xy[tilesize * tilesize];
    __shared__ float collected_opacity[tilesize * tilesize];
    __shared__ float3 collected_cov2d_inv[tilesize * tilesize];
    __shared__ float3 collected_color[tilesize * tilesize];

    const int batch_id = blockIdx.y;
    const int tile_id = blockIdx.x + 1;

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id-1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id) / tiles_num_x) * tilesize + y_in_tile;

    if (tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        float transmittance = 1.0f;
        bool done = false;
        float3 final_color{ 0,0,0 };
        int last_contributor = 0;
        for (int offset = start_index_in_tile; offset < end_index_in_tile; offset += tilesize * tilesize)
        {
            int num_done = __syncthreads_count(done);
            if (num_done == blockDim.x * blockDim.y)
                break;

            int valid_num = min(tilesize * tilesize, end_index_in_tile - offset);
            //load to shared memory
            if (threadIdx.y * blockDim.x + threadIdx.x < valid_num)
            {
                int i=threadIdx.y * blockDim.x + threadIdx.x;
                int index = offset + i;
                int point_id = sorted_points[batch_id][index];
                collected_xy[i].x = mean2d[batch_id][point_id][0];
                collected_xy[i].y = mean2d[batch_id][point_id][1];
                collected_cov2d_inv[i].x = cov2d_inv[batch_id][point_id][0][0];
                collected_cov2d_inv[i].y = cov2d_inv[batch_id][point_id][0][1];
                collected_cov2d_inv[i].z = cov2d_inv[batch_id][point_id][1][1];

                collected_color[i].x = color[batch_id][point_id][0];
                collected_color[i].y = color[batch_id][point_id][1];
                collected_color[i].z = color[batch_id][point_id][2];
                collected_opacity[i] = opacity[batch_id][point_id][0];
            }
            __syncthreads();

            //process
            for (int i = 0; i < valid_num && done == false; i++)
            {

                float2 xy = collected_xy[i];
                float2 d = { xy.x - pixel_x,xy.y - pixel_y };
                float3 cur_color = collected_color[i];
                float cur_opacity = collected_opacity[i];
                float3 cur_cov2d_inv = collected_cov2d_inv[i];

                float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                if (power > 0.0f)
                    continue;

                float alpha = min(0.99f, cur_opacity * exp(power));
                if (alpha < 1.0f / 255.0f)
                    continue;

                if (transmittance*(1-alpha) < 0.0001f)
                {
                    done = true;
                    continue;
                }

                final_color.x += cur_color.x * alpha * transmittance;
                final_color.y += cur_color.y * alpha * transmittance;
                final_color.z += cur_color.z * alpha * transmittance;
                transmittance *= (1 - alpha);
                last_contributor = offset + i;
                /*if (pixel_x == 651 && pixel_y == 835)
                {
                    printf("\nforward:%e %e ", transmittance, alpha);
                }*/


            }
            __syncthreads();
        }

        output_img[batch_id][tile_id][y_in_tile][x_in_tile][0] = final_color.x;
        output_img[batch_id][tile_id][y_in_tile][x_in_tile][1] = final_color.y;
        output_img[batch_id][tile_id][y_in_tile][x_in_tile][2] = final_color.z;

        output_last_contributor[batch_id][tile_id][y_in_tile][x_in_tile] = last_contributor;
        output_transmitance[batch_id][tile_id][y_in_tile][x_in_tile] = transmittance;
    }
}

std::vector<Tensor> rasterize_forward(
    Tensor sorted_points,
    Tensor start_index,
    Tensor  mean2d,// 
    Tensor  cov2d_inv,
    Tensor  color,
    Tensor  opacity,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y
)
{
    DeviceGuard guard( mean2d.device());

    int64_t viewsnum = start_index.sizes()[0];
    int64_t tilesnum = tilesnum_x * tilesnum_y;

    std::vector<int64_t> shape_img{ viewsnum, tilesnum + 1,tilesize,tilesize,3 };
    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    Tensor output_img = torch::zeros(shape_img, opt_img);

    std::vector<int64_t> shape_t{ viewsnum, tilesnum + 1 , tilesize, tilesize };
    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    Tensor output_transmitance = torch::zeros(shape_t, opt_t);

    std::vector<int64_t> shape_c{ viewsnum, tilesnum + 1, tilesize, tilesize };
    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(start_index.device()).requires_grad(false);
    Tensor output_last_contributor = torch::zeros(shape_c, opt_c);



    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);
    if (tilesize == 16)
    {
        raster_forward_kernel<16> << <Block3d, Thread3d >> >
            (
                sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                 mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                 cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                 color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                 opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                tilesnum_x
                );
        CUDA_CHECK_ERRORS;
    }
    else if (tilesize == 32)
    {
        raster_forward_kernel<32> << <Block3d, Thread3d >> >
            (
                sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                 mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                 cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                 color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                 opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                tilesnum_x
                );
        CUDA_CHECK_ERRORS;
    }


    return { output_img ,output_transmitance ,output_last_contributor };
}
#define REDUTCION_BACKWARD
#ifdef REDUTCION_BACKWARD
template <int tilesize>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> mean2d,         //[batch,point_num,2]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,point_num,2,2]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,point_num,3]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,          //[batch,point_num,1]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> final_transmitance,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,tile,tilesize,tilesize,3]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_mean2d,         //[batch,point_num,2]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,point_num,2,2]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,point_num,3]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_opacity,          //[batch,point_num,1]
    int tiles_num_x
)
{


    /*__shared__ float2 collected_xy[tilesize * tilesize];
    __shared__ float collected_opacity[tilesize * tilesize];
    __shared__ float3 collected_cov2d_inv[tilesize * tilesize];
    __shared__ float3 collected_color[tilesize * tilesize];
    __shared__ int collected_point_id[tilesize * tilesize];*/
    __shared__ int collected_point_id[tilesize * tilesize / 4];
    __shared__ float collected_color_x[tilesize * tilesize / 4];
    __shared__ float collected_color_y[tilesize * tilesize / 4];
    __shared__ float collected_color_z[tilesize * tilesize / 4];
    __shared__ float collected_invcov_x[tilesize * tilesize / 4];
    __shared__ float collected_invcov_y[tilesize * tilesize / 4];
    __shared__ float collected_invcov_z[tilesize * tilesize / 4];
    __shared__ float collected_mean_x[tilesize * tilesize / 4];
    __shared__ float collected_mean_y[tilesize * tilesize / 4];
    __shared__ float collected_opacity[tilesize * tilesize / 4];

    __shared__ float grad_color_x[tilesize * tilesize];
    __shared__ float grad_color_y[tilesize * tilesize];
    __shared__ float grad_color_z[tilesize * tilesize];
    __shared__ float grad_invcov_x[tilesize * tilesize];
    __shared__ float grad_invcov_y[tilesize * tilesize];
    __shared__ float grad_invcov_z[tilesize * tilesize];
    __shared__ float grad_mean_x[tilesize * tilesize];
    __shared__ float grad_mean_y[tilesize * tilesize];
    __shared__ float grad_opacity[tilesize * tilesize];



    const int batch_id = blockIdx.y;
    const int tile_id = blockIdx.x + 1;

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id) / tiles_num_x) * tilesize + y_in_tile;

    if (tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        float transmittance = final_transmitance[batch_id][tile_id][y_in_tile][x_in_tile];
        int pixel_lst_index = last_contributor[batch_id][tile_id][y_in_tile][x_in_tile];
        float3 d_pixel{ 0,0,0 };
        d_pixel.x = d_img[batch_id][tile_id][y_in_tile][x_in_tile][0];
        d_pixel.y = d_img[batch_id][tile_id][y_in_tile][x_in_tile][1];
        d_pixel.z = d_img[batch_id][tile_id][y_in_tile][x_in_tile][2];
        bool Done = (d_pixel.x == 0 && d_pixel.y == 0 && d_pixel.z == 0);

        float3 accum_rec{ 0,0,0 };
        float3 last_color{ 0,0,0 };
        float last_alpha = 0;
        
        for (int offset = end_index_in_tile-1; offset >= start_index_in_tile; offset -= (tilesize * tilesize / 4))
        {

            int valid_num = min(tilesize * tilesize / 4, offset- start_index_in_tile+1);
            //load to shared memory
            if (threadIdx.y * blockDim.x + threadIdx.x < valid_num)
            {
                int i = threadIdx.y * blockDim.x + threadIdx.x;
                int index = offset - i;
                int point_id = sorted_points[batch_id][index];
                collected_point_id[i] = point_id;
                collected_opacity[i] = opacity[batch_id][point_id][0];

                collected_mean_x[i] = mean2d[batch_id][point_id][0];
                collected_mean_y[i] = mean2d[batch_id][point_id][1];

                collected_invcov_x[i] = cov2d_inv[batch_id][point_id][0][0];
                collected_invcov_y[i] = cov2d_inv[batch_id][point_id][0][1];
                collected_invcov_z[i] = cov2d_inv[batch_id][point_id][1][1];

                collected_color_x[i] = color[batch_id][point_id][0];
                collected_color_y[i] = color[batch_id][point_id][1];
                collected_color_z[i] = color[batch_id][point_id][2];
            }

            __syncthreads();


            //process
            
            for (int i = 0; i < valid_num; i++)
            {
                int index = offset - i;
                bool bSkip = false;
                if (index > pixel_lst_index)
                {
                    bSkip = true;
                }
                int threadidx = threadIdx.y * blockDim.x + threadIdx.x;

                float2 xy{ collected_mean_x[i], collected_mean_y[i] };
                float2 d { xy.x - pixel_x,xy.y - pixel_y };
                float3 cur_color{collected_color_x[i],collected_color_y[i],collected_color_z[i] };
                float cur_opacity = collected_opacity[i];
                float3 cur_cov2d_inv = { collected_invcov_x[i],collected_invcov_y[i],collected_invcov_z[i] };

                float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                if (power > 0.0f)
                    bSkip = true;

                float G = exp(power);
                float alpha = min(0.99f, cur_opacity * G);
                if (alpha < 1.0f / 255.0f)
                    bSkip = true;

                if (bSkip == false && Done == false)
                {
                    transmittance /= (1 - alpha);
                    //color
                    grad_color_x[threadidx] = alpha * transmittance * d_pixel.x;
                    grad_color_y[threadidx] = alpha * transmittance * d_pixel.y;
                    grad_color_z[threadidx] = alpha * transmittance * d_pixel.z;


                    //alpha
                    accum_rec.x = last_alpha * last_color.x + (1.0f - last_alpha) * accum_rec.x;
                    accum_rec.y = last_alpha * last_color.y + (1.0f - last_alpha) * accum_rec.y;
                    accum_rec.z = last_alpha * last_color.z + (1.0f - last_alpha) * accum_rec.z;
                    last_color = cur_color;
                    last_alpha = alpha;

                    float d_alpha = 0;
                    d_alpha += (cur_color.x - accum_rec.x) * transmittance * d_pixel.x;
                    d_alpha += (cur_color.y - accum_rec.y) * transmittance * d_pixel.y;
                    d_alpha += (cur_color.z - accum_rec.z) * transmittance * d_pixel.z;

                    //opacity
                    grad_opacity[threadidx] = G * d_alpha;

                    //cov2d_inv
                    float d_G = cur_opacity * d_alpha;
                    float d_power = G * d_G;
                    grad_invcov_x[threadidx] = -0.5f * d.x * d.x * d_power;
                    grad_invcov_z[threadidx] = -0.5f * d.y * d.y * d_power;
                    grad_invcov_y[threadidx] = -0.5f * d.x * d.y * d_power;
                    //atomicAdd(&(d_cov2d_inv[batch_id][collected_point_id[i]][1][0]), -0.5f * d.y * d.x * d_power);

                    //mean2d
                    float d_deltax = (-cur_cov2d_inv.x * d.x - cur_cov2d_inv.y * d.y) * d_power;
                    float d_deltay = (-cur_cov2d_inv.z * d.y - cur_cov2d_inv.y * d.x) * d_power;
                    //d_x=d_deltax
                    //d_y=d_deltay
                    grad_mean_x[threadidx] = d_deltax;
                    grad_mean_y[threadidx] = d_deltay;
                }
                else
                {
                    grad_color_x[threadidx] = 0;
                    grad_color_y[threadidx] = 0;
                    grad_color_z[threadidx] = 0;

                    grad_invcov_x[threadidx] = 0;
                    grad_invcov_y[threadidx] = 0;
                    grad_invcov_z[threadidx] = 0;

                    grad_mean_x[threadidx] = 0;
                    grad_mean_y[threadidx] = 0;

                    grad_opacity[threadidx] = 0;

                }

                __syncthreads();
                //reduction
                for (int step_size = tilesize * tilesize / 2; step_size > 0; step_size /= 2)
                {
                    if (threadidx < step_size)
                    {
                        grad_color_x[threadidx] += grad_color_x[threadidx + step_size];
                        grad_color_y[threadidx] += grad_color_y[threadidx + step_size];
                        grad_color_z[threadidx] += grad_color_z[threadidx + step_size];

                        grad_invcov_x[threadidx] += grad_invcov_x[threadidx + step_size];
                        grad_invcov_y[threadidx] += grad_invcov_y[threadidx + step_size];
                        grad_invcov_z[threadidx] += grad_invcov_z[threadidx + step_size];

                        grad_mean_x[threadidx] += grad_mean_x[threadidx + step_size];
                        grad_mean_y[threadidx] += grad_mean_y[threadidx + step_size];

                        grad_opacity[threadidx] += grad_opacity[threadidx + step_size];
                    }
                    __syncthreads();
                }

                if (threadidx == 0)
                {
                    int point_id = collected_point_id[i];
                    atomicAdd(&d_color[batch_id][point_id][0], grad_color_x[0]);
                    atomicAdd(&d_color[batch_id][point_id][1], grad_color_y[0]);
                    atomicAdd(&d_color[batch_id][point_id][2], grad_color_z[0]);

                    atomicAdd(&d_opacity[batch_id][point_id][0], grad_opacity[0]);

                    atomicAdd(&d_mean2d[batch_id][point_id][0], grad_mean_x[0]);
                    atomicAdd(&d_mean2d[batch_id][point_id][1], grad_mean_y[0]);

                    atomicAdd(&d_cov2d_inv[batch_id][point_id][0][0], grad_invcov_x[0]);
                    atomicAdd(&d_cov2d_inv[batch_id][point_id][0][1], grad_invcov_y[0]);
                    atomicAdd(&d_cov2d_inv[batch_id][point_id][1][0], grad_invcov_y[0]);
                    atomicAdd(&d_cov2d_inv[batch_id][point_id][1][1], grad_invcov_z[0]);
                }
                __syncthreads();

            }
            __syncthreads();
            
        }
        

    }
}
#else
template <int tilesize>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> mean2d,         //[batch,point_num,2]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,point_num,2,2]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,point_num,3]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,          //[batch,point_num,1]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> final_transmitance,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,tile,tilesize,tilesize,3]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_mean2d,         //[batch,point_num,2]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,point_num,2,2]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,point_num,3]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_opacity,          //[batch,point_num,1]
    int tiles_num_x
)
{


    __shared__ float2 collected_xy[tilesize * tilesize];
    __shared__ float collected_opacity[tilesize * tilesize];
    __shared__ float3 collected_cov2d_inv[tilesize * tilesize];
    __shared__ float3 collected_color[tilesize * tilesize];
    __shared__ int collected_point_id[tilesize * tilesize];

    const int batch_id = blockIdx.y;
    const int tile_id = blockIdx.x + 1;

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id) / tiles_num_x) * tilesize + y_in_tile;

    if (tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        float transmittance = final_transmitance[batch_id][tile_id][y_in_tile][x_in_tile];
        int pixel_lst_index = last_contributor[batch_id][tile_id][y_in_tile][x_in_tile];
        float3 d_pixel{ 0,0,0 };
        d_pixel.x = d_img[batch_id][tile_id][y_in_tile][x_in_tile][0];
        d_pixel.y = d_img[batch_id][tile_id][y_in_tile][x_in_tile][1];
        d_pixel.z = d_img[batch_id][tile_id][y_in_tile][x_in_tile][2];
        bool Done = (d_pixel.x == 0 && d_pixel.y == 0 && d_pixel.z == 0);

        float3 accum_rec{ 0,0,0 };
        float3 last_color{ 0,0,0 };
        float last_alpha = 0;

        for (int offset = end_index_in_tile - 1; offset >= start_index_in_tile; offset -= tilesize * tilesize)
        {

            int valid_num = min(tilesize * tilesize, offset - start_index_in_tile + 1);
            //load to shared memory
            if (threadIdx.y * blockDim.x + threadIdx.x < valid_num)
            {
                int i = threadIdx.y * blockDim.x + threadIdx.x;
                int index = offset - i;
                int point_id = sorted_points[batch_id][index];
                collected_point_id[i] = point_id;
                collected_xy[i].x = mean2d[batch_id][point_id][0];
                collected_xy[i].y = mean2d[batch_id][point_id][1];
                collected_cov2d_inv[i].x = cov2d_inv[batch_id][point_id][0][0];
                collected_cov2d_inv[i].y = cov2d_inv[batch_id][point_id][0][1];
                collected_cov2d_inv[i].z = cov2d_inv[batch_id][point_id][1][1];

                collected_color[i].x = color[batch_id][point_id][0];
                collected_color[i].y = color[batch_id][point_id][1];
                collected_color[i].z = color[batch_id][point_id][2];
                collected_opacity[i] = opacity[batch_id][point_id][0];
            }
            __syncthreads();


            //process

            for (int i = 0; i < valid_num && Done == false; i++)
            {
                int index = offset - i;
                if (index > pixel_lst_index)
                {
                    continue;
                }

                float2 xy = collected_xy[i];
                float2 d = { xy.x - pixel_x,xy.y - pixel_y };
                float3 cur_color = collected_color[i];
                float cur_opacity = collected_opacity[i];
                float3 cur_cov2d_inv = collected_cov2d_inv[i];

                float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                if (power > 0.0f)
                    continue;

                float G = exp(power);
                float alpha = min(0.99f, cur_opacity * G);
                if (alpha < 1.0f / 255.0f)
                    continue;

                transmittance /= (1 - alpha);

                //color
                atomicAdd(&(d_color[batch_id][collected_point_id[i]][0]), alpha * transmittance * d_pixel.x);
                atomicAdd(&(d_color[batch_id][collected_point_id[i]][1]), alpha * transmittance * d_pixel.y);
                atomicAdd(&(d_color[batch_id][collected_point_id[i]][2]), alpha * transmittance * d_pixel.z);


                //alpha
                accum_rec.x = last_alpha * last_color.x + (1.0f - last_alpha) * accum_rec.x;
                accum_rec.y = last_alpha * last_color.y + (1.0f - last_alpha) * accum_rec.y;
                accum_rec.z = last_alpha * last_color.z + (1.0f - last_alpha) * accum_rec.z;
                last_color = cur_color;
                last_alpha = alpha;

                float d_alpha = 0;
                d_alpha += (cur_color.x - accum_rec.x) * transmittance * d_pixel.x;
                d_alpha += (cur_color.y - accum_rec.y) * transmittance * d_pixel.y;
                d_alpha += (cur_color.z - accum_rec.z) * transmittance * d_pixel.z;

                //opacity
                atomicAdd(&(d_opacity[batch_id][collected_point_id[i]][0]), G * d_alpha);

                //cov2d_inv
                float d_G = cur_opacity * d_alpha;
                float d_power = G * d_G;
                atomicAdd(&(d_cov2d_inv[batch_id][collected_point_id[i]][0][0]), -0.5f * d.x * d.x * d_power);
                atomicAdd(&(d_cov2d_inv[batch_id][collected_point_id[i]][1][1]), -0.5f * d.y * d.y * d_power);
                atomicAdd(&(d_cov2d_inv[batch_id][collected_point_id[i]][0][1]), -0.5f * d.x * d.y * d_power);
                atomicAdd(&(d_cov2d_inv[batch_id][collected_point_id[i]][1][0]), -0.5f * d.y * d.x * d_power);

                //mean2d
                float d_deltax = (-cur_cov2d_inv.x * d.x - cur_cov2d_inv.y * d.y) * d_power;
                float d_deltay = (-cur_cov2d_inv.z * d.y - cur_cov2d_inv.y * d.x) * d_power;
                //d_x=d_deltax
                //d_y=d_deltay
                atomicAdd(&(d_mean2d[batch_id][collected_point_id[i]][0]), d_deltax);
                atomicAdd(&(d_mean2d[batch_id][collected_point_id[i]][1]), d_deltay);

            }
            __syncthreads();

        }


    }
}
#endif


std::vector<Tensor> rasterize_backward(
    Tensor sorted_points,
    Tensor start_index,
    Tensor mean2d,// 
    Tensor cov2d_inv,
    Tensor color,
    Tensor opacity,
    Tensor final_transmitance,
    Tensor last_contributor,
    Tensor d_img,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y
)
{
    DeviceGuard guard(mean2d.device());

    int64_t viewsnum = start_index.sizes()[0];
    int64_t tilesnum = tilesnum_x * tilesnum_y;

    Tensor d_mean2d = torch::zeros_like(mean2d,mean2d.options());
    Tensor d_cov2d_inv = torch::zeros_like(cov2d_inv, mean2d.options());
    Tensor d_color = torch::zeros_like(color, mean2d.options());
    Tensor d_opacity = torch::zeros_like(opacity, mean2d.options());

    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);
    if (tilesize == 16)
    {
        //cudaDeviceSynchronize();
        raster_backward_kernel<16> << <Block3d, Thread3d >> >
            (
                sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
                last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                d_mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
                d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
                d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
                d_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
                tilesnum_x
                );
        CUDA_CHECK_ERRORS;
    }
    else if (tilesize == 32)
    {
        raster_backward_kernel<32> << <Block3d, Thread3d >> >
            (
                sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
                last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
                d_mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
                d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
                d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
                d_opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
                tilesnum_x
                );
        CUDA_CHECK_ERRORS;
    }


    return { d_mean2d ,d_cov2d_inv ,d_color,d_opacity };
}