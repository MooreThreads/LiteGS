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
#include "raster_binning.h"
#include <ATen/core/TensorAccessor.h>
//using namespace at;

#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

 __global__ void duplicate_with_keys_kernel(
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> LU,//viewnum,pointnum
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> RD,
    const torch::PackedTensorAccessor32<int64_t, 2,torch::RestrictPtrTraits> prefix_sum,//view,pointnum
    int TileSizeX,
    torch::PackedTensorAccessor32 < int16_t, 2, torch::RestrictPtrTraits> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_pointId
    )
{
    int view_id = blockIdx.y;
    int point_id = blockIdx.x*blockDim.x + threadIdx.x;

    if (point_id < prefix_sum.size(1))
    {
        int end = prefix_sum[view_id][point_id];
        //int end = prefix_sum[view_id][point_id+1];
        int l = LU[view_id][point_id][0];
        int u = LU[view_id][point_id][1];
        int r = RD[view_id][point_id][0];
        int d = RD[view_id][point_id][1];
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



std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU, at::Tensor RD, at::Tensor prefix_sum, int64_t allocate_size, int64_t TilesSizeX)
{
    at::DeviceGuard guard(LU.device());
    int64_t view_num = LU.sizes()[0];
    int64_t points_num = LU.sizes()[1];

    std::vector<int64_t> output_shape{ view_num, allocate_size };

    auto opt = torch::TensorOptions().dtype(torch::kInt16).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_tileId = torch::zeros(output_shape, opt);
    opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_pointId= torch::zeros(output_shape, opt);

    dim3 Block3d(std::ceil(points_num/1024.0f), view_num, 1);
    

    duplicate_with_keys_kernel<<<Block3d ,1024>>>(
        LU.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        RD.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
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




template <int tilesize>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> mean2d,         //[batch,point_num,2]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,point_num,2,2]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,point_num,3]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[point_num,1]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tiles,          //[batch,tiles_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output_transmitance,    //[batch,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    int tiles_num_x
)
{


    __shared__ float2 collected_xy[tilesize * tilesize / 4];
    __shared__ float collected_opacity[tilesize * tilesize / 4];
    __shared__ float3 collected_cov2d_inv[tilesize * tilesize / 4];
    __shared__ float3 collected_color[tilesize * tilesize / 4];

    const int batch_id = blockIdx.y;
    int tile_id = tiles[batch_id][blockIdx.x];

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id-1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id-1) / tiles_num_x) * tilesize + y_in_tile;

    if (tile_id!=0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        float transmittance = 1.0f;
        bool done = false;
        float3 final_color{ 0,0,0 };
        int last_contributor = 0;
        for (int offset = start_index_in_tile; offset < end_index_in_tile; offset += tilesize * tilesize / 4)
        {
            int num_done = __syncthreads_count(done);
            if (num_done == blockDim.x * blockDim.y)
                break;

            int valid_num = min(tilesize * tilesize / 4, end_index_in_tile - offset);
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
                collected_opacity[i] = opacity[point_id][0];
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


            }
            __syncthreads();
        }

        output_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile] = final_color.x;
        output_img[batch_id][1][blockIdx.x][y_in_tile][x_in_tile] = final_color.y;
        output_img[batch_id][2][blockIdx.x][y_in_tile][x_in_tile] = final_color.z;

        output_last_contributor[batch_id][blockIdx.x][y_in_tile][x_in_tile] = last_contributor;
        output_transmitance[batch_id][blockIdx.x][y_in_tile][x_in_tile] = transmittance;
    }
}

std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor  mean2d,// 
    at::Tensor  cov2d_inv,
    at::Tensor  color,
    at::Tensor  opacity,
    at::Tensor  tiles,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y,
    int64_t img_h,
    int64_t img_w
)
{
    at::DeviceGuard guard( mean2d.device());

    int64_t viewsnum = start_index.sizes()[0];
    int64_t tilesnum = tiles.sizes()[1];

    std::vector<int64_t> shape_img{ viewsnum,3, tilesnum,tilesize,tilesize };
    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_img = torch::zeros(shape_img, opt_img);

    std::vector<int64_t> shape_t{ viewsnum, tilesnum, tilesize, tilesize };
    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_transmitance = torch::zeros(shape_t, opt_t);

    std::vector<int64_t> shape_c{ viewsnum, tilesnum, tilesize, tilesize };
    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(start_index.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::zeros(shape_c, opt_c);



    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);
    switch (tilesize)
    {
    case 8:
        raster_forward_kernel<8> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x);
        break;
    case 16:
        raster_forward_kernel<16> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x);
        break;
    case 32:
        raster_forward_kernel<32> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x);
        break;
    default:
        ;
    }
    CUDA_CHECK_ERRORS;
    


    return { output_img ,output_transmitance ,output_last_contributor };
}


__global__ void raster_backward_kernel_8x8(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> mean2d,         //[batch,point_num,2]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,point_num,2,2]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,point_num,3]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[batch,point_num,1]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> final_transmitance,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_mean2d,         //[batch,point_num,2]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,point_num,2,2]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,point_num,3]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[point_num,1]
    int tiles_num_x, int img_h, int img_w
)
{
    const int tilesize = 8;
    extern __shared__ float shared_buffer[];
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    const int batch_id = blockIdx.y;
    const int tile_index = block.group_index().x * warp.meta_group_size() + warp.meta_group_rank();
    int tile_id = 0;
    if (tile_index < tiles.size(1))
        tile_id = tiles[batch_id][tile_index];


    if (tile_id != 0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        //loop pixels in tile
        for (int pixel_id = warp.thread_rank(); pixel_id < tilesize * tilesize; pixel_id += 32)
        {
            float transmittance = final_transmitance[batch_id][tile_index][pixel_id / tilesize][pixel_id % tilesize];//todo
            int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + pixel_id % tilesize;
            int pixel_y = ((tile_id - 1) / tiles_num_x) * tilesize + pixel_id / tilesize;
            int pixel_lst_index = last_contributor[batch_id][tile_index][pixel_id / tilesize][pixel_id % tilesize];

            float3 d_pixel{ 0,0,0 };
            if (pixel_x < img_w && pixel_y < img_h)
            {
                d_pixel.x = d_img[batch_id][0][tile_index][pixel_id / tilesize][pixel_id % tilesize];
                d_pixel.y = d_img[batch_id][1][tile_index][pixel_id / tilesize][pixel_id % tilesize];
                d_pixel.z = d_img[batch_id][2][tile_index][pixel_id / tilesize][pixel_id % tilesize];
            }
            //loop points
            float3 accum_rec{ 0,0,0 };
            for (int point_index = end_index_in_tile - 1; point_index >= start_index_in_tile; point_index--)
            {
                bool skip = point_index > pixel_lst_index;
                int point_id = sorted_points[batch_id][point_index];
                float3 grad_color{ 0,0,0 };
                float3 grad_invcov{ 0,0,0 };
                float2 grad_mean{ 0,0 };
                float grad_opacity{ 0 };

                if (skip == false)
                {
                    float2 xy{ mean2d[batch_id][point_id][0],mean2d[batch_id][point_id][1] };
                    float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                    float4 cur_color{ color[batch_id][point_id][0],color[batch_id][point_id][1],color[batch_id][point_id][2],opacity[point_id][0] };
                    float3 cur_cov2d_inv{ cov2d_inv[batch_id][point_id][0][0],cov2d_inv[batch_id][point_id][0][1],cov2d_inv[batch_id][point_id][1][1] };


                    float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                    skip |= power > 0.0f;

                    float G = exp(power);
                    float alpha = min(0.99f, cur_color.w * G);
                    skip |= (alpha < 1.0f / 255.0f);
                    if (skip == false)
                    {
                        transmittance /= (1 - alpha);
                        //color
                        grad_color.x = alpha * transmittance * d_pixel.x;
                        grad_color.y = alpha * transmittance * d_pixel.y;
                        grad_color.z = alpha * transmittance * d_pixel.z;


                        //alpha
                        float d_alpha = 0;
                        d_alpha += (cur_color.x - accum_rec.x) * transmittance * d_pixel.x;
                        d_alpha += (cur_color.y - accum_rec.y) * transmittance * d_pixel.y;
                        d_alpha += (cur_color.z - accum_rec.z) * transmittance * d_pixel.z;
                        accum_rec.x = alpha * cur_color.x + (1.0f - alpha) * accum_rec.x;
                        accum_rec.y = alpha * cur_color.y + (1.0f - alpha) * accum_rec.y;
                        accum_rec.z = alpha * cur_color.z + (1.0f - alpha) * accum_rec.z;

                        //opacity
                        grad_opacity = G * d_alpha;

                        //cov2d_inv
                        float d_G = cur_color.w * d_alpha;
                        float d_power = G * d_G;
                        grad_invcov.x = -0.5f * d.x * d.x * d_power;
                        grad_invcov.y = -0.5f * d.x * d.y * d_power;
                        grad_invcov.z = -0.5f * d.y * d.y * d_power;

                        //mean2d
                        float d_deltax = (-cur_cov2d_inv.x * d.x - cur_cov2d_inv.y * d.y) * d_power;
                        float d_deltay = (-cur_cov2d_inv.z * d.y - cur_cov2d_inv.y * d.x) * d_power;
                        grad_mean.x = d_deltax;
                        grad_mean.y = d_deltay;
                    }

                }

                
                if (warp.all(skip) == false)
                {
                    for (int offset = 16; offset > 0; offset /= 2)
                    {
                        grad_color.x += __shfl_down_sync(0xffffffff, grad_color.x, offset);
                        grad_color.y += __shfl_down_sync(0xffffffff, grad_color.y, offset);
                        grad_color.z += __shfl_down_sync(0xffffffff, grad_color.z, offset);

                        grad_invcov.x += __shfl_down_sync(0xffffffff, grad_invcov.x, offset);
                        grad_invcov.y += __shfl_down_sync(0xffffffff, grad_invcov.y, offset);
                        grad_invcov.z += __shfl_down_sync(0xffffffff, grad_invcov.z, offset);

                        grad_mean.x += __shfl_down_sync(0xffffffff, grad_mean.x, offset);
                        grad_mean.y += __shfl_down_sync(0xffffffff, grad_mean.y, offset);

                        grad_opacity += __shfl_down_sync(0xffffffff, grad_opacity, offset);
                    }
                    if (warp.thread_rank() == 0)
                    {
                        atomicAdd(&d_color[batch_id][point_id][0], grad_color.x);
                        atomicAdd(&d_color[batch_id][point_id][1], grad_color.y);
                        atomicAdd(&d_color[batch_id][point_id][2], grad_color.z);

                        atomicAdd(&d_cov2d_inv[batch_id][point_id][0][0], grad_invcov.x);
                        atomicAdd(&d_cov2d_inv[batch_id][point_id][0][1], grad_invcov.y);
                        atomicAdd(&d_cov2d_inv[batch_id][point_id][1][0], grad_invcov.y);
                        atomicAdd(&d_cov2d_inv[batch_id][point_id][1][1], grad_invcov.z);

                        atomicAdd(&d_mean2d[batch_id][point_id][0], grad_mean.x);
                        atomicAdd(&d_mean2d[batch_id][point_id][1], grad_mean.y);

                        atomicAdd(&d_opacity[point_id][0], grad_opacity);
                    }
                }
            }

        }


    }
}


template <int tilesize>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> mean2d,         //[batch,point_num,2]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,point_num,2,2]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,point_num,3]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[point_num,1]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> final_transmitance,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_mean2d,         //[batch,point_num,2]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,point_num,2,2]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,point_num,3]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[point_num,1]
    int tiles_num_x, int img_h, int img_w
)
{
    __shared__ int collected_point_id[tilesize * tilesize / 4];
    __shared__ float4 collected_color[tilesize * tilesize / 4];
    __shared__ float3 collected_invcov[tilesize * tilesize / 4];
    __shared__ float2 collected_mean[tilesize * tilesize / 4];

    __shared__ float gradient_buffer[tilesize * tilesize * 9];
    float* const grad_color_x = gradient_buffer;
    float* const grad_color_y = gradient_buffer + 1 * tilesize * tilesize;
    float* const grad_color_z = gradient_buffer + 2 * tilesize * tilesize;
    float* const grad_invcov_x = gradient_buffer + 3 * tilesize * tilesize;
    float* const grad_invcov_y = gradient_buffer + 4 * tilesize * tilesize;
    float* const grad_invcov_z = gradient_buffer + 5 * tilesize * tilesize;
    float* const grad_mean_x = gradient_buffer + 6 * tilesize * tilesize;
    float* const grad_mean_y = gradient_buffer + 7 * tilesize * tilesize;
    float* const grad_opacity = gradient_buffer + 8 * tilesize * tilesize;
    __shared__ float shared_gradient_sum[9];

    const int batch_id = blockIdx.y;
    int tile_id = tiles[batch_id][blockIdx.x];
    auto block = cg::this_thread_block();
    auto cuda_tile = cg::tiled_partition<32>(block);

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id - 1) / tiles_num_x) * tilesize + y_in_tile;

    if (tile_id!=0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];
        if (start_index_in_tile == -1)
        {
            return;
        }

        float transmittance = final_transmitance[batch_id][blockIdx.x][y_in_tile][x_in_tile];
        int pixel_lst_index = last_contributor[batch_id][blockIdx.x][y_in_tile][x_in_tile];

        float3 d_pixel{ 0,0,0 };
        if (pixel_x < img_w && pixel_y < img_h)
        {
            d_pixel.x = d_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
            d_pixel.y = d_img[batch_id][1][blockIdx.x][y_in_tile][x_in_tile];
            d_pixel.z = d_img[batch_id][2][blockIdx.x][y_in_tile][x_in_tile];
        }

        float3 accum_rec{ 0,0,0 };
        for (int offset = end_index_in_tile-1; offset >= start_index_in_tile; offset -= (tilesize * tilesize / 4))
        {

            int valid_num = min(tilesize * tilesize / 4, offset- start_index_in_tile+1);
            //load to shared memory
            __syncthreads();
            if (threadIdx.y * blockDim.x + threadIdx.x < valid_num)
            {
                int i = threadIdx.y * blockDim.x + threadIdx.x;
                int index = offset - i;
                int point_id = sorted_points[batch_id][index];
                collected_point_id[i] = point_id;

                float2 _mean2d = *((float2*)(mean2d[batch_id][point_id].data()));
                collected_mean[i] = _mean2d;
                float4 _cov2d_inv= *((float4*)cov2d_inv[batch_id][point_id].data());
                collected_invcov[i].x = _cov2d_inv.x;
                collected_invcov[i].y = _cov2d_inv.y;
                collected_invcov[i].z = _cov2d_inv.w;
                float3 _color = *((float3*)(color[batch_id][point_id].data()));
                collected_color[i].x = _color.x;
                collected_color[i].y = _color.y;
                collected_color[i].z = _color.z;
                collected_color[i].w = opacity[point_id][0];
            }
            __syncthreads();


            //process
            
            for (int i = 0; i < valid_num; i++)
            {
                int index = offset - i;
                bool bSkip = false;
                int threadidx = threadIdx.y * blockDim.x + threadIdx.x;
                if (index > pixel_lst_index)
                {
                    bSkip = true;
                }
                else
                {

                    float2 xy = collected_mean[i];
                    float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                    float4 cur_color = collected_color[i];
                    float3 cur_cov2d_inv = collected_invcov[i];

                    float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                    bSkip |= power > 0.0f;

                    float G = exp(power);
                    float alpha = min(0.99f, cur_color.w * G);
                    bSkip |= alpha < 1.0f / 255.0f;

                    if (bSkip == false)
                    {
                        transmittance /= (1 - alpha);
                        //color
                        grad_color_x[threadidx] = alpha * transmittance * d_pixel.x;
                        grad_color_y[threadidx] = alpha * transmittance * d_pixel.y;
                        grad_color_z[threadidx] = alpha * transmittance * d_pixel.z;


                        //alpha
                        float d_alpha = 0;
                        d_alpha += (cur_color.x - accum_rec.x) * transmittance * d_pixel.x;
                        d_alpha += (cur_color.y - accum_rec.y) * transmittance * d_pixel.y;
                        d_alpha += (cur_color.z - accum_rec.z) * transmittance * d_pixel.z;
                        accum_rec.x = alpha * cur_color.x + (1.0f - alpha) * accum_rec.x;
                        accum_rec.y = alpha * cur_color.y + (1.0f - alpha) * accum_rec.y;
                        accum_rec.z = alpha * cur_color.z + (1.0f - alpha) * accum_rec.z;

                        //opacity
                        grad_opacity[threadidx] = G * d_alpha;

                        //cov2d_inv
                        float d_G = cur_color.w * d_alpha;
                        float d_power = G * d_G;
                        grad_invcov_x[threadidx] = -0.5f * d.x * d.x * d_power;
                        grad_invcov_y[threadidx] = -0.5f * d.x * d.y * d_power;
                        grad_invcov_z[threadidx] = -0.5f * d.y * d.y * d_power;

                        //mean2d
                        float d_deltax = (-cur_cov2d_inv.x * d.x - cur_cov2d_inv.y * d.y) * d_power;
                        float d_deltay = (-cur_cov2d_inv.z * d.y - cur_cov2d_inv.y * d.x) * d_power;
                        grad_mean_x[threadidx] = d_deltax;
                        grad_mean_y[threadidx] = d_deltay;
                    }
                }

                
                if (bSkip == true )
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

                //__syncthreads();
                bool block_skip=__syncthreads_and(bSkip);
                int point_id = collected_point_id[i];
                //reduction
                if (block_skip==false)
                {
                    
                    int threadid_in_warp = threadidx % 32;
                    int wraps_num = tilesize * tilesize / 32;
                    for (int property_id = threadidx / 32; property_id < 9; property_id+= wraps_num)//9 property num
                    {
                        float gradient_sum = 0;
                        for (int reduction_i = threadid_in_warp; reduction_i < tilesize * tilesize; reduction_i+=32)
                        {
                            gradient_sum += gradient_buffer[property_id * tilesize * tilesize + reduction_i];
                        }
                        for (int offset = 16; offset > 0; offset /= 2)
                        {
                            gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, offset);
                        }
                        if (threadid_in_warp == 0)
                        {
                            shared_gradient_sum[property_id] = gradient_sum;
                        }
                    }
                    __syncthreads();
                    if (threadidx == 0)
                    {
                        atomicAdd(&d_color[batch_id][point_id][0], shared_gradient_sum[0]);
                        atomicAdd(&d_color[batch_id][point_id][1], shared_gradient_sum[1]);
                        atomicAdd(&d_color[batch_id][point_id][2], shared_gradient_sum[2]);
                                
                        atomicAdd(&d_cov2d_inv[batch_id][point_id][0][0], shared_gradient_sum[3]);
                        atomicAdd(&d_cov2d_inv[batch_id][point_id][0][1], shared_gradient_sum[4]);
                        atomicAdd(&d_cov2d_inv[batch_id][point_id][1][0], shared_gradient_sum[4]);
                        atomicAdd(&d_cov2d_inv[batch_id][point_id][1][1], shared_gradient_sum[5]);

                        atomicAdd(&d_mean2d[batch_id][point_id][0], shared_gradient_sum[6]);
                        atomicAdd(&d_mean2d[batch_id][point_id][1], shared_gradient_sum[7]);

                        atomicAdd(&d_opacity[point_id][0], shared_gradient_sum[8]);
                    }
                    
                }

            }
            
        }
        

    }
}


std::vector<at::Tensor> rasterize_backward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor mean2d,// 
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    at::Tensor tiles,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y,
    int64_t img_h,
    int64_t img_w
)
{
    at::DeviceGuard guard(mean2d.device());

    int64_t viewsnum = start_index.sizes()[0];
    int64_t tilesnum = tiles.sizes()[1];

    at::Tensor d_mean2d = torch::zeros_like(mean2d,mean2d.options());
    at::Tensor d_cov2d_inv = torch::zeros_like(cov2d_inv, mean2d.options());
    at::Tensor d_color = torch::zeros_like(color, mean2d.options());
    at::Tensor d_opacity = torch::zeros_like(opacity, mean2d.options());

    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);

    int TilesNumInBlock = 4;
    dim3 Block3d8x8((tilesnum + TilesNumInBlock - 1) / TilesNumInBlock, viewsnum, 1);
    int ThreadsNum8x8 = 32 * TilesNumInBlock;
    
    switch (tilesize)
    {
    case 8:
        //todo cuda perfer shared
        raster_backward_kernel_8x8 << <Block3d8x8, ThreadsNum8x8 >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w);
        break;
        //raster_backward_kernel<8> << <Block3d, Thread3d >> > (
        //    sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        //    start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        //    mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        //    cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        //    color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        //    opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        //    tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        //    final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
        //    last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        //    d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        //    d_mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        //    d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
        //    d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        //    d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
        //    tilesnum_x, img_h, img_w);
        //break;
    case 16:
        raster_backward_kernel<16> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w);
        break;
    case 32:
        raster_backward_kernel<32> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_mean2d.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w);
        break;
    default:
        ;
    }
    CUDA_CHECK_ERRORS;
    
    return { d_mean2d ,d_cov2d_inv ,d_color,d_opacity };
}


template <typename scalar_t,bool TRNASPOSE=true>
__global__ void jacobian_rayspace_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> translated_position,    //[batch,point_num,4] 
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> camera_focal,    //[batch,2] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jacobian         //[batch,point_num,3,3]
    )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;
    if (batch_id < translated_position.size(0) && index < translated_position.size(1))
    {
        float focalx = camera_focal[batch_id][0];
        float focaly = camera_focal[batch_id][1];

        float reciprocal_tz = 1.0f/translated_position[batch_id][index][2];
        float square_reciprocal_tz = reciprocal_tz * reciprocal_tz;

        jacobian[batch_id][index][0][0] = focalx * reciprocal_tz;
        jacobian[batch_id][index][1][1] = focaly * reciprocal_tz;
        if (TRNASPOSE)
        {
            jacobian[batch_id][index][0][2] = -focalx * translated_position[batch_id][index][0] * square_reciprocal_tz;
            jacobian[batch_id][index][1][2] = -focaly * translated_position[batch_id][index][1] * square_reciprocal_tz;
        }
        else
        {
            jacobian[batch_id][index][2][0] = -focalx * translated_position[batch_id][index][0] * square_reciprocal_tz;
            jacobian[batch_id][index][2][1] = -focaly * translated_position[batch_id][index][1] * square_reciprocal_tz;
        }
    }
}

at::Tensor jacobianRayspace(
    at::Tensor translated_position, //N,P,4
    at::Tensor camera_focal, //N,2
    bool bTranspose
)
{
    int N = translated_position.size(0);
    int P = translated_position.size(1);
    at::Tensor jacobian_matrix = torch::zeros({N,P,3,3}, translated_position.options());

    int threadsnum = 256;
    dim3 Block3d(std::ceil(P/(float)threadsnum), N, 1);
    if (bTranspose)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(translated_position.type(), __FUNCTION__, [&] {jacobian_rayspace_kernel<scalar_t,true > << <Block3d, threadsnum >> > (
            translated_position.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            camera_focal.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            jacobian_matrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });
    }
    else
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(translated_position.type(), __FUNCTION__, [&] {jacobian_rayspace_kernel<scalar_t, false > << <Block3d, threadsnum >> > (
            translated_position.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            camera_focal.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            jacobian_matrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });
    }

    CUDA_CHECK_ERRORS;
    return jacobian_matrix;

}

__global__ void create_transform_matrix_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> quaternion,    //[point_num,3]  
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> scale,    //[point_num,4] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> transform         //[point_num,3,3]
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index < quaternion.size(0))
    {
        float r = quaternion[index][0];
        float x = quaternion[index][1];
        float y = quaternion[index][2];
        float z = quaternion[index][3];

        float scale_x = scale[index][0];
        float scale_y = scale[index][1];
        float scale_z = scale[index][2];

        transform[index][0][0] = (1 - 2 * (y * y + z * z))*scale_x;
        transform[index][0][1] = 2 * (x * y + r * z) * scale_x;
        transform[index][0][2] = 2 * (x * z - r * y) * scale_x;

        transform[index][1][0] = 2 * (x * y - r * z) * scale_y;
        transform[index][1][1] = (1 - 2 * (x * x + z * z)) * scale_y;
        transform[index][1][2] = 2 * (y * z + r * x) * scale_y;

        transform[index][2][0] = 2 * (x * z + r * y) * scale_z;
        transform[index][2][1] = 2 * (y * z - r * x) * scale_z;
        transform[index][2][2] = (1 - 2 * (x * x + y * y)) * scale_z;
    }
}

at::Tensor createTransformMatrix_forward(at::Tensor quaternion, at::Tensor scale)
{
    int P = quaternion.size(0);
    at::Tensor transform_matrix = torch::zeros({ P,3,3 }, scale.options());

    int threadsnum = 256;
    int blocknum=std::ceil(P / (float)threadsnum);
    create_transform_matrix_forward_kernel << <blocknum, threadsnum >> > (
        quaternion.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        transform_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return transform_matrix;
}

__global__ void create_transform_matrix_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> quaternion,    //[point_num,3]  
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> scale,    //[point_num,4] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_transform,         //[point_num,3,3]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_quaternion,    //[point_num,3]  
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_scale    //[point_num,4] 

)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index < quaternion.size(0))
    {
        float r = quaternion[index][0];
        float x = quaternion[index][1];
        float y = quaternion[index][2];
        float z = quaternion[index][3];

        float dt[9];
        dt[0 * 3 + 0] = grad_transform[index][0][0];
        dt[0 * 3 + 1] = grad_transform[index][0][1];
        dt[0 * 3 + 2] = grad_transform[index][0][2];

        dt[1 * 3 + 0] = grad_transform[index][1][0];
        dt[1 * 3 + 1] = grad_transform[index][1][1];
        dt[1 * 3 + 2] = grad_transform[index][1][2];

        dt[2 * 3 + 0] = grad_transform[index][2][0];
        dt[2 * 3 + 1] = grad_transform[index][2][1];
        dt[2 * 3 + 2] = grad_transform[index][2][2];

        {
            float grad_scale_x = 0;
            grad_scale_x += (1 - 2 * (y * y + z * z)) * dt[0 * 3 + 0];
            grad_scale_x += 2 * (x * y + r * z) * dt[0 * 3 + 1];
            grad_scale_x += 2 * (x * z - r * y) * dt[0 * 3 + 2];
            grad_scale [index][0] = grad_scale_x;
        }

        {
            float grad_scale_y = 0;
            grad_scale_y += 2 * (x * y - r * z) * dt[1 * 3 + 0];
            grad_scale_y += (1 - 2 * (x * x + z * z)) * dt[1 * 3 + 1];
            grad_scale_y += 2 * (y * z + r * x) * dt[1 * 3 + 2];
            grad_scale [index][1] = grad_scale_y;
        }

        {
            float grad_scale_z = 0;
            grad_scale_z += 2 * (x * z + r * y) * dt[2 * 3 + 0];
            grad_scale_z += 2 * (y * z - r * x) * dt[2 * 3 + 1];
            grad_scale_z += (1 - 2 * (x * x + y * y)) * dt[2 * 3 + 2];
            grad_scale [index][2] = grad_scale_z;
        }

        {
            dt[0 * 3 + 0] *= scale[index][0];
            dt[0 * 3 + 1] *= scale[index][0];
            dt[0 * 3 + 2] *= scale[index][0];

            dt[1 * 3 + 0] *= scale[index][1];
            dt[1 * 3 + 1] *= scale[index][1];
            dt[1 * 3 + 2] *= scale[index][1];

            dt[2 * 3 + 0] *= scale[index][2];
            dt[2 * 3 + 1] *= scale[index][2];
            dt[2 * 3 + 2] *= scale[index][2];

            grad_quaternion[index][0] = 2 * z * (dt[0*3+1] - dt[1*3+0]) + 2 * y * (dt[2*3+0] - dt[0*3+2]) + 2 * x * (dt[1*3+2] - dt[2*3+1]);
            grad_quaternion[index][1] = 2 * y * (dt[1*3+0] + dt[0*3+1]) + 2 * z * (dt[2*3+0] + dt[0*3+2]) + 2 * r * (dt[1*3+2] - dt[2*3+1]) - 4 * x * (dt[2*3+2] + dt[1*3+1]);
            grad_quaternion[index][2] = 2 * x * (dt[1*3+0] + dt[0*3+1]) + 2 * r * (dt[2*3+0] - dt[0*3+2]) + 2 * z * (dt[1*3+2] + dt[2*3+1]) - 4 * y * (dt[2*3+2] + dt[0*3+0]);
            grad_quaternion[index][3] = 2 * r * (dt[0*3+1] - dt[1*3+0]) + 2 * x * (dt[2*3+0] + dt[0*3+2]) + 2 * y * (dt[1*3+2] + dt[2*3+1]) - 4 * z * (dt[1*3+1] + dt[0*3+0]);
        }




    }
}


std::vector<at::Tensor> createTransformMatrix_backward(at::Tensor transform_matrix_grad, at::Tensor quaternion, at::Tensor scale)
{
    //todo
    int P = quaternion.size(0);
    at::Tensor grad_quaternion = torch::zeros({ P,4 }, transform_matrix_grad.options());
    at::Tensor grad_scale = torch::zeros({ P,3 }, transform_matrix_grad.options());

    int threadsnum = 256;
    int blocknum=std::ceil(P / (float)threadsnum);
    create_transform_matrix_backward_kernel << <blocknum, threadsnum >> > (
        quaternion.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        transform_matrix_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_quaternion.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_scale.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;


    return { grad_quaternion,grad_scale };
}


__global__ void world2ndc_backword_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_project_matrix,    //[batch,4,4]  
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc_position,    //[batch,point_num,4] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> repc_hom_w_tensor,         //[batch,point_num,1]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_ndc_pos,    //[batch,point_num,4]  
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_position    //[point_num,4] 

)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int batch_id = 0; batch_id < ndc_position.size(0); batch_id++)
    {
        if (batch_id < ndc_position.size(0) && index < ndc_position.size(1))
        {
            float repc_hom_w = repc_hom_w_tensor[batch_id][index][0];

            float mul1 = ndc_position[batch_id][index][0] * repc_hom_w;
            float mul2 = ndc_position[batch_id][index][1] * repc_hom_w;
            float mul3 = ndc_position[batch_id][index][2] * repc_hom_w;

            float grad_x = (view_project_matrix[batch_id][0][0] * repc_hom_w - view_project_matrix[batch_id][0][3] * mul1) * grad_ndc_pos[batch_id][index][0]
                + (view_project_matrix[batch_id][0][1] * repc_hom_w - view_project_matrix[batch_id][0][3] * mul2) * grad_ndc_pos[batch_id][index][1]
                + (view_project_matrix[batch_id][0][2] * repc_hom_w - view_project_matrix[batch_id][0][3] * mul3) * grad_ndc_pos[batch_id][index][2];

            float grad_y = (view_project_matrix[batch_id][1][0] * repc_hom_w - view_project_matrix[batch_id][1][3] * mul1) * grad_ndc_pos[batch_id][index][0]
                + (view_project_matrix[batch_id][1][1] * repc_hom_w - view_project_matrix[batch_id][1][3] * mul2) * grad_ndc_pos[batch_id][index][1]
                + (view_project_matrix[batch_id][1][2] * repc_hom_w - view_project_matrix[batch_id][1][3] * mul3) * grad_ndc_pos[batch_id][index][2];

            float grad_z = (view_project_matrix[batch_id][2][0] * repc_hom_w - view_project_matrix[batch_id][2][3] * mul1) * grad_ndc_pos[batch_id][index][0]
                + (view_project_matrix[batch_id][2][1] * repc_hom_w - view_project_matrix[batch_id][2][3] * mul2) * grad_ndc_pos[batch_id][index][1]
                + (view_project_matrix[batch_id][2][2] * repc_hom_w - view_project_matrix[batch_id][2][3] * mul3) * grad_ndc_pos[batch_id][index][2];

            grad_position[index][0] += grad_x;
            grad_position[index][1] += grad_y;
            grad_position[index][2] += grad_z;
        }
    }
}

at::Tensor world2ndc_backword(at::Tensor view_project_matrix, at::Tensor ndc_position, at::Tensor repc_hom_w, at::Tensor grad_ndcpos)
{


    int N = grad_ndcpos.size(0);
    int P = grad_ndcpos.size(1);
    at::Tensor d_position = torch::zeros({ P,4 }, grad_ndcpos.options());

    int threadsnum = 256;
    int blocknum=std::ceil(P / (float)threadsnum);

    world2ndc_backword_kernel << <blocknum, threadsnum >> > (
        view_project_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        ndc_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        repc_hom_w.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_ndcpos.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        d_position.packed_accessor32<float, 2, torch::RestrictPtrTraits>());


    return d_position;
}
template <typename scalar_t,int ROW,int COL>
__device__ void load_matrix(scalar_t(* __restrict__ dest)[ROW][COL], const torch::TensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, int32_t> source)
{
    
    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            (*dest)[i][j] = source[i][j];
        }
    }
}

template <typename scalar_t, int ROW, int COL>
__device__ void save_matrix(const scalar_t(*__restrict__ source)[ROW][COL], torch::TensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, int32_t> dest)
{

    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            dest[i][j]=(*source)[i][j];
        }
    }
}

template <typename scalar_t, int M, int N,int K, bool A_trans =false,bool B_trans =false>
__device__ void matmul(scalar_t(* __restrict__ A)[A_trans?M:K], scalar_t(* __restrict__ B)[B_trans?K:N], scalar_t(* __restrict__ output)[N])
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            scalar_t temp = 0.0;
            for (int k = 0; k < K; k++)
            {
                if(A_trans==false && B_trans==false)
                    temp+=A[i][k] * B[k][j];
                else if (A_trans == true && B_trans == false)
                    temp += A[k][i] * B[k][j];
                else if (A_trans == false && B_trans == true)
                    temp += A[i][k] * B[j][k];
                else if (A_trans == true && B_trans == true)
                    temp += A[k][i] * B[j][k];
            }
            output[i][j] = temp;
        }
    }
}

template <typename scalar_t, int M, int N>
__device__ void matmul_AtA(scalar_t(*__restrict__ A)[N], scalar_t(*__restrict__ output)[N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            scalar_t temp = 0.0;
            for (int k = 0; k < M; k++)
            {
                temp += A[k][i] * A[k][j];
            }
            output[i][j] = temp;
        }
    }
}

template <typename scalar_t>
__global__ void create_cov2d_forward(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jacobian_matrix,    //[batch,point_num,3,3] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> view_matrix,    //[batch,4,4] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> world_transform_matrix,    //[point_num,3,3] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cov2d         //[batch,point_num,2,2]
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    __shared__ scalar_t view[3][3];
    if (threadIdx.x < 9 && batch_id < view_matrix.size(0))
    {
        int row = threadIdx.x / 3;
        int col = threadIdx.x % 3;
        view[row][col] = view_matrix[batch_id][row][col];
    }
    __syncthreads();

    if (batch_id < view_matrix.size(0) && index < world_transform_matrix.size(0))
    {
        // world_transform_matrix @ view_matrix
        scalar_t T[3][3];
        scalar_t temp0[3][3];
        load_matrix<scalar_t, 3, 3>(&T, world_transform_matrix[index]);
        matmul<scalar_t, 3, 3, 3>(T, view, temp0);//world_transform_matrix@view_matrix

        scalar_t J[3][2];
        scalar_t temp1[3][2];
        load_matrix<scalar_t, 3, 2>(&J, jacobian_matrix[batch_id][index]);
        matmul<scalar_t, 3, 2, 3>(temp0, J, temp1);//(world_transform_matrix@view_matrix)@jacobian_matrix

        scalar_t result[2][2];
        matmul_AtA<scalar_t, 3, 2>(temp1, result);//A.trans@A

        save_matrix<scalar_t, 2, 2>(&result, cov2d[batch_id][index]);
    }
}


at::Tensor createCov2dDirectly_forward(
    at::Tensor J, //N,P,3,3
    at::Tensor view_matrix, //N,1,4,4
    at::Tensor transform_matrix //P,3,3
)
{
    int N = view_matrix.size(0);
    int P = transform_matrix.size(0);
    assert(J.size(0) == N);
    assert(J.size(1) == P);
    at::Tensor cov2d = torch::zeros({ N,P,2,2 }, transform_matrix.options());

    int threadsnum = 1024;
    dim3 Block3d(std::ceil(P / (float)threadsnum), N, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(transform_matrix.type(), __FUNCTION__, [&] {
        create_cov2d_forward<scalar_t> << <Block3d, threadsnum >> > (
            J.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            transform_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            cov2d.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });

    /*create_cov2d_forward<float> << <Block3d, threadsnum >> > (
        J.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        transform_matrix.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        cov2d.packed_accessor32<float, 4, torch::RestrictPtrTraits>());*/

    CUDA_CHECK_ERRORS;
    return cov2d;

}

template <typename scalar_t>
__global__ void create_cov2d_backward(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cov2d_grad,    //[batch,point_num,2,2] 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jacobian_matrix,    //[batch,point_num,3,3] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> view_matrix,    //[batch,4,4] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> world_transform_matrix,    //[point_num,3,3] 
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> transform_matrix_grad         //[point_num,3,3]
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ scalar_t view[3][3];
    scalar_t dL_dTrans_sum[3][3] = {0};
    for (int batch_id = 0; batch_id < view_matrix.size(0); batch_id++)
    {
        if (threadIdx.x < 9 )
        {
            int row = threadIdx.x / 3;
            int col = threadIdx.x % 3;
            view[row][col] = view_matrix[batch_id][row][col];
        }
        __syncthreads();

        if ( index < world_transform_matrix.size(0))
        {
            scalar_t view_rayspace_transform[3][2];
            scalar_t rayspace_transform[3][2];
            load_matrix<scalar_t, 3, 2>(&rayspace_transform, jacobian_matrix[batch_id][index]);
            matmul<scalar_t, 3, 2, 3>(view, rayspace_transform, view_rayspace_transform);
            scalar_t world_transform[3][3];
            load_matrix<scalar_t, 3, 3>(&world_transform, world_transform_matrix[index]);

            scalar_t T[3][2];
            matmul<scalar_t, 3, 2, 3>(world_transform, view_rayspace_transform, T);

            // cov2d_grad is symmetric.Gradient calculation can be simplified.
            // dL/dT=2 * T@cov2d_grad
            scalar_t dL_dCov2d[2][2];
            scalar_t dL_dT[3][2];
            load_matrix<scalar_t, 2, 2>(&dL_dCov2d, cov2d_grad[batch_id][index]);
            matmul<scalar_t, 3, 2, 2>(T, dL_dCov2d, dL_dT);
            dL_dT[0][0] *= 2; dL_dT[0][1] *= 2;
            dL_dT[1][0] *= 2; dL_dT[1][1] *= 2;
            dL_dT[2][0] *= 2; dL_dT[2][1] *= 2;

            //dL/dtransform = dL_dT@view_rayspace_transform.transpose()
            scalar_t dL_dTrans[3][3];
            matmul<scalar_t, 3, 3, 2, false, true>(dL_dT, view_rayspace_transform, dL_dTrans);
            dL_dTrans_sum[0][0] += dL_dTrans[0][0];
            dL_dTrans_sum[0][1] += dL_dTrans[0][1];
            dL_dTrans_sum[0][2] += dL_dTrans[0][2];
            dL_dTrans_sum[1][0] += dL_dTrans[1][0];
            dL_dTrans_sum[1][1] += dL_dTrans[1][1];
            dL_dTrans_sum[1][2] += dL_dTrans[1][2];
            dL_dTrans_sum[2][0] += dL_dTrans[2][0];
            dL_dTrans_sum[2][1] += dL_dTrans[2][1];
            dL_dTrans_sum[2][2] += dL_dTrans[2][2];

        }
        __syncthreads();
    }

    if (index < world_transform_matrix.size(0))
    {
        save_matrix<scalar_t, 3, 3>(&dL_dTrans_sum, transform_matrix_grad[index]);
    }
}

at::Tensor createCov2dDirectly_backward(
    at::Tensor cov2d_grad, //N,P,2,2
    at::Tensor J, //N,P,3,3
    at::Tensor view_matrix, //N,1,4,4
    at::Tensor transform_matrix //P,3,3
)
{
    int N = view_matrix.size(0);
    int P = transform_matrix.size(0);
    assert(cov2d_grad.size(0) == N);
    assert(cov2d_grad.size(1) == P);
    at::Tensor transform_matrix_grad = torch::zeros({ P,3,3 }, cov2d_grad.options());

    int threadsnum = 1024;
    int blocknum=std::ceil(P / (float)threadsnum);


    create_cov2d_backward<float> << <blocknum, threadsnum >> > (
        cov2d_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        J.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        transform_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        transform_matrix_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
    return transform_matrix_grad;

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

template <typename scalar_t,int degree>
__global__ void sh2rgb_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SHs,    //[point_num,(deg + 1) ** 2,3] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dirs,    //[batch,point_num,3] 
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rgb         //[batch,point_num,3]
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    if (batch_id < rgb.size(0) && index < rgb.size(1))
    {
        torch::TensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, int32_t> sh = SHs[index];

        float3 result;
        result.x = SH_C0 * sh[0][0];
        result.y = SH_C0 * sh[0][1];
        result.z = SH_C0 * sh[0][2];
        if (degree > 0)
        {
            float x = dirs[batch_id][index][0];
            float y = dirs[batch_id][index][1];
            float z = dirs[batch_id][index][2];
            result.x = result.x - SH_C1 * y * sh[1][0] + SH_C1 * z * sh[2][0] - SH_C1 * x * sh[3][0];
            result.y = result.y - SH_C1 * y * sh[1][1] + SH_C1 * z * sh[2][1] - SH_C1 * x * sh[3][1];
            result.z = result.z - SH_C1 * y * sh[1][2] + SH_C1 * z * sh[2][2] - SH_C1 * x * sh[3][2];

            if (degree > 1)
            {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                result.x = result.x + 
                    SH_C2[0] * xy * sh[4][0] +
                    SH_C2[1] * yz * sh[5][0] +
                    SH_C2[2] * (2.0f * zz - xx - yy) * sh[6][0] +
                    SH_C2[3] * xz * sh[7][0] +
                    SH_C2[4] * (xx - yy) * sh[8][0];
                result.y = result.y +
                    SH_C2[0] * xy * sh[4][1] +
                    SH_C2[1] * yz * sh[5][1] +
                    SH_C2[2] * (2.0f * zz - xx - yy) * sh[6][1] +
                    SH_C2[3] * xz * sh[7][1] +
                    SH_C2[4] * (xx - yy) * sh[8][1];
                result.z = result.z +
                    SH_C2[0] * xy * sh[4][2] +
                    SH_C2[1] * yz * sh[5][2] +
                    SH_C2[2] * (2.0f * zz - xx - yy) * sh[6][2] +
                    SH_C2[3] * xz * sh[7][2] +
                    SH_C2[4] * (xx - yy) * sh[8][2];

                if (degree > 2)
                {
                    result.x = result.x +
                        SH_C3[0] * y * (3.0f * xx - yy) * sh[9][0] +
                        SH_C3[1] * xy * z * sh[10][0] +
                        SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11][0] +
                        SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12][0] +
                        SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13][0] +
                        SH_C3[5] * z * (xx - yy) * sh[14][0] +
                        SH_C3[6] * x * (xx - 3.0f * yy) * sh[15][0];
                    result.y = result.y +
                        SH_C3[0] * y * (3.0f * xx - yy) * sh[9][1] +
                        SH_C3[1] * xy * z * sh[10][1] +
                        SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11][1] +
                        SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12][1] +
                        SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13][1] +
                        SH_C3[5] * z * (xx - yy) * sh[14][1] +
                        SH_C3[6] * x * (xx - 3.0f * yy) * sh[15][1];
                    result.z = result.z +
                        SH_C3[0] * y * (3.0f * xx - yy) * sh[9][2] +
                        SH_C3[1] * xy * z * sh[10][2] +
                        SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11][2] +
                        SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12][2] +
                        SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13][2] +
                        SH_C3[5] * z * (xx - yy) * sh[14][2] +
                        SH_C3[6] * x * (xx - 3.0f * yy) * sh[15][2];
                }
            }

        }
        result.x += 0.5f;
        result.y += 0.5f;
        result.z += 0.5f;
        rgb[batch_id][index][0] = result.x;
        rgb[batch_id][index][1] = result.y;
        rgb[batch_id][index][2] = result.z;
    }
}

at::Tensor sh2rgb_forward(int64_t degree, at::Tensor sh, at::Tensor dir)
{
    int N = dir.size(0);
    int P = dir.size(1);
    at::Tensor rgb = torch::zeros({ N,P,3 }, sh.options());

    int threadsnum = 1024;
    dim3 Block3d(std::ceil(P / (float)threadsnum), N, 1);

    switch (degree)
    {
    case 0:
        sh2rgb_forward_kernel<float, 0> << <Block3d, threadsnum >> > (
            sh.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 1:
        sh2rgb_forward_kernel<float, 1> << <Block3d, threadsnum >> > (
            sh.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 2:
        sh2rgb_forward_kernel<float, 2> << <Block3d, threadsnum >> > (
            sh.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 3:
        sh2rgb_forward_kernel<float, 3> << <Block3d, threadsnum >> > (
            sh.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    default:
        ;
    }

    

    CUDA_CHECK_ERRORS;
    return rgb;
}



template <typename scalar_t, int degree>
__global__ void sh2rgb_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dirs,    //[batch,point_num,3] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rgb_grad,         //[batch,point_num,3]
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SHs_grad   //[point_num,(deg + 1) ** 2,3] 
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int batch_id = 0; batch_id < rgb_grad.size(0); batch_id++)
    {
        if ( index < rgb_grad.size(1))
        {
            torch::TensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, int32_t> dL_dsh = SHs_grad[index];
            float3 dL_dRGB{ rgb_grad[batch_id][index][0], rgb_grad[batch_id][index][1], rgb_grad[batch_id][index][2] };

            float dRGBdsh0 = SH_C0;
            dL_dsh[0][0] += dRGBdsh0 * dL_dRGB.x;
            dL_dsh[0][1] += dRGBdsh0 * dL_dRGB.y;
            dL_dsh[0][2] += dRGBdsh0 * dL_dRGB.z;

            if (degree > 0)
            {
                float x = dirs[batch_id][index][0];
                float y = dirs[batch_id][index][1];
                float z = dirs[batch_id][index][2];

                float dRGBdsh1 = -SH_C1 * y;
                float dRGBdsh2 = SH_C1 * z;
                float dRGBdsh3 = -SH_C1 * x;
                dL_dsh[1][0] += dRGBdsh1 * dL_dRGB.x;
                dL_dsh[2][0] += dRGBdsh2 * dL_dRGB.x;
                dL_dsh[3][0] += dRGBdsh3 * dL_dRGB.x;
                dL_dsh[1][1] += dRGBdsh1 * dL_dRGB.y;
                dL_dsh[2][1] += dRGBdsh2 * dL_dRGB.y;
                dL_dsh[3][1] += dRGBdsh3 * dL_dRGB.y;
                dL_dsh[1][2] += dRGBdsh1 * dL_dRGB.z;
                dL_dsh[2][2] += dRGBdsh2 * dL_dRGB.z;
                dL_dsh[3][2] += dRGBdsh3 * dL_dRGB.z;

                if (degree > 1)
                {
                    float xx = x * x, yy = y * y, zz = z * z;
                    float xy = x * y, yz = y * z, xz = x * z;

                    float dRGBdsh4 = SH_C2[0] * xy;
                    float dRGBdsh5 = SH_C2[1] * yz;
                    float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
                    float dRGBdsh7 = SH_C2[3] * xz;
                    float dRGBdsh8 = SH_C2[4] * (xx - yy);

                    dL_dsh[4][0] += dRGBdsh4 * dL_dRGB.x;
                    dL_dsh[5][0] += dRGBdsh5 * dL_dRGB.x;
                    dL_dsh[6][0] += dRGBdsh6 * dL_dRGB.x;
                    dL_dsh[7][0] += dRGBdsh7 * dL_dRGB.x;
                    dL_dsh[8][0] += dRGBdsh8 * dL_dRGB.x;
                    dL_dsh[4][1] += dRGBdsh4 * dL_dRGB.y;
                    dL_dsh[5][1] += dRGBdsh5 * dL_dRGB.y;
                    dL_dsh[6][1] += dRGBdsh6 * dL_dRGB.y;
                    dL_dsh[7][1] += dRGBdsh7 * dL_dRGB.y;
                    dL_dsh[8][1] += dRGBdsh8 * dL_dRGB.y;
                    dL_dsh[4][2] += dRGBdsh4 * dL_dRGB.z;
                    dL_dsh[5][2] += dRGBdsh5 * dL_dRGB.z;
                    dL_dsh[6][2] += dRGBdsh6 * dL_dRGB.z;
                    dL_dsh[7][2] += dRGBdsh7 * dL_dRGB.z;
                    dL_dsh[8][2] += dRGBdsh8 * dL_dRGB.z;

                    if (degree > 2)
                    {
                        float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                        float dRGBdsh10 = SH_C3[1] * xy * z;
                        float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                        float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                        float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                        float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                        float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                        dL_dsh[9][0] += dRGBdsh9 * dL_dRGB.x;
                        dL_dsh[10][0] += dRGBdsh10 * dL_dRGB.x;
                        dL_dsh[11][0] += dRGBdsh11 * dL_dRGB.x;
                        dL_dsh[12][0] += dRGBdsh12 * dL_dRGB.x;
                        dL_dsh[13][0] += dRGBdsh13 * dL_dRGB.x;
                        dL_dsh[14][0] += dRGBdsh14 * dL_dRGB.x;
                        dL_dsh[15][0] += dRGBdsh15 * dL_dRGB.x;
                        dL_dsh[9][1] += dRGBdsh9 * dL_dRGB.y;
                        dL_dsh[10][1] += dRGBdsh10 * dL_dRGB.y;
                        dL_dsh[11][1] += dRGBdsh11 * dL_dRGB.y;
                        dL_dsh[12][1] += dRGBdsh12 * dL_dRGB.y;
                        dL_dsh[13][1] += dRGBdsh13 * dL_dRGB.y;
                        dL_dsh[14][1] += dRGBdsh14 * dL_dRGB.y;
                        dL_dsh[15][1] += dRGBdsh15 * dL_dRGB.y;
                        dL_dsh[9][2] += dRGBdsh9 * dL_dRGB.z;
                        dL_dsh[10][2] += dRGBdsh10 * dL_dRGB.z;
                        dL_dsh[11][2] += dRGBdsh11 * dL_dRGB.z;
                        dL_dsh[12][2] += dRGBdsh12 * dL_dRGB.z;
                        dL_dsh[13][2] += dRGBdsh13 * dL_dRGB.z;
                        dL_dsh[14][2] += dRGBdsh14 * dL_dRGB.z;
                        dL_dsh[15][2] += dRGBdsh15 * dL_dRGB.z;
                    }
                }

            }
        }
    }
}

at::Tensor sh2rgb_backward(int64_t degree, at::Tensor rgb_grad, int64_t sh_dim, at::Tensor dir)
{
    int N = rgb_grad.size(0);
    int P = rgb_grad.size(1);
    int C = rgb_grad.size(2);
    at::Tensor sh_grad = torch::zeros({P,sh_dim ,C}, rgb_grad.options());

    int threadsnum = 256;
    int blocknum=std::ceil(P / (float)threadsnum);

    switch (degree)
    {
    case 0:
        sh2rgb_backward_kernel<float, 0> << <blocknum, threadsnum >> > (
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 1:
        sh2rgb_backward_kernel<float, 1> << <blocknum, threadsnum >> > (
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 2:
        sh2rgb_backward_kernel<float, 2> << <blocknum, threadsnum >> > (
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 3:
        sh2rgb_backward_kernel<float, 3> << <blocknum, threadsnum >> > (
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    default:
        ;
    }



    CUDA_CHECK_ERRORS;
    return sh_grad;
}


template <typename scalar_t>
__global__ void eigh_and_inv_2x2matrix_kernel_forward(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,    //[batch,point_num,2,2] 
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> val,   //[batch,point_num,2] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> vec,   //[batch,point_num,2,2] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> inv   //[batch,point_num,2,2] 
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    if (batch_id < input.size(0) && index < input.size(1))
    {
        float input_matrix[2][2] = { {input[batch_id][index][0][0],input[batch_id][index][0][1]},{input[batch_id][index][1][0],input[batch_id][index][1][1]}};
        float det = input_matrix[0][0] * input_matrix[1][1] - input_matrix[0][1] * input_matrix[1][0];
        float mid = 0.5f * (input_matrix[0][0] + input_matrix[1][1]);
        float temp = sqrt(max(mid * mid - det,1e-9f));

        val[batch_id][index][0] = mid - temp;
        val[batch_id][index][1] = mid + temp;

        float vec_y_0 = ((mid - temp) - input_matrix[0][0]) / input_matrix[0][1];
        float vec_y_1 = ((mid + temp) - input_matrix[0][0]) / input_matrix[0][1];

        float square_sum_0_recip = 1/sqrt(1 + vec_y_0 * vec_y_0);
        float square_sum_1_recip = 1/sqrt(1 + vec_y_1 * vec_y_1);

        vec[batch_id][index][0][0] = square_sum_0_recip; vec[batch_id][index][0][1] = vec_y_0 * square_sum_0_recip;
        vec[batch_id][index][1][0] = square_sum_1_recip; vec[batch_id][index][1][1] = vec_y_1 * square_sum_1_recip;
        
        float det_recip = 1 / det;
        inv[batch_id][index][0][1] = -input_matrix[0][1] * det_recip;
        inv[batch_id][index][1][0] = -input_matrix[1][0] * det_recip;
        inv[batch_id][index][0][0] = input_matrix[1][1] * det_recip;
        inv[batch_id][index][1][1] = input_matrix[0][0] * det_recip;

    }

}


template <typename scalar_t>
__global__ void inv_2x2matrix_kernel_backward(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> Invmatrix,    //[batch,point_num,2,2] 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dInvmatrix,    //[batch,point_num,2,2] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dMatrix   //[batch,point_num,2,2] 
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    if (batch_id < Invmatrix.size(0) && index < Invmatrix.size(1))
    {
        scalar_t inv_matrix[2][2];
        scalar_t dl_dinvmatrix[2][2];
        scalar_t temp[2][2];
        scalar_t dl_dmatrix[2][2];

        load_matrix<scalar_t, 2, 2>(&inv_matrix, Invmatrix[batch_id][index]);
        load_matrix<scalar_t, 2, 2>(&dl_dinvmatrix, dL_dInvmatrix[batch_id][index]);

        matmul<scalar_t, 2, 2, 2>(inv_matrix, dl_dinvmatrix, temp);
        matmul<scalar_t, 2, 2, 2>(temp, inv_matrix, dl_dmatrix);

        dl_dmatrix[0][0] = -dl_dmatrix[0][0]; dl_dmatrix[0][1] = -dl_dmatrix[0][1];
        dl_dmatrix[1][0] = -dl_dmatrix[1][0]; dl_dmatrix[1][1] = -dl_dmatrix[1][1];

        save_matrix<scalar_t, 2, 2>(&dl_dmatrix, dL_dMatrix[batch_id][index]);
    }

}

std::vector<at::Tensor> eigh_and_inv_2x2matrix_forward(at::Tensor input)
{
    int N = input.size(0);
    int P = input.size(1);
    at::Tensor vec = torch::zeros({ N,P,2,2 }, input.options().requires_grad(false));
    at::Tensor val = torch::zeros({ N,P,2 }, input.options().requires_grad(false));
    at::Tensor inv = torch::zeros({ N,P,2,2 }, input.options());

    int threadsnum = 1024;
    dim3 Block3d(std::ceil(P / (float)threadsnum), N, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), __FUNCTION__, [&] {eigh_and_inv_2x2matrix_kernel_forward<scalar_t > << <Block3d, threadsnum >> > (
        input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        val.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        vec.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        inv.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });

    return { val,vec,inv };
}

at::Tensor inv_2x2matrix_backward(at::Tensor inv_matrix,at::Tensor dL_dInvMatrix)
{
    int N = inv_matrix.size(0);
    int P = inv_matrix.size(1);
    at::Tensor dL_dMatrix = torch::zeros_like(dL_dInvMatrix);

    int threadsnum = 1024;
    dim3 Block3d(std::ceil(P / (float)threadsnum), N, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(inv_matrix.type(), __FUNCTION__, [&] {inv_2x2matrix_kernel_backward<scalar_t > << <Block3d, threadsnum >> > (
        inv_matrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        dL_dInvMatrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        dL_dMatrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });

    return dL_dMatrix;

}