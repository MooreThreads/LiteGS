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
#include "raster.h"


template <int tilesize>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,2,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tiles,          //[batch,tiles_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output_transmitance,    //[batch,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    int tiles_num_x,int img_h,int img_w
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
        if (start_index_in_tile != -1)
        {
            for (int offset = start_index_in_tile; offset < end_index_in_tile; offset += tilesize * tilesize / 4)
            {
                int num_done = __syncthreads_count(done);
                if (num_done == blockDim.x * blockDim.y)
                    break;

                int valid_num = min(tilesize * tilesize / 4, end_index_in_tile - offset);
                //load to shared memory
                if (threadIdx.y * blockDim.x + threadIdx.x < valid_num)
                {
                    int i = threadIdx.y * blockDim.x + threadIdx.x;
                    int index = offset + i;
                    int point_id = sorted_points[batch_id][index];
                    collected_xy[i].x = (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f;
                    collected_xy[i].y = (ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f;
                    collected_cov2d_inv[i].x = cov2d_inv[batch_id][0][0][point_id];
                    collected_cov2d_inv[i].y = cov2d_inv[batch_id][0][1][point_id];
                    collected_cov2d_inv[i].z = cov2d_inv[batch_id][1][1][point_id];

                    collected_color[i].x = color[batch_id][0][point_id];
                    collected_color[i].y = color[batch_id][1][point_id];
                    collected_color[i].z = color[batch_id][2][point_id];
                    collected_opacity[i] = opacity[0][point_id];
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

                    if (transmittance * (1 - alpha) < 0.0001f)
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
    at::Tensor  ndc,// 
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
    at::DeviceGuard guard( ndc.device());

    int64_t viewsnum = start_index.sizes()[0];
    int64_t tilesnum = tiles.sizes()[1];

    std::vector<int64_t> shape_img{ viewsnum,3, tilesnum,tilesize,tilesize };
    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_img = torch::empty(shape_img, opt_img);

    std::vector<int64_t> shape_t{ viewsnum, tilesnum, tilesize, tilesize };
    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_transmitance = torch::empty(shape_t, opt_t);

    std::vector<int64_t> shape_c{ viewsnum, tilesnum, tilesize, tilesize };
    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(start_index.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty(shape_c, opt_c);



    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);
    switch (tilesize)
    {
    case 8:
        raster_forward_kernel<8> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x,img_h,img_w);
        break;
    case 16:
        raster_forward_kernel<16> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x,img_h,img_w);
        break;
    case 32:
        raster_forward_kernel<32> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x,img_h,img_w);
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
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,2,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> final_transmitance,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,2,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w
)
{
    const int tilesize = 8;
    extern __shared__ float shared_buffer[];
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    const int batch_id = blockIdx.y;
    const int tile_index = blockIdx.x * (blockDim.x / (tilesize * tilesize)) + threadIdx.x / (tilesize * tilesize);
    int tile_id = 0;
    if (tile_index < tiles.size(1))
        tile_id = tiles[batch_id][tile_index];


    if (tile_id != 0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];
        if (start_index_in_tile != -1)
        {
            int pixel_id= threadIdx.x % (tilesize * tilesize);
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
                    float2 xy;
                    xy.x = (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f;
                    xy.y = (ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f;
                    float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                    float4 cur_color{ color[batch_id][0][point_id],color[batch_id][1][point_id],color[batch_id][2][point_id],opacity[0][point_id] };
                    float3 cur_cov2d_inv{ cov2d_inv[batch_id][0][0][point_id],cov2d_inv[batch_id][0][1][point_id],cov2d_inv[batch_id][1][1][point_id] };


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
                        atomicAdd(&d_color[batch_id][0][point_id], grad_color.x);
                        atomicAdd(&d_color[batch_id][1][point_id], grad_color.y);
                        atomicAdd(&d_color[batch_id][2][point_id], grad_color.z);

                        atomicAdd(&d_cov2d_inv[batch_id][0][0][point_id], grad_invcov.x);
                        atomicAdd(&d_cov2d_inv[batch_id][0][1][point_id], grad_invcov.y);
                        atomicAdd(&d_cov2d_inv[batch_id][1][0][point_id], grad_invcov.y);
                        atomicAdd(&d_cov2d_inv[batch_id][1][1][point_id], grad_invcov.z);

                        atomicAdd(&d_ndc[batch_id][0][point_id], grad_mean.x * 0.5f * img_w);
                        atomicAdd(&d_ndc[batch_id][1][point_id], grad_mean.y * 0.5f * img_h);

                        atomicAdd(&d_opacity[0][point_id], grad_opacity);
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
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,2,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> final_transmitance,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,2,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w
)
{
    const int property_num = 9;
    __shared__ float gradient_buffer[tilesize * tilesize * property_num];
    float* const grad_color_x = gradient_buffer;
    float* const grad_color_y = gradient_buffer + 1 * tilesize * tilesize;
    float* const grad_color_z = gradient_buffer + 2 * tilesize * tilesize;
    float* const grad_invcov_x = gradient_buffer + 3 * tilesize * tilesize;
    float* const grad_invcov_y = gradient_buffer + 4 * tilesize * tilesize;
    float* const grad_invcov_z = gradient_buffer + 5 * tilesize * tilesize;
    float* const grad_mean_x = gradient_buffer + 6 * tilesize * tilesize;
    float* const grad_mean_y = gradient_buffer + 7 * tilesize * tilesize;
    float* const grad_opacity = gradient_buffer + 8 * tilesize * tilesize;
    __shared__ float* global_grad_addr[property_num];

    const int batch_id = blockIdx.y;
    int tile_id = tiles[batch_id][blockIdx.x];
    auto block = cg::this_thread_block();
    auto cuda_tile = cg::tiled_partition<32>(block);

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id - 1) / tiles_num_x) * tilesize + y_in_tile;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        global_grad_addr[0] = &d_color[batch_id][0][0];
        global_grad_addr[1] = &d_color[batch_id][1][0];
        global_grad_addr[2] = &d_color[batch_id][2][0];

        global_grad_addr[3] = &d_cov2d_inv[batch_id][0][0][0];
        global_grad_addr[4] = &d_cov2d_inv[batch_id][0][1][0];
        global_grad_addr[5] = &d_cov2d_inv[batch_id][1][1][0];

        global_grad_addr[6] = &d_ndc[batch_id][0][0];
        global_grad_addr[7] = &d_ndc[batch_id][1][0];

        global_grad_addr[8] = &d_opacity[0][0];
    }
    __syncthreads();

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
        for (int index = end_index_in_tile-1; index >= start_index_in_tile; index --)
        {
            bool bSkip = false;
            int threadidx = threadIdx.y * blockDim.x + threadIdx.x;
            int point_id= sorted_points[batch_id][index];
            if (index > pixel_lst_index)
            {
                bSkip = true;
            }
            else
            {

                float2 xy{ (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f ,(ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f };
                float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                float4 cur_color{ color[batch_id][0][point_id],color[batch_id][1][point_id],color[batch_id][2][point_id],opacity[0][point_id] };
                float3 cur_cov2d_inv{ cov2d_inv[batch_id][0][0][point_id] ,cov2d_inv[batch_id][0][1][point_id],cov2d_inv[batch_id][1][1][point_id] };

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
                    grad_mean_x[threadidx] = d_deltax * 0.5f * img_w;
                    grad_mean_y[threadidx] = d_deltay * 0.5f * img_h;
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

            //reduction
            bool block_skip=__syncthreads_and(bSkip);
            if (block_skip==false)
            {
                int warps_num = cuda_tile.meta_group_size();
                if (warps_num > property_num)
                {
                    int property_id = cuda_tile.meta_group_rank();
                    if (property_id < property_num)
                    {
                        float gradient_sum = 0;
                        for (int reduction_i = 0; reduction_i < tilesize * tilesize; reduction_i += cuda_tile.num_threads())
                        {
                            gradient_sum += gradient_buffer[property_id * tilesize * tilesize + reduction_i + cuda_tile.thread_rank()];
                        }
                        gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, 16);
                        gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, 8);
                        gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, 4);
                        gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, 2);
                        gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, 1);
                        if (cuda_tile.thread_rank() == 0)
                        {
                            atomicAdd(global_grad_addr[property_id] + point_id, gradient_sum);
                            if (property_id == 4)
                            {
                                atomicAdd(&d_cov2d_inv[batch_id][1][0][point_id], gradient_sum);
                            }
                        }
                    }
                }
                else
                {
                    int property_offset = 0;
                    for (; property_offset+8 < property_num; property_offset += 8)
                    {
                        int property_per_warp=8 / cuda_tile.meta_group_size();
                        int threads_per_property = cuda_tile.num_threads() / property_per_warp;
                        int property_id = threadidx / threads_per_property + property_offset;
                        int pixel_offset = threadidx % threads_per_property;
                        if (property_id < property_num)
                        {
                            float gradient_sum = 0;
                            for (int i = 0; i < tilesize * tilesize; i += threads_per_property)
                            {
                                gradient_sum += gradient_buffer[property_id * tilesize * tilesize + i + pixel_offset];
                            }
                            for (int i = threads_per_property/2; i >= 1; i /= 2)
                            {
                                gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, i); 
                            }
                            if (pixel_offset == 0)
                            {
                                atomicAdd(global_grad_addr[property_id] + point_id, gradient_sum);
                                if (property_id == 4)
                                {
                                    atomicAdd(&d_cov2d_inv[batch_id][1][0][point_id], gradient_sum);
                                }
                            }
                        }
                    }

                    if (threadidx < tilesize * tilesize)
                    {
                        for (; property_offset < property_num; property_offset++)
                        {
                            float gradient_sum = gradient_buffer[property_offset * tilesize * tilesize + threadidx];
                            for (int i = cuda_tile.num_threads() / 2; i >= 1; i /= 2)
                            {
                                gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, i);
                            }
                            if (cuda_tile.thread_rank()==0)
                            {
                                atomicAdd(global_grad_addr[property_offset] + point_id, gradient_sum);
                            }
                        }
                    }

                }
                
            }
            __syncthreads();

        }
        
    }
}


std::vector<at::Tensor> rasterize_backward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor ndc,// 
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
    at::DeviceGuard guard(ndc.device());

    int64_t viewsnum = start_index.sizes()[0];
    int64_t tilesnum = tiles.sizes()[1];

    at::Tensor d_ndc = torch::zeros_like(ndc, ndc.options());
    at::Tensor d_cov2d_inv = torch::zeros_like(cov2d_inv, ndc.options());
    at::Tensor d_color = torch::zeros_like(color, ndc.options());
    at::Tensor d_opacity = torch::zeros_like(opacity, ndc.options());

    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);

    int TilesNumInBlock = 1;
    dim3 Block3d8x8((tilesnum + TilesNumInBlock - 1) / TilesNumInBlock, viewsnum, 1);
    int ThreadsNum8x8 = 64 * TilesNumInBlock;
    
    switch (tilesize)
    {
    case 8:
        //todo cuda perfer shared
        //raster_backward_kernel_8x8 << <Block3d8x8, ThreadsNum8x8 >> > (
        raster_backward_kernel<8> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w);
        break;
    case 16:
        raster_backward_kernel<16> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w);
        break;
    case 32:
        raster_backward_kernel<32> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w);
        break;
    default:
        ;
    }
    CUDA_CHECK_ERRORS;
    
    return { d_ndc ,d_cov2d_inv ,d_color,d_opacity };
}

