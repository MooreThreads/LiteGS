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


template <int tilesize,bool enable_depth>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_depth,     //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    int tiles_num_x,int img_h,int img_w
)
{


    __shared__ float2 collected_xy[tilesize * tilesize / 4];
    __shared__ float collected_depth[tilesize * tilesize / 4];
    __shared__ float collected_opacity[tilesize * tilesize / 4];
    __shared__ float3 collected_cov2d_inv[tilesize * tilesize / 4];
    __shared__ float3 collected_color[tilesize * tilesize / 4];

    const int batch_id = blockIdx.y;
    int tile_id = blockIdx.x + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0)
    {
        tile_id = specific_tiles[batch_id][blockIdx.x];
    }

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id-1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id-1) / tiles_num_x) * tilesize + y_in_tile;

    if (tile_id!=0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        float transmittance = 1.0f;
        float inv_depth = 0.0f;
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
                    if (enable_depth)
                    {
                        collected_depth[i] = ndc[batch_id][2][point_id];
                    }
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
                    if (enable_depth)
                    {
                        inv_depth += collected_depth[i] * alpha * transmittance;
                    }
                    transmittance *= (1 - alpha);
                    last_contributor = offset + i;


                }
                __syncthreads();
            }
        }
        output_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile] = final_color.x;
        output_img[batch_id][1][blockIdx.x][y_in_tile][x_in_tile] = final_color.y;
        output_img[batch_id][2][blockIdx.x][y_in_tile][x_in_tile] = final_color.z;
        output_transmitance[batch_id][0][blockIdx.x][y_in_tile][x_in_tile] = transmittance;
        if (enable_depth)
        {
            output_depth[batch_id][0][blockIdx.x][y_in_tile][x_in_tile] = inv_depth;
        }

        output_last_contributor[batch_id][blockIdx.x][y_in_tile][x_in_tile] = last_contributor;
        
    }
}

std::vector<at::Tensor> rasterize_RGBA_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor  ndc,// 
    at::Tensor  cov2d_inv,
    at::Tensor  color,
    at::Tensor  opacity,
    std::optional<at::Tensor>  specific_tiles_arg,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w
)
{
    at::DeviceGuard guard( ndc.device());

    int64_t viewsnum = start_index.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tilesize));
    int tilesnum_y = std::ceil(img_h / float(tilesize));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    at::Tensor specific_tiles;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        tilesnum = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
    }
    

    std::vector<int64_t> shape_img{ viewsnum,3, tilesnum,tilesize,tilesize };
    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_img = torch::empty(shape_img, opt_img);

    std::vector<int64_t> shape_t{ viewsnum,1, tilesnum, tilesize, tilesize };
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
        raster_forward_kernel<8,false> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x,img_h,img_w);
        break;
    case 16:
        raster_forward_kernel<16, false> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x,img_h,img_w);
        break;
    default:
        assert(false);

    }
    CUDA_CHECK_ERRORS;
    


    return { output_img ,output_transmitance ,output_last_contributor };
}

std::vector<at::Tensor> rasterize_RGBAD_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor  ndc,// 
    at::Tensor  cov2d_inv,
    at::Tensor  color,
    at::Tensor  opacity,
    std::optional<at::Tensor>  specific_tiles_arg,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w
)
{
    at::DeviceGuard guard(ndc.device());

    int64_t viewsnum = start_index.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tilesize));
    int tilesnum_y = std::ceil(img_h / float(tilesize));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    at::Tensor specific_tiles;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        tilesnum = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
    }


    std::vector<int64_t> shape_img{ viewsnum,3, tilesnum,tilesize,tilesize };
    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_img = torch::empty(shape_img, opt_img);

    std::vector<int64_t> shape_t{ viewsnum,1, tilesnum, tilesize, tilesize };
    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_transmitance = torch::empty(shape_t, opt_t);
    at::Tensor output_depth = torch::empty(shape_t, opt_t);

    std::vector<int64_t> shape_c{ viewsnum, tilesnum, tilesize, tilesize };
    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(start_index.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty(shape_c, opt_c);



    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);
    switch (tilesize)
    {
    case 8:
        raster_forward_kernel<8, true> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x, img_h, img_w);
        break;
    case 16:
        raster_forward_kernel<16, true> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tilesnum_x, img_h, img_w);
        break;
    default:
        assert(false);

    }
    CUDA_CHECK_ERRORS;



    return { output_img ,output_transmitance,output_depth ,output_last_contributor };
}

template <int tilesize>
__global__ void raster_backward_kernel_warp_reduction(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,2,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w
)
{
    __shared__ float4 collected_color[tilesize * tilesize];
    __shared__ float3 collected_invcov[tilesize * tilesize];
    __shared__ float2 collected_mean[tilesize * tilesize];

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int threadidx = threadIdx.y * blockDim.x + threadIdx.x;
    const int x_in_tile = (warp.meta_group_rank() % (tilesize / 8)) * 8 + warp.thread_rank() % 8;
    const int y_in_tile = (warp.meta_group_rank() / (tilesize / 8)) * 4 + warp.thread_rank() / 8;

    const int batch_id = blockIdx.y;
    const int tile_index = blockIdx.x;
    int tile_id = blockIdx.x + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0)
    {
        tile_id = specific_tiles[batch_id][blockIdx.x];
    }


    if (tile_id != 0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];
        if (start_index_in_tile != -1)
        {
            float transmittance = final_transmitance[batch_id][0][tile_index][y_in_tile][x_in_tile];
            int pixel_lst_index = last_contributor[batch_id][tile_index][y_in_tile][x_in_tile];

            float3 d_pixel{ 0,0,0 };
            int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + x_in_tile;
            int pixel_y = ((tile_id - 1) / tiles_num_x) * tilesize + y_in_tile;
            if (pixel_x < img_w && pixel_y < img_h)
            {
                d_pixel.x = d_img[batch_id][0][tile_index][y_in_tile][x_in_tile];
                d_pixel.y = d_img[batch_id][1][tile_index][y_in_tile][x_in_tile];
                d_pixel.z = d_img[batch_id][2][tile_index][y_in_tile][x_in_tile];
            }
            //loop points
            float3 accum_rec{ 0,0,0 };
            for (int offset = end_index_in_tile - 1; offset >= start_index_in_tile; offset -= (tilesize * tilesize))
            {
                int collected_num = min(tilesize * tilesize, offset - start_index_in_tile + 1);
                if (threadIdx.y * blockDim.x + threadIdx.x < collected_num)
                {
                    int index = offset - threadidx;
                    int point_id = sorted_points[batch_id][index];

                    collected_mean[threadidx].x = (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f;
                    collected_mean[threadidx].y = (ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f;
                    collected_invcov[threadidx].x = cov2d_inv[batch_id][0][0][point_id];
                    collected_invcov[threadidx].y = cov2d_inv[batch_id][0][1][point_id];
                    collected_invcov[threadidx].z = cov2d_inv[batch_id][1][1][point_id];
                    collected_color[threadidx].x = color[batch_id][0][point_id];
                    collected_color[threadidx].y = color[batch_id][1][point_id];
                    collected_color[threadidx].z = color[batch_id][2][point_id];
                    collected_color[threadidx].w = opacity[0][point_id];
                }
                __syncthreads();
                for (int i = 0; i < collected_num; i++)
                {
                    int point_index = offset - i;
                    bool skip = point_index > pixel_lst_index;
                    float3 grad_color{ 0,0,0 };
                    float3 grad_invcov{ 0,0,0 };
                    float2 grad_mean{ 0,0 };
                    float grad_opacity{ 0 };

                    if (skip == false)
                    {
                        float2 xy = collected_mean[i];
                        float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                        float4 cur_color = collected_color[i];
                        float3 cur_cov2d_inv = collected_invcov[i];


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
                            int point_id = sorted_points[batch_id][point_index];
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
                __syncthreads();
            }

            
        }
    }
}

__device__ int atomicAggInc(int* ptr)
{
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;

    // elect the first active thread to perform atomic add
    if (g.thread_rank() == 0) {
        prev = atomicAdd(ptr, g.size());
    }

    // broadcast previous value within the warp
    // and add each active thread’s rank to it
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}

template <int tilesize>
__global__ void raster_backward_kernel_multibatch_reduction(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,2,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w
)
{
    __shared__ float4 collected_color[tilesize * tilesize ];
    __shared__ float3 collected_invcov[tilesize * tilesize ];
    __shared__ float2 collected_mean[tilesize * tilesize ];

    constexpr int property_num = 9;
    constexpr int threadsnum_per_property = tilesize * tilesize / property_num;
    __shared__ float gradient_buffer[(tilesize * tilesize+ threadsnum_per_property) * property_num];//"+threadsnum_per_property" to avoid bank conflict
    float* const grad_color_x = gradient_buffer;
    float* const grad_color_y = gradient_buffer + 1 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_color_z = gradient_buffer + 2 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_x = gradient_buffer + 3 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_y = gradient_buffer + 4 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_z = gradient_buffer + 5 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_mean_x = gradient_buffer + 6 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_mean_y = gradient_buffer + 7 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_opacity = gradient_buffer + 8 * (tilesize * tilesize + threadsnum_per_property);
    __shared__ int valid_pix_num;

    const int batch_id = blockIdx.y;
    int tile_id = blockIdx.x + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0)
    {
        tile_id = specific_tiles[batch_id][blockIdx.x];
    }
    auto block = cg::this_thread_block();
    auto cuda_tile = cg::tiled_partition<32>(block);
    int threadidx = threadIdx.y * blockDim.x + threadIdx.x;

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id - 1) / tiles_num_x) * tilesize + y_in_tile;

    float* global_grad_addr = nullptr;
    switch (threadidx) {
    case 0:
        global_grad_addr = &d_color[batch_id][0][0];
        break;
    case 1:
        global_grad_addr = &d_color[batch_id][1][0];
        break;
    case 2:
        global_grad_addr = &d_color[batch_id][2][0];
        break;
    case 3:
        global_grad_addr = &d_cov2d_inv[batch_id][0][0][0];
        break;
    case 4:
        global_grad_addr = &d_cov2d_inv[batch_id][0][1][0];
        break;
    case 5:
        global_grad_addr = &d_cov2d_inv[batch_id][1][1][0];
        break;
    case 6:
        global_grad_addr = &d_ndc[batch_id][0][0];
        break;
    case 7:
        global_grad_addr = &d_ndc[batch_id][1][0];
        break;
    case 8:
        global_grad_addr = &d_opacity[0][0];
        break;
    default:
        break;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        valid_pix_num = 0;
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

        float transmittance = final_transmitance[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
        int pixel_lst_index = last_contributor[batch_id][blockIdx.x][y_in_tile][x_in_tile];
        float3 d_pixel{ 0,0,0 };
        if (pixel_x < img_w && pixel_y < img_h)
        {
            d_pixel.x = d_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
            d_pixel.y = d_img[batch_id][1][blockIdx.x][y_in_tile][x_in_tile];
            d_pixel.z = d_img[batch_id][2][blockIdx.x][y_in_tile][x_in_tile];
        }

        float3 accum_rec{ 0,0,0 };
        for (int offset = end_index_in_tile - 1; offset >= start_index_in_tile; offset -= (tilesize * tilesize ))
        {
            int collected_num = min(tilesize * tilesize , offset - start_index_in_tile + 1);
            if (threadIdx.y * blockDim.x + threadIdx.x < collected_num)
            {
                int index = offset - threadidx;
                int point_id = sorted_points[batch_id][index];

                collected_mean[threadidx].x = (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f;
                collected_mean[threadidx].y = (ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f;
                collected_invcov[threadidx].x = cov2d_inv[batch_id][0][0][point_id];
                collected_invcov[threadidx].y = cov2d_inv[batch_id][0][1][point_id];
                collected_invcov[threadidx].z = cov2d_inv[batch_id][1][1][point_id];
                collected_color[threadidx].x = color[batch_id][0][point_id];
                collected_color[threadidx].y = color[batch_id][1][point_id];
                collected_color[threadidx].z = color[batch_id][2][point_id];
                collected_color[threadidx].w = opacity[0][point_id];
            }
            __syncthreads();
            for (int i = 0; i < collected_num; i++)
            {
                int index = offset - i;
                bool bSkip = true;
                float alpha = 0.0f;
                float G = 0.0f;
                float2 xy = collected_mean[i];
                float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                float4 cur_color = collected_color[i];
                float3 cur_cov2d_inv = collected_invcov[i];
                if (index <= pixel_lst_index)
                {
                    float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                    G = exp(power);
                    alpha = min(0.99f, cur_color.w * G);
                    bSkip = !((power <= 0.0f) && (alpha >= 1.0f / 255.0f));
                }
                __syncthreads();
                if (bSkip == false)
                {
                    int shared_mem_offset = atomicAggInc(&valid_pix_num);
                    transmittance /= (1 - alpha);
                    //color
                    grad_color_x[shared_mem_offset] = alpha * transmittance * d_pixel.x;
                    grad_color_y[shared_mem_offset] = alpha * transmittance * d_pixel.y;
                    grad_color_z[shared_mem_offset] = alpha * transmittance * d_pixel.z;

                    //alpha
                    float d_alpha = 0;
                    d_alpha += (cur_color.x - accum_rec.x) * transmittance * d_pixel.x;
                    d_alpha += (cur_color.y - accum_rec.y) * transmittance * d_pixel.y;
                    d_alpha += (cur_color.z - accum_rec.z) * transmittance * d_pixel.z;
                    accum_rec.x = alpha * cur_color.x + (1.0f - alpha) * accum_rec.x;
                    accum_rec.y = alpha * cur_color.y + (1.0f - alpha) * accum_rec.y;
                    accum_rec.z = alpha * cur_color.z + (1.0f - alpha) * accum_rec.z;

                    //opacity
                    grad_opacity[shared_mem_offset] = G * d_alpha;

                    //cov2d_inv
                    float d_G = cur_color.w * d_alpha;
                    float d_power = G * d_G;
                    grad_invcov_x[shared_mem_offset] = -0.5f * d.x * d.x * d_power;
                    grad_invcov_y[shared_mem_offset] = -0.5f * d.x * d.y * d_power;
                    grad_invcov_z[shared_mem_offset] = -0.5f * d.y * d.y * d_power;

                    //mean2d
                    float d_deltax = (-cur_cov2d_inv.x * d.x - cur_cov2d_inv.y * d.y) * d_power;
                    float d_deltay = (-cur_cov2d_inv.z * d.y - cur_cov2d_inv.y * d.x) * d_power;
                    grad_mean_x[shared_mem_offset] = d_deltax * 0.5f * img_w;
                    grad_mean_y[shared_mem_offset] = d_deltay * 0.5f * img_h;
                }

                __syncthreads();
                if (valid_pix_num > 0)
                {
                    int property_id = threadidx / threadsnum_per_property;
                    int ele_offset = threadidx % threadsnum_per_property;
                    if (property_id < property_num)
                    {
                        float sum = 0;
                        for (int i = ele_offset; i < valid_pix_num; i += threadsnum_per_property)
                        {
                            sum += gradient_buffer[property_id * (tilesize * tilesize + threadsnum_per_property) + i];
                        }
                        gradient_buffer[property_id * (tilesize * tilesize + threadsnum_per_property) + ele_offset] = sum;
                    }
                    __syncthreads();
                    if (threadidx < property_num)
                    {
                        float sum = 0;
                        for (int i = 0; i < threadsnum_per_property; i++)
                        {
                            sum += gradient_buffer[threadidx * (tilesize * tilesize + threadsnum_per_property) + i];
                        }
                        int point_id = sorted_points[batch_id][index];
                        atomicAdd(global_grad_addr + point_id, sum);
                        if (threadidx == 4)
                        {
                            atomicAdd(&d_cov2d_inv[batch_id][1][0][point_id], sum);
                        }
                    }
                    valid_pix_num = 0;


                }
            }
            

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
    std::optional<at::Tensor> specific_tiles_arg,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w
)
{
    at::DeviceGuard guard(ndc.device());

    int64_t viewsnum = start_index.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tilesize));
    int tilesnum_y = std::ceil(img_h / float(tilesize));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    at::Tensor specific_tiles;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        tilesnum = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
    }

    at::Tensor d_ndc = torch::zeros_like(ndc, ndc.options());
    at::Tensor d_cov2d_inv = torch::zeros_like(cov2d_inv, ndc.options());
    at::Tensor d_color = torch::zeros_like(color, ndc.options());
    at::Tensor d_opacity = torch::zeros_like(opacity, ndc.options());

    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);

    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<8>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<16>, cudaFuncCachePreferShared);

    switch (tilesize)
    {
    case 8:
        //raster_backward_kernel_warp_reduction<8> << <Block3d, Thread3d >> > (
        raster_backward_kernel_multibatch_reduction<8> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w);
        break;
    case 16:
        raster_backward_kernel_warp_reduction<16> << <Block3d, Thread3d >> > (
        //raster_backward_kernel_multibatch_reduction<16> << <Block3d, Thread3d >> >(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
            d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w);
        break;
    default:
        assert(false);
        ;
    }
    CUDA_CHECK_ERRORS;
    
    return { d_ndc ,d_cov2d_inv ,d_color,d_opacity };
}

