#ifndef __CUDACC__
#define __CUDACC__
#define __NVCC__
#endif
#include "cuda_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <math.h>
namespace cg = cooperative_groups;

#include <c10/cuda/CUDAException.h>
#include <ATen/core/TensorAccessor.h>

#include "cuda_errchk.h"
#include "raster.h"


struct PackedParams
{
    float ndc_x;
    float ndc_y;
    float ndc_z;
    float inv_cov00;
    float inv_cov01;
    float inv_cov11;
    float color_r;
    float color_g;
    float color_b;
    float opacity;
};

template <int tile_size_y, int tile_size_x, bool enable_trans, bool enable_depth>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float/*torch::Half*/, 3, torch::RestrictPtrTraits> packed_params,         //[batch,point_num,10]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_depth,     //[batch,1,tile,tilesize, tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    int tiles_num_x, int img_h, int img_w
)
{
    //assert blockDim.x==32
    const int pixels_per_thread = tile_size_x * tile_size_y / 32;
    const int batch_id = blockIdx.y;
    int tile_id = blockIdx.x * blockDim.y + threadIdx.y + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0 && (blockIdx.x * blockDim.y + threadIdx.y < specific_tiles.size(1)))
    {
        tile_id = specific_tiles[batch_id][blockIdx.x * blockDim.y + threadIdx.y];
    }

    if (tile_id != 0 && tile_id < start_index.size(1) - 1)
    {

        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        int last_contributor = 0;
        if (start_index_in_tile != -1)
        {
            float4 rgba_buffer[pixels_per_thread];
            int lst_contributor[pixels_per_thread];
            float alpha[pixels_per_thread];
#pragma unroll
            for (int i = 0; i < pixels_per_thread; i++)
            {
                rgba_buffer[i] = float4{ 0.0f,0.0f,0.0f,1.0f };
            }

            int finish_pixel = 0;
            int index_in_tile = start_index_in_tile;
            for (; (index_in_tile < end_index_in_tile) && (finish_pixel < (1<<pixels_per_thread)-1); index_in_tile++)
            {
                int point_id = sorted_points[batch_id][index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                float2 xy{ (float(params.ndc_x) + 1.0f) * 0.5f * img_w - 0.5f ,(float(params.ndc_y) + 1.0f) * 0.5f * img_h - 0.5f };

                const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + threadIdx.x % tile_size_x;
                const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + threadIdx.x / tile_size_x * pixels_per_thread;
                float2 d = { xy.x - pixel_x,xy.y - pixel_y };
                float exponent = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                float bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                float neg_half_c = -0.5f * params.inv_cov11;
                //exponent+=(cy+bx)*delta - 0.5*c*delta*delta
#pragma unroll
                for (int i = 0; i < pixels_per_thread; i++)
                {
                    float power = exponent + i * bxcy + i * i * neg_half_c;
                    power = power > 0 ? -10.0f : power;
                    alpha[i] = min(0.99f, params.opacity * __expf(power));
                }
                //alpha[1] = (2 * alpha[0] + alpha[3]) * (1 / 3.0f);
                //alpha[2] = (alpha[0] + 2 * alpha[3]) * (1 / 3.0f);

#pragma unroll
                for (int i = 0; i < pixels_per_thread; i++)
                {
                    if (alpha[i] >= (1 / 255.0f)&& (finish_pixel & (1 << i))==0)
                    {
                        float test_t = rgba_buffer[i].w * (1 - alpha[i]);
                        if (blockIdx.x * blockDim.y + threadIdx.y == 10531 && threadIdx.x == 6 && i == 0 && lst_contributor[i]==5222315)
                        {
                            printf("test_t:%f lst:%d\n", test_t,lst_contributor[i]);
                        }
                        if (test_t < 0.0001f)
                        {
                            if (blockIdx.x * blockDim.y + threadIdx.y == 10531 && threadIdx.x == 6)
                            {
                                printf("finish test_t:%f lst:%d\n", test_t, lst_contributor[i]);
                            }
                            finish_pixel = (finish_pixel|(1<<i));
                        }
                        else
                        {
                            float weight = rgba_buffer[i].w * alpha[i];
                            rgba_buffer[i].x += (float(params.color_r) * weight);
                            rgba_buffer[i].y += (float(params.color_g) * weight);
                            rgba_buffer[i].z += (float(params.color_b) * weight);
                            rgba_buffer[i].w = test_t;
                            lst_contributor[i] = index_in_tile;
                        }
                    }
                }

            }


            int tile_index = blockIdx.x * blockDim.y + threadIdx.y;
            auto ourput_r = output_img[batch_id][0][tile_index];
            auto ourput_g = output_img[batch_id][1][tile_index];
            auto ourput_b = output_img[batch_id][2][tile_index];
            auto ourput_t = output_transmitance[batch_id][0][tile_index];
            auto output_last_index = output_last_contributor[batch_id][tile_index];
#pragma unroll
            for (int i = 0; i < pixels_per_thread; i++)
            {
                const int output_x = threadIdx.x % tile_size_x;
                const int output_y = threadIdx.x / tile_size_x * pixels_per_thread + i;
                ourput_r[output_y][output_x] = rgba_buffer[i].x;
                ourput_g[output_y][output_x] = rgba_buffer[i].y;
                ourput_b[output_y][output_x] = rgba_buffer[i].z;
                ourput_t[output_y][output_x] = rgba_buffer[i].w;
                output_last_index[output_y][output_x] = lst_contributor[i];
            }

        }
    }
}


std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor  packed_params,//packed param
    std::optional<at::Tensor>  specific_tiles_arg,
    int64_t img_h,
    int64_t img_w,
    int64_t tile_h,
    int64_t tile_w,
    bool enable_trans,
    bool enable_depth
)
{
    at::DeviceGuard guard(packed_params.device());

    int64_t viewsnum = start_index.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    at::Tensor specific_tiles;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        tilesnum = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
    }


    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_img = torch::empty({ viewsnum,3, tilesnum,tile_h,tile_w }, opt_img);

    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(enable_trans);
    at::Tensor output_transmitance = torch::empty({ viewsnum,1, tilesnum, tile_h, tile_w }, opt_t);

    at::Tensor output_depth = torch::empty({ 0,0, 0, 0, 0 }, opt_t);
    if (enable_depth)
    {
        output_depth = torch::empty({ viewsnum,1, tilesnum, tile_h, tile_w }, opt_t.requires_grad(true));
    }

    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(start_index.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty({ viewsnum, tilesnum, tile_h, tile_w }, opt_c);


    int tiles_per_block = 4;
    dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
    dim3 Thread3d(32, tiles_per_block);

    raster_forward_kernel<8, 8, false, false> << <Block3d, Thread3d >> > (sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        tilesnum_x, img_h, img_w);

    CUDA_CHECK_ERRORS;

    return { output_img ,output_transmitance,output_depth ,output_last_contributor };
}

template <int tile_size_y, int tile_size_x, bool enable_trans_grad, bool enable_depth_grad>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_params,         // //[batch,point_num,10]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w)
{
    const int pixels_per_thread = tile_size_x * tile_size_y / 32;
    const int batch_id = blockIdx.y;
    int tile_id = blockIdx.x * blockDim.y + threadIdx.y + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0 && (blockIdx.x * blockDim.y + threadIdx.y < specific_tiles.size(1)))
    {
        tile_id = specific_tiles[batch_id][blockIdx.x * blockDim.y + threadIdx.y];
    }

    if (tile_id != 0 && tile_id < start_index.size(1) - 1)
    {

        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        if (start_index_in_tile != -1)
        {
            float4 rgba_buffer[pixels_per_thread];
            float4 grad_img_buffer[pixels_per_thread];
            const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + threadIdx.x % tile_size_x;
            const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + threadIdx.x / tile_size_x * pixels_per_thread;
#pragma unroll
            for (int i = 0; i < pixels_per_thread; i++)
            {
                float t = final_transmitance[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][pixel_y + i][pixel_x];
                float dL_dr = d_img[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][pixel_y + i][pixel_x];
                float dL_dg = d_img[batch_id][1][blockIdx.x * blockDim.y + threadIdx.y][pixel_y + i][pixel_x];
                float dL_db = d_img[batch_id][2][blockIdx.x * blockDim.y + threadIdx.y][pixel_y + i][pixel_x];
                float dL_da = 0;
                if (enable_trans_grad)
                    float dL_da = d_trans_img[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][pixel_y + i][pixel_x];

                rgba_buffer[i] = float4{ 0.0f,0.0f,0.0f,t };
                grad_img_buffer[i] = float4{ dL_dr ,dL_dg ,dL_db,dL_da };
            }

            int index_in_tile = end_index_in_tile - 1;
            for (; (index_in_tile >= start_index_in_tile); index_in_tile--)
            {
                int pixel_lst_index = last_contributor[batch_id][blockIdx.x * blockDim.y + threadIdx.y][pixel_y][pixel_x];
                if (index_in_tile > pixel_lst_index)
                {
                    continue;
                }

                int point_id = sorted_points[batch_id][index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                float2 xy{ (float(params.ndc_x) + 1.0f) * 0.5f * img_w - 0.5f ,(float(params.ndc_y) + 1.0f) * 0.5f * img_h - 0.5f };
                float2 d = { xy.x - pixel_x,xy.y - pixel_y };
                float exponent = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                float bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                float neg_half_c = -0.5f * params.inv_cov11;
                //exponent+=(cy+bx)*delta - 0.5*c*delta*delta

                float d_alpha = 0;
                float3 grad_color{ 0,0,0 };
                float grad_opacity = 0;
                float d_bxcy = 0;
                float d_neg_half_c = 0;
                float d_exponent = 0;
#pragma unroll
                for (int i = 0; i < pixels_per_thread; i++)
                {
                    float power = exponent + i * bxcy + i * i * neg_half_c;
                    float G = __expf(power);
                    float alpha = min(0.99f, params.opacity * power);
                    if (alpha > (1 / 255.0f))
                    {
                        rgba_buffer[i].w /= (1 - alpha);
                        grad_color.x += alpha * rgba_buffer[i].w * grad_img_buffer[i].x;
                        grad_color.y += alpha * rgba_buffer[i].w * grad_img_buffer[i].y;
                        grad_color.z += alpha * rgba_buffer[i].w * grad_img_buffer[i].z;


                        d_alpha += (params.color_r - grad_img_buffer[i].x) * rgba_buffer[i].w * grad_img_buffer[i].x;
                        d_alpha += (params.color_g - grad_img_buffer[i].y) * rgba_buffer[i].w * grad_img_buffer[i].y;
                        d_alpha += (params.color_b - grad_img_buffer[i].z) * rgba_buffer[i].w * grad_img_buffer[i].z;
                        grad_img_buffer[i].x = alpha * params.color_r + (1.0f - alpha) * grad_img_buffer[i].x;
                        grad_img_buffer[i].y = alpha * params.color_g + (1.0f - alpha) * grad_img_buffer[i].y;
                        grad_img_buffer[i].z = alpha * params.color_b + (1.0f - alpha) * grad_img_buffer[i].z;
                        if (enable_trans_grad)
                        {
                            d_alpha -= grad_img_buffer[i].w * final_transmitance[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][pixel_y + i][pixel_x] / (1 - alpha);
                        }

                        grad_opacity += d_alpha * G;
                        float d_G = params.opacity * d_alpha;
                        float d_power = power * d_G;
                        d_bxcy += d_power * i;
                        d_neg_half_c += d_power * i * i;
                        d_exponent += d_power;
                    }
                }

                //grad color&opacity
                //grad_color.x = WarpReduce(temp_storage[threadIdx.y]).Sum(grad_color.x);
                //grad_color.y = WarpReduce(temp_storage[threadIdx.y]).Sum(grad_color.y);
                //grad_color.z = WarpReduce(temp_storage[threadIdx.y]).Sum(grad_color.z);
                //grad_opacity = WarpReduce(temp_storage[threadIdx.y]).Sum(grad_opacity);
                if (threadIdx.x == 0)
                {
                    atomicAdd(&d_color[batch_id][0][point_id], grad_color.x);
                    atomicAdd(&d_color[batch_id][1][point_id], grad_color.y);
                    atomicAdd(&d_color[batch_id][2][point_id], grad_color.z);
                    atomicAdd(&d_opacity[0][point_id], grad_opacity);
                }
                float3 grad_invcov{ 0,0,0 };
                grad_invcov.x = -0.5f * d.x * d.x * d_exponent;
                grad_invcov.y = -0.5f * d.x * d.y * d_exponent + d.x * d_bxcy;
                grad_invcov.z = -0.5f * d.y * d.y * d_exponent - 0.5f * d_neg_half_c;


            }
        }
    }
}

template <int tile_size_y, int tile_size_x, bool enable_trans_grad, bool enable_depth_grad>
__global__ void raster_backward_kernel_warp_reduction(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_params,         // //[batch,point_num,10]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w
)
{

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int x_in_tile = (threadIdx.y * blockDim.x + threadIdx.x) % tile_size_x;
    const int y_in_tile = (threadIdx.y * blockDim.x + threadIdx.x) / tile_size_x;

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
            float f_transmittance = final_transmitance[batch_id][0][tile_index][y_in_tile][x_in_tile];
            float transmittance = f_transmittance;
            int pixel_lst_index = last_contributor[batch_id][tile_index][y_in_tile][x_in_tile];

            float3 d_pixel{ 0,0,0 };
            float d_trans_pixel = 0;
            float d_depth_pixel = 0;
            int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + x_in_tile;
            int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + y_in_tile;
            if (pixel_x < img_w && pixel_y < img_h)
            {
                d_pixel.x = d_img[batch_id][0][tile_index][y_in_tile][x_in_tile];
                d_pixel.y = d_img[batch_id][1][tile_index][y_in_tile][x_in_tile];
                d_pixel.z = d_img[batch_id][2][tile_index][y_in_tile][x_in_tile];
                if (enable_trans_grad)
                {
                    d_trans_pixel = d_trans_img[batch_id][0][tile_index][y_in_tile][x_in_tile];
                }
                if (enable_depth_grad)
                {
                    d_depth_pixel = d_depth_img[batch_id][0][tile_index][y_in_tile][x_in_tile];
                }
            }
            //loop points
            float3 accum_rec{ 0,0,0 };
            float accum_depth = 0;
            for (int offset = end_index_in_tile - 1; offset >= start_index_in_tile; offset--)
            {

                int point_index = offset;
                bool skip = point_index > pixel_lst_index;
                float3 grad_color{ 0,0,0 };
                float3 grad_invcov{ 0,0,0 };
                float2 grad_ndc_xy{ 0,0 };
                float grad_ndc_z = 0;
                float grad_opacity{ 0 };

                if (skip == false)
                {
                    int point_id = sorted_points[batch_id][point_index];
                    PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                    float2 xy{ (float(params.ndc_x) + 1.0f) * 0.5f * img_w - 0.5f ,(float(params.ndc_y) + 1.0f) * 0.5f * img_h - 0.5f };
                    float2 d{ xy.x - pixel_x,xy.y - pixel_y };


                    float power = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y) - params.inv_cov01 * d.x * d.y;
                    skip |= power > 0.0f;

                    float G = __expf(power);
                    float alpha = min(0.99f, params.opacity * G);
                    skip |= (alpha < 1.0f / 255.0f);
                    if (skip == false)
                    {
                        transmittance /= (1 - alpha);
                        //color
                        grad_color.x = alpha * transmittance * d_pixel.x;
                        grad_color.y = alpha * transmittance * d_pixel.y;
                        grad_color.z = alpha * transmittance * d_pixel.z;
                        grad_ndc_z = alpha * transmittance * d_depth_pixel;


                        //alpha
                        float d_alpha = 0;
                        d_alpha += (params.color_r - accum_rec.x) * transmittance * d_pixel.x;
                        d_alpha += (params.color_g - accum_rec.y) * transmittance * d_pixel.y;
                        d_alpha += (params.color_b - accum_rec.z) * transmittance * d_pixel.z;
                        accum_rec.x = alpha * params.color_r + (1.0f - alpha) * accum_rec.x;
                        accum_rec.y = alpha * params.color_g + (1.0f - alpha) * accum_rec.y;
                        accum_rec.z = alpha * params.color_b + (1.0f - alpha) * accum_rec.z;
                        if (enable_trans_grad)
                        {
                            d_alpha -= d_trans_pixel * f_transmittance / (1 - alpha);
                        }
                        if (enable_depth_grad)
                        {
                            d_alpha += (params.ndc_z - accum_depth) * transmittance * d_depth_pixel;
                            accum_depth = alpha * params.ndc_z + (1.0f - alpha) * accum_depth;
                        }

                        //opacity
                        grad_opacity = G * d_alpha;

                        //cov2d_inv
                        float d_G = params.opacity * d_alpha;
                        float d_power = G * d_G;
                        grad_invcov.x = -0.5f * d.x * d.x * d_power;
                        grad_invcov.y = -0.5f * d.x * d.y * d_power;
                        grad_invcov.z = -0.5f * d.y * d.y * d_power;

                        //mean2d
                        float d_deltax = (-params.inv_cov00 * d.x - params.inv_cov01 * d.y) * d_power;
                        float d_deltay = (-params.inv_cov11 * d.y - params.inv_cov01 * d.x) * d_power;
                        grad_ndc_xy.x = d_deltax;
                        grad_ndc_xy.y = d_deltay;
                    }

                }


                if (warp.all(skip) == false)
                {
                    grad_color.x = cg::reduce(warp, grad_color.x, cg::plus<float>());
                    grad_color.y = cg::reduce(warp, grad_color.y, cg::plus<float>());
                    grad_color.z = cg::reduce(warp, grad_color.z, cg::plus<float>());

                    grad_invcov.x = cg::reduce(warp, grad_invcov.x, cg::plus<float>());
                    grad_invcov.y = cg::reduce(warp, grad_invcov.y, cg::plus<float>());
                    grad_invcov.z = cg::reduce(warp, grad_invcov.z, cg::plus<float>());

                    grad_ndc_xy.x = cg::reduce(warp, grad_ndc_xy.x, cg::plus<float>());
                    grad_ndc_xy.y = cg::reduce(warp, grad_ndc_xy.y, cg::plus<float>());
                    if (enable_depth_grad)
                        grad_ndc_z = cg::reduce(warp, grad_ndc_z, cg::plus<float>());

                    grad_opacity = cg::reduce(warp, grad_opacity, cg::plus<float>());

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

                        atomicAdd(&d_ndc[batch_id][0][point_id], grad_ndc_xy.x * 0.5f * img_w);
                        atomicAdd(&d_ndc[batch_id][1][point_id], grad_ndc_xy.y * 0.5f * img_h);
                        if (enable_depth_grad)
                        {
                            atomicAdd(&d_ndc[batch_id][2][point_id], grad_ndc_z);
                        }

                        atomicAdd(&d_opacity[0][point_id], grad_opacity);
                    }
                }
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

template <int tilesize, bool enable_trans_grad, bool enable_depth_grad>
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
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w)
{
    __shared__ float4 collected_color[tilesize * tilesize];
    __shared__ float3 collected_invcov[tilesize * tilesize];
    __shared__ float2 collected_mean[tilesize * tilesize + int(enable_depth_grad) * tilesize * tilesize / 2];
    float* collected_depth = (float*)(collected_mean + tilesize * tilesize);

    constexpr int property_num = 9 + enable_depth_grad;

    constexpr int threadsnum_per_property = tilesize * tilesize / property_num;
    __shared__ float gradient_buffer[(tilesize * tilesize + threadsnum_per_property) * property_num];//"+threadsnum_per_property" to avoid bank conflict
    float* const grad_color_x = gradient_buffer;
    float* const grad_color_y = gradient_buffer + 1 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_color_z = gradient_buffer + 2 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_x = gradient_buffer + 3 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_y = gradient_buffer + 4 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_z = gradient_buffer + 5 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_ndc_x = gradient_buffer + 6 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_ndc_y = gradient_buffer + 7 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_opacity = gradient_buffer + 8 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_ndc_z = gradient_buffer + 9 * (tilesize * tilesize + threadsnum_per_property);

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
    case 9:
        global_grad_addr = &d_ndc[batch_id][2][0];
        break;
    default:
        break;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        valid_pix_num = 0;
    }
    __syncthreads();

    if (tile_id != 0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];
        if (start_index_in_tile == -1)
        {
            return;
        }

        float f_transmittance = final_transmitance[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
        float transmittance = f_transmittance;
        int pixel_lst_index = last_contributor[batch_id][blockIdx.x][y_in_tile][x_in_tile];
        float3 d_pixel{ 0,0,0 };
        float d_trans_pixel = 0;
        float d_depth_pixel = 0;
        if (pixel_x < img_w && pixel_y < img_h)
        {
            d_pixel.x = d_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
            d_pixel.y = d_img[batch_id][1][blockIdx.x][y_in_tile][x_in_tile];
            d_pixel.z = d_img[batch_id][2][blockIdx.x][y_in_tile][x_in_tile];
            if (enable_trans_grad)
            {
                d_trans_pixel = d_trans_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
            }
            if (enable_depth_grad)
            {
                d_depth_pixel = d_depth_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
            }
        }

        float3 accum_rec{ 0,0,0 };
        float accum_depth = 0;
        for (int offset = end_index_in_tile - 1; offset >= start_index_in_tile; offset -= (tilesize * tilesize))
        {
            int collected_num = min(tilesize * tilesize, offset - start_index_in_tile + 1);
            if (threadIdx.y * blockDim.x + threadIdx.x < collected_num)
            {
                int index = offset - threadidx;
                int point_id = sorted_points[batch_id][index];

                collected_mean[threadidx].x = (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f;
                collected_mean[threadidx].y = (ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f;
                if (enable_depth_grad)
                {
                    collected_depth[threadidx] = ndc[batch_id][2][point_id];
                }
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
                    if (enable_depth_grad)
                    {
                        grad_ndc_z[shared_mem_offset] = alpha * transmittance * d_depth_pixel;
                    }

                    //alpha
                    float d_alpha = 0;
                    d_alpha += (cur_color.x - accum_rec.x) * transmittance * d_pixel.x;
                    d_alpha += (cur_color.y - accum_rec.y) * transmittance * d_pixel.y;
                    d_alpha += (cur_color.z - accum_rec.z) * transmittance * d_pixel.z;
                    accum_rec.x = alpha * cur_color.x + (1.0f - alpha) * accum_rec.x;
                    accum_rec.y = alpha * cur_color.y + (1.0f - alpha) * accum_rec.y;
                    accum_rec.z = alpha * cur_color.z + (1.0f - alpha) * accum_rec.z;
                    if (enable_trans_grad)
                    {
                        d_alpha -= d_trans_pixel * f_transmittance / (1 - alpha);
                    }
                    if (enable_depth_grad)
                    {
                        d_alpha += (collected_depth[i] - accum_depth) * transmittance * d_depth_pixel;
                        accum_depth = alpha * collected_depth[i] + (1.0f - alpha) * accum_depth;
                    }

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
                    grad_ndc_x[shared_mem_offset] = d_deltax * 0.5f * img_w;
                    grad_ndc_y[shared_mem_offset] = d_deltay * 0.5f * img_h;
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
    at::Tensor packed_params,// 
    at::Tensor ndc,// 
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    std::optional<at::Tensor> specific_tiles_arg,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    std::optional<at::Tensor> d_trans_img_arg,
    std::optional<at::Tensor> d_depth_img_arg,
    int64_t img_h,
    int64_t img_w,
    int64_t tilesize_h,
    int64_t tilesize_w
)
{
    at::DeviceGuard guard(packed_params.device());

    int64_t viewsnum = start_index.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tilesize_w));
    int tilesnum_y = std::ceil(img_h / float(tilesize_h));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    at::Tensor specific_tiles;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        tilesnum = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
    }
    at::Tensor d_trans_img;
    if (d_trans_img_arg.has_value())
    {
        d_trans_img = *d_trans_img_arg;
    }
    else
    {
        d_trans_img = torch::empty({ 0,0,0,0,0 }, d_img.options());
    }
    at::Tensor d_depth_img;
    if (d_depth_img_arg.has_value())
    {
        d_depth_img = *d_depth_img_arg;
    }
    else
    {
        d_depth_img = torch::empty({ 0,0,0,0,0 }, d_img.options());
    }
    int batch_num = packed_params.size(0);
    int points_num = packed_params.size(1);
    at::Tensor d_ndc = torch::zeros({ batch_num,4,points_num }, packed_params.options());
    at::Tensor d_cov2d_inv = torch::zeros({ batch_num,2,2,points_num }, packed_params.options());
    at::Tensor d_color = torch::zeros({ batch_num,3,points_num }, packed_params.options());
    at::Tensor d_opacity = torch::zeros({ 1,points_num }, packed_params.options());

    //int tiles_per_block = 4;
    //dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
    //dim3 Thread3d(32, tiles_per_block);
    //raster_backward_kernel_warp_reduction<16, 8, false, false> << <Block3d, Thread3d >> > (

    /*dim3 Block3d(tilesnum, viewsnum, 1);
    raster_backward_kernel_warp_reduction<8, 8, false, false> << <Block3d, 8 * 8 >> > (
        sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),
        last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
        d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
        tilesnum_x, img_h, img_w
    );*/
    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<8, false, false>, cudaFuncCachePreferShared);
    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(8, 8, 1);
    raster_backward_kernel_multibatch_reduction<8, false, false> << <Block3d, Thread3d >> > (
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
        d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
        d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
        tilesnum_x, img_h, img_w
        );


    return { d_ndc ,d_cov2d_inv ,d_color,d_opacity };
}
