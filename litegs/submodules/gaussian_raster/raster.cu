#ifndef __CUDACC__
#define __CUDACC__
#define __NVCC__
#endif
#include "cuda_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <math.h>
#include <cuda_fp16.h>
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
};

struct RGBA16
{
    half r;
    half g;
    half b;
    half a;
};

struct RGBA16x2
{
    half2 r;
    half2 g;
    half2 b;
    half2 a;
};

struct RegisterBuffer
{
    half2 r;
    half2 g;
    half2 b;
    half2 t;
    unsigned int lst_contributor;//simd ushort2
    half2 alpha;
};

#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))
inline __device__ half2 fast_exp_approx(half2 input) {
    half2 output;
    half2 log2_e(1.4426950409f, 1.4426950409f);
    half2 scaled_input = input * log2_e;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(__HALF2_TO_UI(output)) : "r"(__HALF2_TO_CUI(scaled_input)));
    return output;
}

template <int tile_size_y, int tile_size_x, bool enable_trans, bool enable_depth>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float/*torch::Half*/, 3, torch::RestrictPtrTraits> packed_params,         //[batch,point_num,6]
    const torch::PackedTensorAccessor32<torch::Half, 3, torch::RestrictPtrTraits> packed_rgba16,         //[batch,point_num,4]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_depth,     //[batch,1,tile,tilesize, tilesize]
    torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    int tiles_num_x, int img_h, int img_w
)
{
    //assert blockDim.x==32

    constexpr int VECTOR_SIZE = 2;
    constexpr int PIXELS_PER_THREAD = (tile_size_x * tile_size_y) / (32* VECTOR_SIZE);//half2: 32 pixel per warp->64 pixel per warp
    constexpr float SCALER = 128.0f;
    constexpr float INV_SCALER = 1.0f / 128;

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
            RegisterBuffer reg_buffer[PIXELS_PER_THREAD];
#pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD; i++)
            {
                reg_buffer[i].r = half2(0, 0);
                reg_buffer[i].g = half2(0, 0);
                reg_buffer[i].b = half2(0, 0);
                //alpha_min 1/256
                //t_min 1/8192
                //-> t_mul_alpha_min 1/(256*8192) -> half underflow
                reg_buffer[i].t = half2(SCALER, SCALER);//mul 128.0f to avoid underflow; t_max * 128 * color_max < half_max;
                reg_buffer[i].lst_contributor = 0;//simd ushort2
            }

            unsigned int any_active = 0xffffffffu;
            int index_in_tile = 0;
            auto points_id_in_tile = &sorted_points[batch_id][start_index_in_tile];
            for (; (index_in_tile+ start_index_in_tile < end_index_in_tile) && (any_active != 0); index_in_tile++)
            {
                int point_id = points_id_in_tile[index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                RGBA16 temp = *((RGBA16*)&packed_rgba16[batch_id][point_id][0]);
                RGBA16x2 point_color_x2;
                point_color_x2.r = half2(temp.r, temp.r);
                point_color_x2.g = half2(temp.g, temp.g);
                point_color_x2.b = half2(temp.b, temp.b);
                point_color_x2.a = half2(temp.a, temp.a);
                float2 xy{ (params.ndc_x + 1.0f) * 0.5f * img_w - 0.5f ,(params.ndc_y + 1.0f) * 0.5f * img_h - 0.5f };

                const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + (threadIdx.x * VECTOR_SIZE) % tile_size_x ;
                const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + (threadIdx.x * VECTOR_SIZE) / tile_size_x * PIXELS_PER_THREAD;
                float2 d { xy.x - pixel_x,xy.y - pixel_y };
                float2 basic{
                    -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y),
                    -0.5f * (params.inv_cov00 * (d.x - 1) * (d.x - 1) + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * (d.x - 1) * d.y)
                };
                float2 bxcy{
                    params.inv_cov11 * d.y + params.inv_cov01 * d.x,
                    params.inv_cov11 * d.y + params.inv_cov01 * (d.x - 1)
                };
                float2 neg_half_c{
                    -0.5f * params.inv_cov11,
                    -0.5f * params.inv_cov11
                };
                //basic+=(cy+bx)*delta - 0.5*c*delta*delta

                any_active = 0;
#pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD; i++)
                {
                    half2 power{
                        basic.x + i * bxcy.x + i * i * neg_half_c.x,
                        basic.y + i * bxcy.y + i * i * neg_half_c.y
                    };
                    unsigned int active_mask = 0xffffffffu;
                    active_mask = __hgt2_mask(reg_buffer[i].t, half2(SCALER / 8192, SCALER / 8192));
                    any_active |= active_mask;

                    unsigned int alpha_valid_mask = 0xffffffffu;
                    alpha_valid_mask &= __hle2_mask(power, half2(0, 0));
                    reg_buffer[i].alpha = point_color_x2.a * fast_exp_approx(power);
                    alpha_valid_mask &= __hge2_mask(reg_buffer[i].alpha, half2(1.0f / 256, 1.0f / 256));
                    reg_buffer[i].alpha = __hmin2(half2(255.0f / 256, 255.0f / 256), reg_buffer[i].alpha);

                    reg_buffer[i].lst_contributor += (0x00010001 & active_mask);
                    reinterpret_cast<unsigned int*>(&reg_buffer[i].alpha)[0] &= (active_mask & alpha_valid_mask);

                    half2 weight = reg_buffer[i].t * reg_buffer[i].alpha;
                    reg_buffer[i].r += (point_color_x2.r * weight);
                    reg_buffer[i].g += (point_color_x2.g * weight);
                    reg_buffer[i].b += (point_color_x2.b * weight);
                    reg_buffer[i].t = reg_buffer[i].t * (half2(1.0f, 1.0f) - reg_buffer[i].alpha);
                }
                //reg_buffer[1].alpha = (half2(2.0f, 2.0f) * reg_buffer[0].alpha + reg_buffer[3].alpha) * half2(1.0f / 3, 1.0f / 3);
                //reg_buffer[2].alpha = (reg_buffer[0].alpha + half2(2.0f, 2.0f) * reg_buffer[3].alpha) * half2(1.0f / 3, 1.0f / 3);

            }


            int tile_index = blockIdx.x * blockDim.y + threadIdx.y;
            auto ourput_r = output_img[batch_id][0][tile_index];
            auto ourput_g = output_img[batch_id][1][tile_index];
            auto ourput_b = output_img[batch_id][2][tile_index];
            auto ourput_t = output_transmitance[batch_id][0][tile_index];
            auto output_last_index = output_last_contributor[batch_id][tile_index];
#pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD; i++)
            {
                const int output_x = (threadIdx.x * VECTOR_SIZE) % tile_size_x;
                const int output_y = (threadIdx.x * VECTOR_SIZE) / tile_size_x * PIXELS_PER_THREAD + i;

                reinterpret_cast<float2*>(&ourput_r[output_y][output_x])[0] = float2{
                    float(reg_buffer[i].r.x) * INV_SCALER,
                    float(reg_buffer[i].r.y) * INV_SCALER };

                reinterpret_cast<float2*>(&ourput_g[output_y][output_x])[0] = float2{
                    float(reg_buffer[i].g.x) * INV_SCALER,
                    float(reg_buffer[i].g.y) * INV_SCALER };

                reinterpret_cast<float2*>(&ourput_b[output_y][output_x])[0] = float2{
                    float(reg_buffer[i].b.x) * INV_SCALER,
                    float(reg_buffer[i].b.y) * INV_SCALER };

                reinterpret_cast<float2*>(&ourput_t[output_y][output_x])[0] = float2{
                    float(reg_buffer[i].t.x) * INV_SCALER,
                    float(reg_buffer[i].t.y) * INV_SCALER };

                reinterpret_cast<unsigned int*>(&output_last_index[output_y][output_x])[0] = reg_buffer[i].lst_contributor;//ushort2
            }

        }
    }
}


std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor  packed_params,//packed param
    at::Tensor  rgba16,//packed param
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

    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kShort).layout(torch::kStrided).device(start_index.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty({ viewsnum, tilesnum, tile_h, tile_w }, opt_c);


    int tiles_per_block = 4;
    dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
    dim3 Thread3d(32, tiles_per_block);

    raster_forward_kernel<8, 8, false, false> << <Block3d, Thread3d >> > (sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rgba16.packed_accessor32<torch::Half, 3, torch::RestrictPtrTraits>(),
        specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        output_last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
        tilesnum_x, img_h, img_w);

    CUDA_CHECK_ERRORS;

    return { output_img ,output_transmitance,output_depth ,output_last_contributor };
}


struct BackwardRegisterBuffer
{
    half2 r;
    half2 g;
    half2 b;
    half2 t;
    half2 alpha;
};


template<class T, bool boardcast>
inline __device__ void warp_reduce_sum(T& data)
{
    data += __shfl_down_sync(0xffffffff, data, 16);
    data += __shfl_down_sync(0xffffffff, data, 8);
    data += __shfl_down_sync(0xffffffff, data, 4);
    data += __shfl_down_sync(0xffffffff, data, 2);
    data += __shfl_down_sync(0xffffffff, data, 1);
    if (boardcast)
        data = __shfl_sync(0xffffffff, data, 0);
}

template<>
inline __device__ void warp_reduce_sum<float, false>(float& data)
{
    int exponent = (__float_as_uint(data) >> 23) & 0xff;
    exponent = __reduce_max_sync(0xffffffff, exponent) - 127;
    int scale_exponent = 23 - exponent;
    bool valid = (exponent > -127) && (scale_exponent < 128);

    float scaler = __uint_as_float(0 | ((scale_exponent + 127) << 23));
    float inv_scaler = __uint_as_float(0 | ((127 - scale_exponent) << 23));
    int scaled_value = static_cast<int>(data * scaler);
    scaled_value = __reduce_add_sync(0xffffffff, scaled_value) * valid;

    data = scaled_value * inv_scaler;
}

template<>
inline __device__ void warp_reduce_sum<float2, false>(float2& data)
{
    int exponent = (__float_as_uint(data.x) >> 23) & 0xff;
    exponent = max(exponent, (__float_as_uint(data.y) >> 23) & 0xff);
    exponent = __reduce_max_sync(0xffffffff, exponent) - 127;
    int scale_exponent = 23 - exponent;
    bool valid = (exponent > -127) && (scale_exponent < 128);

    float scaler = __uint_as_float(0 | ((scale_exponent + 127) << 23));
    float inv_scaler = __uint_as_float(0 | ((127 - scale_exponent) << 23));

    int scaled_value_x = static_cast<int>(data.x * scaler);
    scaled_value_x = __reduce_add_sync(0xffffffff, scaled_value_x) * valid;
    data.x = scaled_value_x * inv_scaler;
    int scaled_value_y = static_cast<int>(data.y * scaler);
    scaled_value_y = __reduce_add_sync(0xffffffff, scaled_value_y) * valid;
    data.y = scaled_value_y * inv_scaler;
}

template<>
inline __device__ void warp_reduce_sum<float3, false>(float3& data)
{
    int exponent = (__float_as_uint(data.x) >> 23) & 0xff;
    exponent = max(exponent, (__float_as_uint(data.y) >> 23) & 0xff);
    exponent = max(exponent, (__float_as_uint(data.z) >> 23) & 0xff);
    exponent = __reduce_max_sync(0xffffffff, exponent) - 127;
    int scale_exponent = 23 - exponent;
    bool valid = (exponent > -127) && (scale_exponent < 128);

    float scaler = __uint_as_float(0 | ((scale_exponent + 127) << 23));
    float inv_scaler = __uint_as_float(0 | ((127 - scale_exponent) << 23));

    int scaled_value_x = static_cast<int>(data.x * scaler);
    scaled_value_x = __reduce_add_sync(0xffffffff, scaled_value_x) * valid;
    data.x = scaled_value_x * inv_scaler;
    int scaled_value_y = static_cast<int>(data.y * scaler);
    scaled_value_y = __reduce_add_sync(0xffffffff, scaled_value_y) * valid;
    data.y = scaled_value_y * inv_scaler;
    int scaled_value_z = static_cast<int>(data.z * scaler);
    scaled_value_z = __reduce_add_sync(0xffffffff, scaled_value_z) * valid;
    data.z = scaled_value_z * inv_scaler;
}

template <int tile_size_y, int tile_size_x, bool enable_trans_grad, bool enable_depth_grad>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_params,         // //[batch,point_num,6]
    const torch::PackedTensorAccessor32<torch::Half, 3, torch::RestrictPtrTraits> packed_rgba16,         // //[batch,point_num,6]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w)
{
    constexpr int VECTOR_SIZE = 2;
    constexpr int PIXELS_PER_THREAD = (tile_size_x * tile_size_y) / (32 * VECTOR_SIZE);//half2: 32 pixel per warp->64 pixel per warp
    constexpr float SCALER = 128.0f;
    constexpr float INV_SCALER = 1.0f / 128;

    __shared__ half2 shared_img_grad[3][PIXELS_PER_THREAD][4 * 32];
    __shared__ unsigned int shared_last_contributor[PIXELS_PER_THREAD][4 * 32];//ushort2

    const int batch_id = blockIdx.y;
    int tile_id = blockIdx.x * blockDim.y + threadIdx.y + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0 && (blockIdx.x * blockDim.y + threadIdx.y < specific_tiles.size(1)))
    {
        tile_id = specific_tiles[batch_id][blockIdx.x * blockDim.y + threadIdx.y];
    }

    if (tile_id != 0 && tile_id < start_index.size(1) - 1)
    {

        int start_index_in_tile = start_index[batch_id][tile_id];
        int index_in_tile = 0;

        if (start_index_in_tile != -1)
        {
            BackwardRegisterBuffer reg_buffer[PIXELS_PER_THREAD];
            //int lst[pixels_per_thread];
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD; i++)
            {
                reg_buffer[i].r = half2(0.0f, 0.0f);
                reg_buffer[i].g = half2(0.0f, 0.0f);
                reg_buffer[i].b = half2(0.0f, 0.0f);

                const int in_tile_x = (threadIdx.x * VECTOR_SIZE) % tile_size_x;
                const int in_tile_y = (threadIdx.x * VECTOR_SIZE) / tile_size_x * PIXELS_PER_THREAD;
                float t0 = final_transmitance[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x];
                float t1 = final_transmitance[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x+1];
                reg_buffer[i].t = half2(t0 * SCALER, t1 * SCALER);

                shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                    d_img[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x],
                    d_img[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x + 1]);
                shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                    d_img[batch_id][1][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x],
                    d_img[batch_id][1][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x + 1]); 
                shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                    d_img[batch_id][2][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x],
                    d_img[batch_id][2][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x + 1]); 
                
                int last0 = last_contributor[batch_id][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x] - 1;
                int last1 = last_contributor[batch_id][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x + 1] - 1;
                index_in_tile = max(max(index_in_tile, last0), last1);
                shared_last_contributor[i][threadIdx.y * blockDim.x + threadIdx.x] = (last1 << 16 | last0);
            }
            index_in_tile = __reduce_max_sync(0xffffffff, index_in_tile);

            const int* points_in_tile = &sorted_points[batch_id][start_index_in_tile];
            const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + (threadIdx.x * VECTOR_SIZE) % tile_size_x;
            const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + (threadIdx.x * VECTOR_SIZE) / tile_size_x * PIXELS_PER_THREAD;

            for (; (index_in_tile >= 0); index_in_tile--)
            {
                float2 basic;
                float2 bxcy;
                float2 neg_half_c;
                float2 d{ 0,0 };
                int point_id = points_in_tile[index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                {
                    float2 xy{ (float(params.ndc_x) + 1.0f) * 0.5f * img_w - 0.5f ,(float(params.ndc_y) + 1.0f) * 0.5f * img_h - 0.5f };
                    d.x = xy.x - pixel_x;
                    d.y = xy.y - pixel_y;
                    basic=float2{
                    -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y),
                    -0.5f * (params.inv_cov00 * (d.x - 1) * (d.x - 1) + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * (d.x - 1) * d.y)
                    };
                    bxcy=float2{
                        params.inv_cov11 * d.y + params.inv_cov01 * d.x,
                        params.inv_cov11 * d.y + params.inv_cov01 * (d.x - 1)
                    };
                    neg_half_c=float2{
                        -0.5f * params.inv_cov11,
                        -0.5f * params.inv_cov11
                    };
                }//basic+=(cy+bx)*delta - 0.5*c*delta*delta

                RGBA16 temp = *((RGBA16*)&packed_rgba16[batch_id][point_id][0]);
                RGBA16x2 point_color_x2;
                point_color_x2.r = half2(temp.r, temp.r);
                point_color_x2.g = half2(temp.g, temp.g);
                point_color_x2.b = half2(temp.b, temp.b);
                point_color_x2.a = half2(temp.a, temp.a);
                

                half2 grad_r = half2(0, 0);
                half2 grad_g = half2(0, 0);
                half2 grad_b = half2(0, 0);
                half2 grad_a = half2(0, 0);
                half2 grad_bxcy = half2(0, 0);
                half2 grad_neg_half_c = half2(0, 0);
                half2 grad_basic = half2(0, 0);
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD; i++)
                {
                    half2 power{ basic.x + i * bxcy.x + i * i * neg_half_c.x,
                        basic.y + i * bxcy.y + i * i * neg_half_c.y };
                    half2 G = fast_exp_approx(power);
                    half2 alpha = point_color_x2.a * G;
                    alpha = __hmin2(half2(255.0f / 256, 255.0f / 256), alpha);

                    unsigned int valid_mask = 0xffffffffu;
                    valid_mask &= __hle2_mask(power, half2(0, 0));
                    valid_mask &= __hge2_mask(alpha, half2(1.0f / 256, 1.0f / 256));
                    valid_mask &= __vcmpleu2(index_in_tile << 16 | index_in_tile, shared_last_contributor[i][threadIdx.y * blockDim.x + threadIdx.x]);

                    if (__any_sync(0xffffffff, valid_mask!=0))
                    {
                        reinterpret_cast<unsigned int*>(&alpha)[0] &= valid_mask;
                        reinterpret_cast<unsigned int*>(&G)[0] &= valid_mask;

                        reg_buffer[i].t = __h2div(reg_buffer[i].t,(half2(1.0f,1.0f) - alpha));//0-2^(-10)
                        grad_r += alpha * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                        grad_g += alpha * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                        grad_b += alpha * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];

                        half2 d_alpha = half2(0,0);
                        d_alpha += (point_color_x2.r - reg_buffer[i].r) * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                        d_alpha += (point_color_x2.g - reg_buffer[i].g) * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                        d_alpha += (point_color_x2.b - reg_buffer[i].b) * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];
                        reg_buffer[i].r += alpha * (point_color_x2.r - reg_buffer[i].r);//0-256
                        reg_buffer[i].g += alpha * (point_color_x2.g - reg_buffer[i].g);
                        reg_buffer[i].b += alpha * (point_color_x2.b - reg_buffer[i].b);
                        if (enable_trans_grad)
                        {
                            //d_alpha -= dL_drbgaimg.z * final_transmitance[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + i][in_tile_x] / (1 - alpha);
                        }

                        grad_a += d_alpha * G;
                        half2 d_G = point_color_x2.a * d_alpha;
                        half2 d_power = G * d_G;
                        grad_bxcy += d_power * half2(i,i);
                        grad_neg_half_c += d_power * half2(i, i) * half2(i, i);
                        grad_basic += d_power;
                    }
                }
                
                //unsigned mask = __ballot_sync(0xffffffff, grad_opacity!=0);
                if (__any_sync(0xffffffff, grad_a.x!=half(0)|| grad_a.y!=half(0)))
                {
                    half2 rg{ grad_r.x + grad_r.y ,grad_g.x + grad_g.y };
                    half2 ba{ grad_b.x + grad_b.y ,grad_a.x + grad_a.y };
                    warp_reduce_sum<half2, false>(rg);
                    warp_reduce_sum<half2, false>(ba);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&d_color[batch_id][0][point_id], float(rg.x)* INV_SCALER);
                        atomicAdd(&d_color[batch_id][1][point_id], float(rg.y)* INV_SCALER);
                        atomicAdd(&d_color[batch_id][2][point_id], float(ba.x)* INV_SCALER);
                        atomicAdd(&d_opacity[0][point_id], float(ba.y)* INV_SCALER);
                    }

                    float3 grad_invcov{ 0,0,0 };
                    //basic = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                    //bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                    //neg_half_c = -0.5f * params.inv_cov11;
                    grad_invcov.x = -0.5f * d.x * d.x * float(grad_basic.x);
                    grad_invcov.x += -0.5f * (d.x-1) * (d.x-1) * float(grad_basic.y);
                    grad_invcov.x *= INV_SCALER;
                    grad_invcov.y = (-d.x * d.y * float(grad_basic.x) + d.x * float(grad_bxcy.x)) * 0.5f;
                    grad_invcov.y += (-(d.x-1) * d.y * float(grad_basic.y) + (d.x-1) * float(grad_bxcy.y)) * 0.5f;
                    grad_invcov.y *= INV_SCALER;
                    grad_invcov.z = -0.5f * d.y * d.y * float(grad_basic.x) + d.y * float(grad_bxcy.x) - 0.5f * float(grad_neg_half_c.x);
                    grad_invcov.z += -0.5f * d.y * d.y * float(grad_basic.y) + d.y * float(grad_bxcy.y) - 0.5f * float(grad_neg_half_c.y);
                    grad_invcov.z *= INV_SCALER;

                    warp_reduce_sum<float, false>(grad_invcov.x);
                    warp_reduce_sum<float, false>(grad_invcov.y);
                    warp_reduce_sum<float, false>(grad_invcov.z);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&d_cov2d_inv[batch_id][0][0][point_id], grad_invcov.x);
                        atomicAdd(&d_cov2d_inv[batch_id][0][1][point_id], grad_invcov.y);
                        atomicAdd(&d_cov2d_inv[batch_id][1][0][point_id], grad_invcov.y);
                        atomicAdd(&d_cov2d_inv[batch_id][1][1][point_id], grad_invcov.z);
                    }

                    float d_dx = (-params.inv_cov00 * d.x - params.inv_cov01 * d.y) * float(grad_basic.x) + params.inv_cov01 * float(grad_bxcy.x);
                    d_dx += (-params.inv_cov00 * (d.x-1) - params.inv_cov01 * d.y) * float(grad_basic.y) + params.inv_cov01 * float(grad_bxcy.y);
                    d_dx *= INV_SCALER;
                    float d_dy = (-params.inv_cov11 * d.y - params.inv_cov01 * d.x) * float(grad_basic.x) + params.inv_cov11 * float(grad_bxcy.x);
                    d_dy += (-params.inv_cov11 * d.y - params.inv_cov01 * (d.x-1)) * float(grad_basic.y) + params.inv_cov11 * float(grad_bxcy.y);
                    d_dy *= INV_SCALER;
                    float2 d_ndc_xy{ d_dx * 0.5f * img_w,d_dy * 0.5f * img_h };
                    warp_reduce_sum<float2, false>(d_ndc_xy);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&d_ndc[batch_id][0][point_id], d_ndc_xy.x);
                        atomicAdd(&d_ndc[batch_id][1][point_id], d_ndc_xy.y);
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
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_params,         // //[batch,point_num,6]
    const torch::PackedTensorAccessor32<torch::Half, 3, torch::RestrictPtrTraits> packed_rgba16,         // //[batch,point_num,4]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
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
        int pixel_lst_index = last_contributor[batch_id][blockIdx.x][y_in_tile][x_in_tile]-1;
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

                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                RGBA16 point_color = *((RGBA16*)&packed_rgba16[batch_id][point_id][0]);

                collected_mean[threadidx].x = (params.ndc_x + 1.0f) * 0.5f * img_w - 0.5f;
                collected_mean[threadidx].y = (params.ndc_y + 1.0f) * 0.5f * img_h - 0.5f;
                if (enable_depth_grad)
                {
                    collected_depth[threadidx] = params.ndc_z;
                }
                collected_invcov[threadidx].x = params.inv_cov00;
                collected_invcov[threadidx].y = params.inv_cov01;
                collected_invcov[threadidx].z = params.inv_cov11;
                collected_color[threadidx].x = point_color.r;
                collected_color[threadidx].y = point_color.g;
                collected_color[threadidx].z = point_color.b;
                collected_color[threadidx].w = point_color.a;
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
                if (index - start_index_in_tile <= pixel_lst_index)
                {
                    float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                    G = __expf(power);
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
    at::Tensor packed_rgba16,// 
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

    
    int tiles_per_block = 4;
    dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
    dim3 Thread3d(32, tiles_per_block);
    //dim3 Block3d(1, viewsnum, 1);
    //dim3 Thread3d(32, 1);
    raster_backward_kernel<8, 8, false, false> << <Block3d, Thread3d >> > (
        sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        packed_rgba16.packed_accessor32<torch::Half, 3, torch::RestrictPtrTraits>(),
        specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),
        last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
        d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
        d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
        tilesnum_x, img_h, img_w
    );
    
    /*cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<16, false, false>, cudaFuncCachePreferShared);
    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(16, 16, 1);

    raster_backward_kernel_multibatch_reduction<16, false, false> << <Block3d, Thread3d >> > (
        sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        packed_rgba16.packed_accessor32<torch::Half, 3, torch::RestrictPtrTraits>(),
        specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),
        last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
        d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
        d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),
        tilesnum_x, img_h, img_w
        );*/

    CUDA_CHECK_ERRORS;
    return { d_ndc ,d_cov2d_inv ,d_color,d_opacity };
}
