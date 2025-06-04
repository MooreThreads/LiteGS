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
    float pixel_x;
    float pixel_y;
    float depth;
    float inv_cov00;
    float inv_cov01;
    float inv_cov11;
    half2 rg;
    half2 ba;
};

struct PackedGrad
{
    float ndc_x;
    float ndc_y;
    float inv_cov00;
    float inv_cov01;
    float inv_cov11;
    float r;
    float g;
    float b;
    float a;
};

struct RGBA16
{
    half r;
    half g;
    half b;
    half a;
};

struct RGBA32
{
    float r;
    float g;
    float b;
    float a;
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
    const torch::PackedTensorAccessor32<float/*torch::Half*/, 3, torch::RestrictPtrTraits> packed_params,         //[batch,point_num,sizeof(PackedParams)/4]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_depth,     //[batch,1,tile,tilesize, tilesize]
    torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    int tiles_num_x)
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

        if (start_index_in_tile != -1)
        {

            unsigned int any_active = 0xffffffffu;
            int index_in_tile = 0;
            auto points_id_in_tile = &sorted_points[batch_id][start_index_in_tile];
            for (; (index_in_tile+ start_index_in_tile < end_index_in_tile) && (any_active != 0); index_in_tile++)
            {
                int point_id = points_id_in_tile[index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                RGBA16x2 point_color_x2;
                point_color_x2.r = half2(params.rg.x, params.rg.x);
                point_color_x2.g = half2(params.rg.y, params.rg.y);
                point_color_x2.b = half2(params.ba.x, params.ba.x);
                point_color_x2.a = half2(params.ba.y, params.ba.y);
                float2 xy{ params.pixel_x,params.pixel_y };

                const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + threadIdx.x % tile_size_x ;
                const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;
                float2 d { xy.x - pixel_x,xy.y - pixel_y };
                float basic = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                float bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                float neg_half_c = -0.5f * params.inv_cov11;
                //basic+=(cy+bx)*delta - 0.5*c*delta*delta

                any_active = 0;
#pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD; i++)
                {
                    half2 power{
                        basic + 2 * i * bxcy + 2 * i * 2 * i * neg_half_c,
                        basic + (2 * i + 1) * bxcy + (2 * i + 1) * (2 * i + 1) * neg_half_c
                    };
                    unsigned int active_mask = 0xffffffffu;
                    active_mask = __hgt2_mask(reg_buffer[i].t, half2(SCALER / 8192, SCALER / 8192));
                    any_active |= active_mask;

                    unsigned int alpha_valid_mask = 0xffffffffu;
                    //alpha_valid_mask &= __hle2_mask(power, half2(1.0f / (1 << 24), 1.0f / (1 << 24)));//1 ULP:2^(-14) * (0 + 1/1024)
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
            const int output_x = threadIdx.x % tile_size_x;
            const int output_y = threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE + 2 * i;

            ourput_r[output_y][output_x] = min(float(reg_buffer[i].r.x) * INV_SCALER, 1.0f);
            ourput_r[output_y + 1][output_x] = min(float(reg_buffer[i].r.y) * INV_SCALER, 1.0f);

            ourput_g[output_y][output_x] = min(float(reg_buffer[i].g.x) * INV_SCALER, 1.0f);
            ourput_g[output_y + 1][output_x] = min(float(reg_buffer[i].g.y) * INV_SCALER, 1.0f);

            ourput_b[output_y][output_x] = min(float(reg_buffer[i].b.x) * INV_SCALER, 1.0f);
            ourput_b[output_y + 1][output_x] = min(float(reg_buffer[i].b.y) * INV_SCALER, 1.0f);

            ourput_t[output_y][output_x] = float(reg_buffer[i].t.x) * INV_SCALER;
            ourput_t[output_y + 1][output_x] = float(reg_buffer[i].t.y) * INV_SCALER;

            output_last_index[output_y][output_x] = reg_buffer[i].lst_contributor & 0xffff;
            output_last_index[output_y + 1][output_x] = (reg_buffer[i].lst_contributor >> 16) & 0xffff;
        }
    }
}

__global__ void pack_forward_params(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    torch::PackedTensorAccessor32<float/*torch::Half*/, 3, torch::RestrictPtrTraits> packed_params,//[batch,point_num,sizeof(PackedParams)/4]
    int img_h, int img_w
)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < ndc.size(2))
    {
        PackedParams* output = (PackedParams*) & packed_params[blockIdx.y][index][0];
        output->pixel_x = (ndc[blockIdx.y][0][index] + 1.0f) * 0.5f * img_w - 0.5f;
        output->pixel_y = (ndc[blockIdx.y][1][index] + 1.0f) * 0.5f * img_h - 0.5f;
        output->depth = ndc[blockIdx.y][2][index];
        output->inv_cov00 = cov2d_inv[blockIdx.y][0][0][index];
        output->inv_cov01 = cov2d_inv[blockIdx.y][0][1][index];
        output->inv_cov11 = cov2d_inv[blockIdx.y][1][1][index];
        output->rg = half2(color[blockIdx.y][0][index], color[blockIdx.y][1][index]);
        output->ba = half2(color[blockIdx.y][2][index], opacity[0][index]);
    }
}

std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor ndc,// 
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    std::optional<at::Tensor>  specific_tiles_arg,
    int64_t img_h,
    int64_t img_w,
    int64_t tile_h,
    int64_t tile_w,
    bool enable_trans,
    bool enable_depth
)
{
    at::DeviceGuard guard(ndc.device());

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
        specific_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
    }
    //pack params
    int points_num = ndc.size(2);
    at::Tensor packed_params = torch::empty({ viewsnum,points_num,sizeof(PackedParams)/sizeof(float)}, ndc.options().requires_grad(false));
    dim3 Block3d(std::ceil(points_num / 512.0f), viewsnum, 1);
    {
        pack_forward_params<<<Block3d,512>>>(
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            img_h, img_w);
    }
    //raster
    
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

    {
        int tiles_per_block = 4;
        dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
        dim3 Thread3d(32, tiles_per_block);
        raster_forward_kernel<8, 16, false, false> << <Block3d, Thread3d >> > (sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
            tilesnum_x);
        CUDA_CHECK_ERRORS;
    }

    return { output_img ,output_transmitance,output_depth ,output_last_contributor,packed_params };
}


std::vector<at::Tensor> rasterize_forward_packed(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor packed_params,
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
    //raster

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

    {
        int tiles_per_block = 4;
        dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
        dim3 Thread3d(32, tiles_per_block);
        raster_forward_kernel<8, 16, false, false> << <Block3d, Thread3d >> > (sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            output_last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
            tilesnum_x);
        CUDA_CHECK_ERRORS;
    }

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
inline __device__ void warp_reduce_sum<unsigned int, false>(unsigned int& data)
{
    data = __reduce_add_sync(0xffffffff, data);
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

template <int tile_size_y, int tile_size_x,bool enable_statistic, bool enable_trans_grad, bool enable_depth_grad>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_params,         // //[batch,point_num,sizeof(PackedParams)/sizeof(float)]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_grad,         //[batch,point_num,9]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color_square_sum,  //[batch,3,point_num]
    torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> fragment_sum,  //[batch,1,point_num]
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

                const int in_tile_x = threadIdx.x % tile_size_x;
                const int in_tile_y = threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;
                float t0 = final_transmitance[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i][in_tile_x];
                float t1 = final_transmitance[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i + 1][in_tile_x];
                reg_buffer[i].t = half2(t0 * SCALER, t1 * SCALER);

                shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                    d_img[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i][in_tile_x],
                    d_img[batch_id][0][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i + 1][in_tile_x]);
                shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                    d_img[batch_id][1][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i][in_tile_x],
                    d_img[batch_id][1][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i + 1][in_tile_x]);
                shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                    d_img[batch_id][2][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i][in_tile_x],
                    d_img[batch_id][2][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i + 1][in_tile_x]);
                
                int last0 = last_contributor[batch_id][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i][in_tile_x] - 1;
                int last1 = last_contributor[batch_id][blockIdx.x * blockDim.y + threadIdx.y][in_tile_y + 2 * i + 1][in_tile_x] - 1;
                index_in_tile = max(max(index_in_tile, last0), last1);
                shared_last_contributor[i][threadIdx.y * blockDim.x + threadIdx.x] = (last1 << 16 | last0);
            }
            index_in_tile = __reduce_max_sync(0xffffffff, index_in_tile);

            const int* points_in_tile = &sorted_points[batch_id][start_index_in_tile];
            const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + threadIdx.x % tile_size_x;
            const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;

            for (; (index_in_tile >= 0); index_in_tile--)
            {
                float basic;
                float bxcy;
                float neg_half_c;
                float2 d{ 0,0 };
                int point_id = points_in_tile[index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                {
                    float2 xy{ params.pixel_x,params.pixel_y};
                    d.x = xy.x - pixel_x;
                    d.y = xy.y - pixel_y;
                    basic = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                    bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                    neg_half_c = -0.5f * params.inv_cov11;
                }//basic+=(cy+bx)*delta - 0.5*c*delta*delta

                RGBA16x2 point_color_x2;
                point_color_x2.r = half2(params.rg.x, params.rg.x);
                point_color_x2.g = half2(params.rg.y, params.rg.y);
                point_color_x2.b = half2(params.ba.x, params.ba.x);
                point_color_x2.a = half2(params.ba.y, params.ba.y);
                

                half2 grad_r = half2(0, 0);
                half2 grad_g = half2(0, 0);
                half2 grad_b = half2(0, 0);
                half2 grad_r_square = half2(0, 0);
                half2 grad_g_square = half2(0, 0);
                half2 grad_b_square = half2(0, 0);
                unsigned int fragment_count = 0x0;//ushort2
                half2 grad_a = half2(0, 0);
                float grad_bxcy = 0;
                float grad_neg_half_c = 0;
                float grad_basic = 0;
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD; i++)
                {
                    half2 power{ basic + 2 * i * bxcy + 2 * i * 2 * i * neg_half_c,
                        basic + (2 * i + 1) * bxcy + (2 * i + 1) * (2 * i + 1) * neg_half_c };
                    half2 G = fast_exp_approx(power);
                    half2 alpha = point_color_x2.a * G;
                    alpha = __hmin2(half2(255.0f / 256, 255.0f / 256), alpha);

                    unsigned int valid_mask = 0xffffffffu;
                    //valid_mask &= __hle2_mask(power, half2(1.0f / (1 << 24), 1.0f / (1 << 24)));//1 ULP:2^(-14) * (0 + 1/1024)
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
                        if (enable_statistic)
                        {
                            fragment_count += (0x00010001u & valid_mask);
                            half2 temp = alpha * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                            grad_r_square += temp * temp;
                            temp = alpha * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                            grad_g_square += temp * temp;
                            temp = alpha * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];
                            grad_b_square += temp * temp;
                        }


                        half2 d_alpha = half2(0,0);
                        d_alpha += (point_color_x2.r - reg_buffer[i].r) * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                        d_alpha += (point_color_x2.g - reg_buffer[i].g) * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                        d_alpha += (point_color_x2.b - reg_buffer[i].b) * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];
                        reg_buffer[i].r += alpha * (point_color_x2.r - reg_buffer[i].r);//0-256
                        reg_buffer[i].g += alpha * (point_color_x2.g - reg_buffer[i].g);
                        reg_buffer[i].b += alpha * (point_color_x2.b - reg_buffer[i].b);

                        grad_a += d_alpha * G;
                        half2 d_G = point_color_x2.a * d_alpha;
                        half2 d_power = G * d_G;
                        half2 grad_bxcy_x2 = d_power * half2(2 * i, 2 * i + 1);
                        half2 grad_neg_half_c_x2 = d_power * half2(2 * i, 2 * i + 1) * half2(2 * i, 2 * i + 1);
                        half2 grad_basic_x2 = d_power;
                        grad_bxcy += ((float)grad_bxcy_x2.x + (float)grad_bxcy_x2.y);
                        grad_neg_half_c+= ((float)grad_neg_half_c_x2.x + (float)grad_neg_half_c_x2.y);
                        grad_basic += ((float)grad_basic_x2.x + (float)grad_basic_x2.y);
                    }
                }
                
                PackedGrad* grad_addr = (PackedGrad*)&packed_grad[batch_id][point_id][0];
                //unsigned mask = __ballot_sync(0xffffffff, grad_opacity!=0);
                if (__any_sync(0xffffffff, grad_a.x!=half(0)|| grad_a.y!=half(0)))
                {
                    half2 rg{ grad_r.x + grad_r.y ,grad_g.x + grad_g.y };
                    half2 ba{ grad_b.x + grad_b.y ,grad_a.x + grad_a.y };
                    warp_reduce_sum<half2, false>(rg);
                    warp_reduce_sum<half2, false>(ba);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&grad_addr->r, float(rg.x)* INV_SCALER);
                        atomicAdd(&grad_addr->g, float(rg.y)* INV_SCALER);
                        atomicAdd(&grad_addr->b, float(ba.x)* INV_SCALER);
                        atomicAdd(&grad_addr->a, float(ba.y)* INV_SCALER);
                    }
                    if (enable_statistic)
                    {
                        float3 grad_square{ float(grad_r_square.x + grad_r_square.y) * INV_SCALER ,
                            float(grad_g_square.x + grad_g_square.y) * INV_SCALER,
                            float(grad_b_square.x + grad_b_square.y) * INV_SCALER
                        };
                        unsigned int reduced_fragment_count = (fragment_count >> 16u) + (fragment_count & 0xffffu);
                        warp_reduce_sum<float3, false>(grad_square);
                        warp_reduce_sum<unsigned int, false>(reduced_fragment_count);
                        if (threadIdx.x == 0)
                        {
                            atomicAdd(&d_color_square_sum[batch_id][0][point_id], grad_square.x);
                            atomicAdd(&d_color_square_sum[batch_id][1][point_id], grad_square.y);
                            atomicAdd(&d_color_square_sum[batch_id][2][point_id], grad_square.z);
                            atomicAdd(&fragment_sum[batch_id][0][point_id], reduced_fragment_count);
                        }
                    }

                    grad_bxcy *= INV_SCALER;
                    grad_neg_half_c *= INV_SCALER;
                    grad_basic *= INV_SCALER;
                    float3 grad_invcov{ 0,0,0 };
                    //basic = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                    //bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                    //neg_half_c = -0.5f * params.inv_cov11;
                    grad_invcov.x = -0.5f * d.x * d.x * grad_basic;
                    grad_invcov.y = (-d.x * d.y * grad_basic + d.x * grad_bxcy) * 0.5f;
                    grad_invcov.z = -0.5f * d.y * d.y * grad_basic + d.y * grad_bxcy - 0.5f * grad_neg_half_c;

                    warp_reduce_sum<float, false>(grad_invcov.x);
                    warp_reduce_sum<float, false>(grad_invcov.y);
                    warp_reduce_sum<float, false>(grad_invcov.z);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&grad_addr->inv_cov00, grad_invcov.x);
                        atomicAdd(&grad_addr->inv_cov01, grad_invcov.y);
                        atomicAdd(&grad_addr->inv_cov11, grad_invcov.z);
                    }

                    float d_dx = (-params.inv_cov00 * d.x - params.inv_cov01 * d.y) * grad_basic + params.inv_cov01 * grad_bxcy;
                    float d_dy = (-params.inv_cov11 * d.y - params.inv_cov01 * d.x) * grad_basic + params.inv_cov11 * grad_bxcy;
                    float2 d_ndc_xy{ d_dx * 0.5f * img_w,d_dy * 0.5f * img_h };
                    warp_reduce_sum<float2, false>(d_ndc_xy);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&grad_addr->ndc_x, d_ndc_xy.x);
                        atomicAdd(&grad_addr->ndc_y, d_ndc_xy.y);
                    }
                }
            }
        }
    }
}

__global__ void unpack_gradient(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_grad,//[batch,point_num,property_num]
    const float* grad_inv_scaler,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity          //[1,point_num]
)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < packed_grad.size(1))
    {
        PackedGrad* grads = (PackedGrad*)&packed_grad[blockIdx.y][index][0];
        d_ndc[blockIdx.y][0][index] = grads->ndc_x * grad_inv_scaler[0];
        d_ndc[blockIdx.y][1][index] = grads->ndc_y * grad_inv_scaler[0];
        d_cov2d_inv[blockIdx.y][0][0][index] = grads->inv_cov00 * grad_inv_scaler[0];
        d_cov2d_inv[blockIdx.y][0][1][index] = grads->inv_cov01 * grad_inv_scaler[0];
        d_cov2d_inv[blockIdx.y][1][0][index] = grads->inv_cov01 * grad_inv_scaler[0];
        d_cov2d_inv[blockIdx.y][1][1][index] = grads->inv_cov11 * grad_inv_scaler[0];
        d_color[blockIdx.y][0][index] = grads->r * grad_inv_scaler[0];
        d_color[blockIdx.y][1][index] = grads->g * grad_inv_scaler[0];
        d_color[blockIdx.y][2][index] = grads->b * grad_inv_scaler[0];
        if (blockIdx.y == 0)//todo fix
        {
            d_opacity[0][index] = grads->a * grad_inv_scaler[0];
        }
    }
}

std::vector<at::Tensor> rasterize_backward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor packed_params,
    std::optional<at::Tensor> specific_tiles_arg,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    std::optional<at::Tensor> d_trans_img_arg,
    std::optional<at::Tensor> d_depth_img_arg,
    std::optional<at::Tensor> grad_inv_sacler_arg,
    int64_t img_h,
    int64_t img_w,
    int64_t tilesize_h,
    int64_t tilesize_w,
    bool enable_statistic
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
    at::Tensor grad_inv_sacler;
    if (grad_inv_sacler_arg.has_value())
    {
        grad_inv_sacler = *grad_inv_sacler_arg;
    }
    else
    {
        grad_inv_sacler = torch::ones({ 1 }, d_img.options());
    }
    int batch_num = packed_params.size(0);
    int points_num = packed_params.size(1);
    at::Tensor d_ndc = torch::zeros({ batch_num,4,points_num }, packed_params.options());
    at::Tensor d_cov2d_inv = torch::zeros({ batch_num,2,2,points_num }, packed_params.options());
    at::Tensor d_color = torch::zeros({ batch_num,3,points_num }, packed_params.options());
    at::Tensor d_opacity = torch::zeros({ 1,points_num }, packed_params.options());
    at::Tensor packed_grad = torch::zeros({ batch_num,points_num,sizeof(PackedGrad)/sizeof(float)}, packed_params.options());
    at::Tensor d_color_square_sum = torch::zeros({ batch_num,3,points_num }, packed_params.options());
    at::Tensor fragment_sum = torch::zeros({ batch_num,1,points_num }, packed_params.options().dtype(torch::kInt32));
    
    int tiles_per_block = 4;
    dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
    dim3 Thread3d(32, tiles_per_block);
    //dim3 Block3d(1, viewsnum, 1);
    //dim3 Thread3d(32, 1);
    if (enable_statistic == false)
    {
        raster_backward_kernel<8, 16, false, false, false> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            packed_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            d_color_square_sum.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            fragment_sum.packed_accessor32<int, 3, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w
            );
    }
    else
    {
        raster_backward_kernel<8, 16, true, false, false> << <Block3d, Thread3d >> > (
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),
            last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
            d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            packed_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            d_color_square_sum.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
            fragment_sum.packed_accessor32<int, 3, torch::RestrictPtrTraits >(),
            tilesnum_x, img_h, img_w
            );
    }

    CUDA_CHECK_ERRORS;

    dim3 UnpackBlock3d(std::ceil(points_num / 512.0f), batch_num, 1);
    unpack_gradient<<<UnpackBlock3d,512>>>(
        packed_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        (float*)grad_inv_sacler.data_ptr(),
        d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
        d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >());
    CUDA_CHECK_ERRORS;

    return { d_ndc ,d_cov2d_inv ,d_color,d_opacity,d_color_square_sum,fragment_sum };
}
