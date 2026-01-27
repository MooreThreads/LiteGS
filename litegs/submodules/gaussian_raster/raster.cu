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

#include <ATen/core/TensorAccessor.h>

#include "cuda_errchk.h"
#include "raster.h"


struct __align__(16) PackedParams
{
    float pixel_x;
    float pixel_y;
    float depth;
    half2 rg;
    float inv_cov00;
    float inv_cov01;
    float inv_cov11;
    half2 ba;
};

struct PackedGrad
{
    float dx;
    float dy;
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


template<class T, bool boardcast>
inline __device__ void warp_reduce_sum(T& data,unsigned int active_mask)
{
    data += __shfl_down_sync(active_mask, data, 16);
    data += __shfl_down_sync(active_mask, data, 8);
    data += __shfl_down_sync(active_mask, data, 4);
    data += __shfl_down_sync(active_mask, data, 2);
    data += __shfl_down_sync(active_mask, data, 1);
    if (boardcast)
        data = __shfl_sync(active_mask, data, 0);
}

template<>
inline __device__ void warp_reduce_sum<unsigned int, false>(unsigned int& data, unsigned int active_mask)
{
    data = __reduce_add_sync(active_mask, data);
}

template<>
inline __device__ void warp_reduce_sum<float, false>(float& data, unsigned int active_mask)
{
    int exponent = (__float_as_uint(data) >> 23) & 0xff;
    exponent = __reduce_max_sync(active_mask, exponent) - 127;
    int scale_exponent = 23 - exponent;
    bool valid = (exponent > -127) && (scale_exponent < 128);

    float scaler = __uint_as_float(0 | ((scale_exponent + 127) << 23));
    float inv_scaler = __uint_as_float(0 | ((127 - scale_exponent) << 23));
    int scaled_value = static_cast<int>(data * scaler);
    scaled_value = __reduce_add_sync(active_mask, scaled_value) * valid;

    data = scaled_value * inv_scaler;
}

template<>
inline __device__ void warp_reduce_sum<float2, false>(float2& data, unsigned int active_mask)
{
    int exponent = (__float_as_uint(data.x) >> 23) & 0xff;
    exponent = max(exponent, (__float_as_uint(data.y) >> 23) & 0xff);
    exponent = __reduce_max_sync(active_mask, exponent) - 127;
    int scale_exponent = 23 - exponent;
    bool valid = (exponent > -127) && (scale_exponent < 128);

    float scaler = __uint_as_float(0 | ((scale_exponent + 127) << 23));
    float inv_scaler = __uint_as_float(0 | ((127 - scale_exponent) << 23));

    int scaled_value_x = static_cast<int>(data.x * scaler);
    scaled_value_x = __reduce_add_sync(active_mask, scaled_value_x) * valid;
    data.x = scaled_value_x * inv_scaler;
    int scaled_value_y = static_cast<int>(data.y * scaler);
    scaled_value_y = __reduce_add_sync(active_mask, scaled_value_y) * valid;
    data.y = scaled_value_y * inv_scaler;
}

template<>
inline __device__ void warp_reduce_sum<float3, false>(float3& data, unsigned int active_mask)
{
    int exponent = (__float_as_uint(data.x) >> 23) & 0xff;
    exponent = max(exponent, (__float_as_uint(data.y) >> 23) & 0xff);
    exponent = max(exponent, (__float_as_uint(data.z) >> 23) & 0xff);
    exponent = __reduce_max_sync(active_mask, exponent) - 127;
    int scale_exponent = 23 - exponent;
    bool valid = (exponent > -127) && (scale_exponent < 128);

    float scaler = __uint_as_float(0 | ((scale_exponent + 127) << 23));
    float inv_scaler = __uint_as_float(0 | ((127 - scale_exponent) << 23));

    int scaled_value_x = static_cast<int>(data.x * scaler);
    scaled_value_x = __reduce_add_sync(active_mask, scaled_value_x) * valid;
    data.x = scaled_value_x * inv_scaler;
    int scaled_value_y = static_cast<int>(data.y * scaler);
    scaled_value_y = __reduce_add_sync(active_mask, scaled_value_y) * valid;
    data.y = scaled_value_y * inv_scaler;
    int scaled_value_z = static_cast<int>(data.z * scaler);
    scaled_value_z = __reduce_add_sync(active_mask, scaled_value_z) * valid;
    data.z = scaled_value_z * inv_scaler;
}



template <int tile_size_y, int tile_size_x,int subtile_size_y,int subtile_size_x, bool enable_statistic, bool enable_trans, bool enable_depth>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_tile,    //[batch,items]  
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> end_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_subtile,    //[batch,items] 
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> subtile_start,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> subtile_end,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> heavy_tiles,          //[batch,tiles_num]
    const PackedParams* __restrict__ packed_params,         //[batch,point_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_depth,     //[batch,1,tile,tilesize, tilesize]
    torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> out_fragment_count,  //[batch,1,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_fragment_weight_sum,  //[batch,1,point_num]
    int tiles_num_x, int pointsnum)
{
    //assert blockDim.x==32

    constexpr int VECTOR_SIZE = 2;
    constexpr int PIXELS_PER_THREAD = (tile_size_x * tile_size_y) / (32* VECTOR_SIZE);//half2: 32 pixel per warp->64 pixel per warp
    constexpr float SCALER = 128.0f;
    constexpr float INV_SCALER = 1.0f / 128;

    const int batch_id = blockIdx.y;
    packed_params += batch_id * pointsnum;

    if (heavy_tiles.size(1) <= blockIdx.x)
    {
        int task_id = (blockIdx.x - heavy_tiles.size(1)) * blockDim.y + threadIdx.y;
        int tile_id = task_id + 1;// +1, tile_id 0 is invalid
        if (specific_tiles.size(1) != 0)
        {
            if (task_id < specific_tiles.size(1))
            {
                tile_id = specific_tiles[batch_id][task_id];
            }
            else
            {
                tile_id = 0;
            }
        }

        if (tile_id != 0 && tile_id < start_index.size(1))
        {

            int start_index_in_tile = start_index[batch_id][tile_id];
            int end_index_in_tile = end_index[batch_id][tile_id];
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
                auto points_id_in_tile = &primitives_in_tile[batch_id][start_index_in_tile];
                for (; (index_in_tile + start_index_in_tile < end_index_in_tile) && __any_sync(0xffffffff,any_active != 0); index_in_tile++)
                {
                    int point_id = points_id_in_tile[index_in_tile];
                    PackedParams params = packed_params[point_id];
                    RGBA16x2 point_color_x2;
                    point_color_x2.r = half2(params.rg.x, params.rg.x);
                    point_color_x2.g = half2(params.rg.y, params.rg.y);
                    point_color_x2.b = half2(params.ba.x, params.ba.x);
                    point_color_x2.a = half2(params.ba.y, params.ba.y);
                    float2 xy{ params.pixel_x,params.pixel_y };

                    const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + threadIdx.x % tile_size_x;
                    const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;
                    float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                    float bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                    float axby = params.inv_cov00 * d.x + params.inv_cov01 * d.y;
                    float cur_val = -0.5f * (d.x * axby + d.y * bxcy);
                    float cur_diff = bxcy - 0.5f * params.inv_cov11;
                    float second_diff = -params.inv_cov11;
                    //basic+=(cy+bx)*delta - 0.5*c*delta*delta

                    any_active = 0;
                    unsigned int fragment_count = 0x0;//ushort2
                    half2 weight_sum = half2(0, 0);
#pragma unroll
                    for (int i = 0; i < PIXELS_PER_THREAD; i++)
                    {
                        half2 power;
                        power.x = cur_val;
                        cur_val += cur_diff;
                        cur_diff += second_diff;
                        power.y = cur_val;
                        cur_val += cur_diff;
                        cur_diff += second_diff;

                        unsigned int active_mask = 0xffffffffu;
                        active_mask = __hgt2_mask(reg_buffer[i].t, half2(SCALER / 8192, SCALER / 8192));
                        any_active |= active_mask;

                        unsigned int alpha_valid_mask = active_mask;
                        //alpha_valid_mask &= __hle2_mask(power, half2(1.0f / (1 << 24), 1.0f / (1 << 24)));//1 ULP:2^(-14) * (0 + 1/1024)
                        reg_buffer[i].alpha = point_color_x2.a * fast_exp_approx(power);
                        alpha_valid_mask &= __hge2_mask(reg_buffer[i].alpha, half2(1.0f / 256, 1.0f / 256));
                        reg_buffer[i].alpha = __hmin2(half2(255.0f / 256, 255.0f / 256), reg_buffer[i].alpha);

                        reg_buffer[i].lst_contributor += (0x00010001 & active_mask);
                        reinterpret_cast<unsigned int*>(&reg_buffer[i].alpha)[0] &= alpha_valid_mask;

                        half2 weight = reg_buffer[i].t * reg_buffer[i].alpha;
                        if (enable_statistic)
                        {
                            fragment_count += (0x00010001u & alpha_valid_mask);
                            weight_sum += weight;
                        }

                        reg_buffer[i].r += (point_color_x2.r * weight);
                        reg_buffer[i].g += (point_color_x2.g * weight);
                        reg_buffer[i].b += (point_color_x2.b * weight);
                        reg_buffer[i].t = reg_buffer[i].t * (half2(1.0f, 1.0f) - reg_buffer[i].alpha);
                    }
                    //reg_buffer[1].alpha = (half2(2.0f, 2.0f) * reg_buffer[0].alpha + reg_buffer[3].alpha) * half2(1.0f / 3, 1.0f / 3);
                    //reg_buffer[2].alpha = (reg_buffer[0].alpha + half2(2.0f, 2.0f) * reg_buffer[3].alpha) * half2(1.0f / 3, 1.0f / 3);


                    //reduce statistic
                    if (enable_statistic)
                    {
                        unsigned int reduced_fragment_count = (fragment_count >> 16u) + (fragment_count & 0xffffu);
                        warp_reduce_sum<unsigned int, false>(reduced_fragment_count,0xffffffff);
                        float weight_sum_f32 = float(weight_sum.x + weight_sum.y) * INV_SCALER;
                        warp_reduce_sum<float, false>(weight_sum_f32, 0xffffffff);
                        if (threadIdx.x == 0)
                        {
                            atomicAdd(&out_fragment_count[batch_id][0][point_id], reduced_fragment_count);
                            atomicAdd(&out_fragment_weight_sum[batch_id][0][point_id], weight_sum_f32);
                        }

                    }
                }

            }
            int tile_index = tile_id - 1;
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
    else if (blockIdx.x < heavy_tiles.size(1))
    {
        const int warp_id = threadIdx.y;
        const int lane_id = threadIdx.x;
        const int tile_id = heavy_tiles[batch_id][blockIdx.x];
        const int subtile_id = (tile_id << 2) + warp_id;

        const int x_in_tile = warp_id * subtile_size_x + lane_id % subtile_size_x;
        const int y_in_tile = lane_id / subtile_size_x;

        const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + x_in_tile;
        const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + y_in_tile;

        if (tile_id != 0)
        {
            int start_index_in_tile = subtile_start[batch_id][subtile_id];
            int end_index_in_tile = subtile_end[batch_id][subtile_id];

            float transmittance = 1.0f;
            float depth = 0.0f;
            float3 final_color{ 0,0,0 };
            int last_contributor = 0;
            bool pixel_active = true;
            for (int index_in_tile = 0; (index_in_tile + start_index_in_tile < end_index_in_tile) && __any_sync(0xffffffff, pixel_active); index_in_tile++)
            {
                int point_id = primitives_in_subtile[batch_id][start_index_in_tile + index_in_tile];
                PackedParams params = packed_params[point_id];
                float3 cur_color{ params.rg.x,params.rg.y,params.ba.x };
                float cur_opacity = params.ba.y;
                float2 xy{ params.pixel_x,params.pixel_y };
                float2 d = { xy.x - pixel_x,xy.y - pixel_y };

                float power = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y) - params.inv_cov01 * d.x * d.y;
                float alpha = min(255.0f / 256, cur_opacity * __expf(power));
                alpha = ((alpha < 1.0f / 256) || pixel_active == false) ? 0 : alpha;

                float weight = alpha * transmittance;
                if (enable_statistic)
                {
                    unsigned int frag_mask = __ballot_sync(0xffffffff, alpha > 0.0f);
                    int fragment_count = __popc(frag_mask);
                    warp_reduce_sum<float, false>(weight, 0xffffffff);
                    if (lane_id == 0)
                    {
                        atomicAdd(&out_fragment_count[batch_id][0][point_id], fragment_count);
                        atomicAdd(&out_fragment_weight_sum[batch_id][0][point_id], weight);
                    }
                }
                final_color.x += cur_color.x * weight;
                final_color.y += cur_color.y * weight;
                final_color.z += cur_color.z * weight;
                if (enable_depth)
                {
                    depth += params.depth * weight;
                }
                transmittance *= (1 - alpha);
                last_contributor += pixel_active;

                pixel_active = (transmittance > 1.0f / 8192);
            }

            output_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile] = final_color.x;
            output_img[batch_id][1][tile_id - 1][y_in_tile][x_in_tile] = final_color.y;
            output_img[batch_id][2][tile_id - 1][y_in_tile][x_in_tile] = final_color.z;
            output_transmitance[batch_id][0][tile_id - 1][y_in_tile][x_in_tile] = transmittance;
            if (enable_depth)
            {
                output_depth[batch_id][0][tile_id - 1][y_in_tile][x_in_tile] = depth;
            }
            output_last_contributor[batch_id][tile_id - 1][y_in_tile][x_in_tile] = last_contributor;

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

#define RASTER_FORWARD_PARAMS primitives_in_tile.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
tile_start.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
tile_end.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
primitives_in_subtile.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
subtile_start.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
subtile_end.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
heavy_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
reinterpret_cast<PackedParams*>(packed_params.data_ptr<float>()),\
output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),\
fragment_count.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),\
fragment_weight_sum.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
tilesnum_x,packed_params.size(1)

#define ENCODE(STATISTIC, TRANS, DEPTH) (((STATISTIC)*1)<<2)|(((TRANS)*1)<<1)|((DEPTH)*1)

#define LAUNCH_RASTER_FORWARD_KERNEL(TILE_H, TILE_W, STATISTIC, TRANS, DEPTH) \
    raster_forward_kernel<TILE_H, TILE_W, 8, 4, STATISTIC, TRANS, DEPTH> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);

#define DISPATCH_RASTER_FORWARD_KERNEL(STATISTIC, TRANS, DEPTH) \
    if (tile_h == 8 && tile_w == 16) { \
        LAUNCH_RASTER_FORWARD_KERNEL(8, 16, STATISTIC, TRANS, DEPTH); } \
    else if (tile_h == 12 && tile_w == 16) { \
        LAUNCH_RASTER_FORWARD_KERNEL(12, 16, STATISTIC, TRANS, DEPTH); } \
    else if (tile_h == 16 && tile_w == 16) { \
        LAUNCH_RASTER_FORWARD_KERNEL(16, 16, STATISTIC, TRANS, DEPTH); } \
    else if (tile_h == 8 && tile_w == 8) { \
        LAUNCH_RASTER_FORWARD_KERNEL(8, 8, STATISTIC, TRANS, DEPTH); }


std::vector<at::Tensor> rasterize_forward(
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, std::optional<at::Tensor>  specific_tiles_arg,
    std::optional<at::Tensor> primitives_in_subtile_arg, std::optional<at::Tensor> subtile_start_arg, std::optional<at::Tensor> subtile_end_arg, std::optional<at::Tensor>  heavy_tiles_arg,
    at::Tensor ndc,// 
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    int64_t img_h,
    int64_t img_w,
    int64_t tile_h,
    int64_t tile_w,
    bool enable_statistic,
    bool enable_trans,
    bool enable_depth
)
{
    at::DeviceGuard guard(ndc.device());

    int64_t viewsnum = tile_start.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    int64_t render_tile_num = tilesnum;
    int64_t heavy_tile_num = 0;
    at::Tensor specific_tiles;
    at::Tensor heavy_tiles;
    at::Tensor primitives_in_subtile;
    at::Tensor subtile_start;
    at::Tensor subtile_end;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        render_tile_num = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
    }
    if (heavy_tiles_arg.has_value())
    {
        heavy_tiles = *heavy_tiles_arg;
        heavy_tile_num = heavy_tiles.size(1);
        primitives_in_subtile = *primitives_in_subtile_arg;
        subtile_start = *subtile_start_arg;
        subtile_end = *subtile_end_arg;
    }
    else
    {
        heavy_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
        primitives_in_subtile = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
        subtile_start = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
        subtile_end = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
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
    
    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(tile_start.device()).requires_grad(true);
    at::Tensor output_img = torch::empty({ viewsnum,3, tilesnum,tile_h,tile_w }, opt_img);

    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(tile_start.device()).requires_grad(enable_trans);
    at::Tensor output_transmitance = torch::empty({ viewsnum,1, tilesnum, tile_h, tile_w }, opt_t);

    at::Tensor output_depth = torch::empty({ 0,0, 0, 0, 0 }, opt_t);
    if (enable_depth)
    {
        output_depth = torch::empty({ viewsnum,1, tilesnum, tile_h, tile_w }, opt_t.requires_grad(true));
    }

    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kShort).layout(torch::kStrided).device(tile_start.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty({ viewsnum, tilesnum, tile_h, tile_w }, opt_c);

    at::Tensor fragment_count = torch::zeros({ viewsnum,1,points_num }, packed_params.options().dtype(torch::kI32));
    at::Tensor fragment_weight_sum = torch::zeros({ viewsnum,1,points_num }, packed_params.options().dtype(torch::kFloat32));

    {
        int tiles_per_block = 4;
        dim3 Block3d(heavy_tile_num + std::ceil(render_tile_num / float(tiles_per_block)), viewsnum, 1);
        dim3 Thread3d(32, tiles_per_block);
        switch (ENCODE(enable_statistic, enable_trans, enable_depth))
        {
        case ENCODE(false,false,false):
            DISPATCH_RASTER_FORWARD_KERNEL(false, false, false)
            break;
        case ENCODE(true, false, false):
            DISPATCH_RASTER_FORWARD_KERNEL(true, false, false)
            break;
        case ENCODE(false, true, false):
            DISPATCH_RASTER_FORWARD_KERNEL(false, true, false)
            break;
        case ENCODE(false, false, true):
            DISPATCH_RASTER_FORWARD_KERNEL(false, false, true)
            break;
        case ENCODE(true, true, false):
            DISPATCH_RASTER_FORWARD_KERNEL(true, true, false)
            break;
        case ENCODE(true, false, true):
            DISPATCH_RASTER_FORWARD_KERNEL(true, false, true)
            break;
        case ENCODE(false, true, true):
            DISPATCH_RASTER_FORWARD_KERNEL(false, true, true)
            break;
        case ENCODE(true, true, true):
            DISPATCH_RASTER_FORWARD_KERNEL(true, true, true)
            break;
        default:
            break;
        }
        CUDA_CHECK_ERRORS;
    }

    return { output_img ,output_transmitance,output_depth ,output_last_contributor,packed_params,fragment_count,fragment_weight_sum };
}


std::vector<at::Tensor> rasterize_forward_packed(
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, std::optional<at::Tensor>  specific_tiles_arg,
    std::optional<at::Tensor> primitives_in_subtile_arg, std::optional<at::Tensor> subtile_start_arg, std::optional<at::Tensor> subtile_end_arg, std::optional<at::Tensor>  heavy_tiles_arg,
    at::Tensor packed_params,
    int64_t img_h,
    int64_t img_w,
    int64_t tile_h,
    int64_t tile_w,
    bool enable_statistic,
    bool enable_trans,
    bool enable_depth
)
{
    at::DeviceGuard guard(packed_params.device());

    int64_t viewsnum = tile_start.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    int64_t render_tile_num = tilesnum;
    int64_t heavy_tile_num = 0;
    at::Tensor specific_tiles;
    at::Tensor heavy_tiles;
    at::Tensor primitives_in_subtile;
    at::Tensor subtile_start;
    at::Tensor subtile_end;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        render_tile_num = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
    }
    if (heavy_tiles_arg.has_value())
    {
        heavy_tiles = *heavy_tiles_arg;
        heavy_tile_num = heavy_tiles.size(1);
        primitives_in_subtile = *primitives_in_subtile_arg;
        subtile_start = *subtile_start_arg;
        subtile_end = *subtile_end_arg;
    }
    else
    {
        heavy_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
        primitives_in_subtile = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
        subtile_start = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
        subtile_end = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
    }
    //raster

    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(tile_start.device()).requires_grad(true);
    at::Tensor output_img = torch::empty({ viewsnum,3, tilesnum,tile_h,tile_w }, opt_img);

    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(tile_start.device()).requires_grad(enable_trans);
    at::Tensor output_transmitance = torch::empty({ viewsnum,1, tilesnum, tile_h, tile_w }, opt_t);

    at::Tensor output_depth = torch::empty({ 0,0, 0, 0, 0 }, opt_t);
    if (enable_depth)
    {
        output_depth = torch::empty({ viewsnum,1, tilesnum, tile_h, tile_w }, opt_t.requires_grad(true));
    }

    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kShort).layout(torch::kStrided).device(tile_start.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty({ viewsnum, tilesnum, tile_h, tile_w }, opt_c);

    int points_num = packed_params.size(1);
    at::Tensor fragment_count = torch::zeros({ viewsnum,1,points_num }, packed_params.options().dtype(torch::kI32));
    at::Tensor fragment_weight_sum = torch::zeros({ viewsnum,1,points_num }, packed_params.options().dtype(torch::kFloat32));

    {
        int tiles_per_block = 4;
        dim3 Block3d(heavy_tile_num + std::ceil(render_tile_num / float(tiles_per_block)), viewsnum, 1);
        dim3 Thread3d(32, tiles_per_block);
        switch (ENCODE(enable_statistic, enable_trans, enable_depth))
        {
        case ENCODE(false, false, false):
            DISPATCH_RASTER_FORWARD_KERNEL(false, false, false)
            break;
        case ENCODE(true, false, false):
            DISPATCH_RASTER_FORWARD_KERNEL(true, false, false)
            break;
        case ENCODE(false, true, false):
            DISPATCH_RASTER_FORWARD_KERNEL(false, true, false)
            break;
        case ENCODE(false, false, true):
            DISPATCH_RASTER_FORWARD_KERNEL(false, false, true)
            break;
        case ENCODE(true, true, false):
            DISPATCH_RASTER_FORWARD_KERNEL(true, true, false)
            break;
        case ENCODE(true, false, true):
            DISPATCH_RASTER_FORWARD_KERNEL(true, false, true)
            break;
        case ENCODE(false, true, true):
            DISPATCH_RASTER_FORWARD_KERNEL(false, true, true)
            break;
        case ENCODE(true, true, true):
            DISPATCH_RASTER_FORWARD_KERNEL(true, true, true)
            break;
        default:
            break;
        }
        CUDA_CHECK_ERRORS;
    }

    return { output_img ,output_transmitance,output_depth ,output_last_contributor,fragment_count,fragment_weight_sum };
}


struct BackwardRegisterBuffer
{
    half2 r;
    half2 g;
    half2 b;
    half2 t;
    half2 alpha;
};


template <int tile_size_y, int tile_size_x, int subtile_size_y, int subtile_size_x, bool enable_statistic, bool enable_trans_grad, bool enable_depth_grad>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_tile,    //[batch,items]  
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_subtile,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> subtile_start,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> heavy_tiles,          //[batch,tiles_num]
    const PackedParams* __restrict__ packed_params,         //[batch,point_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    PackedGrad* __restrict__ packed_grad,         //[batch,point_num,9]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_err_sum,  //[batch,1,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_err_square_sum,  //[batch,1,point_num]
    int tiles_num_x, int img_h, int img_w,int pointsnum)
{
    constexpr int VECTOR_SIZE = 2;
    constexpr int PIXELS_PER_THREAD = (tile_size_x * tile_size_y) / (32 * VECTOR_SIZE);//half2: 32 pixel per warp->64 pixel per warp
    constexpr float SCALER = 128.0f;
    constexpr float INV_SCALER = 1.0f / 128;//the recovery of gradient is move to "unpack_gradient" kernel 

    __shared__ half2 shared_img_grad[3][PIXELS_PER_THREAD][4 * 32];
    __shared__ half2 shared_trans_grad_buffer[PIXELS_PER_THREAD][4 * 32];
    __shared__ unsigned int shared_last_contributor[PIXELS_PER_THREAD][4 * 32];//ushort2

    const int batch_id = blockIdx.y;
    packed_params += batch_id * pointsnum;
    packed_grad += batch_id * pointsnum;

    if (heavy_tiles.size(1) <= blockIdx.x )
    {
        int task_id = (blockIdx.x - heavy_tiles.size(1)) * blockDim.y + threadIdx.y;
        int tile_id = task_id + 1;// +1, tile_id 0 is invalid
        if (specific_tiles.size(1) != 0)
        {
            if (task_id < specific_tiles.size(1))
            {
                tile_id = specific_tiles[batch_id][task_id];
            }
            else
            {
                tile_id = 0;
            }
        }

        if (tile_id != 0 && tile_id < start_index.size(1))
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
                    float t0 = final_transmitance[batch_id][0][tile_id - 1][in_tile_y + 2 * i][in_tile_x];
                    float t1 = final_transmitance[batch_id][0][tile_id - 1][in_tile_y + 2 * i + 1][in_tile_x];
                    reg_buffer[i].t = half2(t0 * SCALER, t1 * SCALER);
                    if (enable_trans_grad)
                    {
                        shared_trans_grad_buffer[i][threadIdx.y * blockDim.x + threadIdx.x] = reg_buffer[i].t *
                            half2(d_trans_img[batch_id][0][tile_id - 1][in_tile_y + 2 * i][in_tile_x],
                                d_trans_img[batch_id][0][tile_id - 1][in_tile_y + 2 * i + 1][in_tile_x]);
                    }

                    shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                        d_img[batch_id][0][tile_id - 1][in_tile_y + 2 * i][in_tile_x],
                        d_img[batch_id][0][tile_id - 1][in_tile_y + 2 * i + 1][in_tile_x]);
                    shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                        d_img[batch_id][1][tile_id - 1][in_tile_y + 2 * i][in_tile_x],
                        d_img[batch_id][1][tile_id - 1][in_tile_y + 2 * i + 1][in_tile_x]);
                    shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                        d_img[batch_id][2][tile_id - 1][in_tile_y + 2 * i][in_tile_x],
                        d_img[batch_id][2][tile_id - 1][in_tile_y + 2 * i + 1][in_tile_x]);
                    if (enable_trans_grad)
                    {
                        shared_img_grad[3][i][threadIdx.y * blockDim.x + threadIdx.x] = half2(
                            d_trans_img[batch_id][0][tile_id - 1][in_tile_y + 2 * i][in_tile_x],
                            d_trans_img[batch_id][0][tile_id - 1][in_tile_y + 2 * i + 1][in_tile_x]);
                    }
                    unsigned short last0 = last_contributor[batch_id][tile_id - 1][in_tile_y + 2 * i][in_tile_x];
                    last0 = last0 == 0 ? 0 : last0 - 1;
                    unsigned short last1 = last_contributor[batch_id][tile_id - 1][in_tile_y + 2 * i + 1][in_tile_x];
                    last1 = last1 == 0 ? 0 : last1 - 1;
                    index_in_tile = max(max(index_in_tile, last0), last1);
                    shared_last_contributor[i][threadIdx.y * blockDim.x + threadIdx.x] = (last1 << 16 | last0);
                }
                index_in_tile = __reduce_max_sync(0xffffffff, index_in_tile);

                const int* points_in_tile = &primitives_in_tile[batch_id][start_index_in_tile];
                const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + threadIdx.x % tile_size_x;
                const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;

                int point_id = 0;
                int prefetch_point_id = points_in_tile[index_in_tile];
                PackedParams params = packed_params[prefetch_point_id];

                for (; (index_in_tile >= 0); index_in_tile--)
                {
                    point_id = prefetch_point_id;
                    float cur_val;
                    float cur_diff;
                    float second_diff;
                    float2 d{ 0,0 };
                    //Forward Difference
                    {
                        float2 xy{ params.pixel_x,params.pixel_y };
                        d.x = xy.x - pixel_x;
                        d.y = xy.y - pixel_y;
                        float bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                        float axby = params.inv_cov00 * d.x + params.inv_cov01 * d.y;
                        cur_val = -0.5f * (d.x * axby + d.y * bxcy);
                        cur_diff = bxcy - 0.5f * params.inv_cov11;
                        second_diff = -params.inv_cov11;

                    }

                    RGBA16x2 point_color_x2;
                    point_color_x2.r = half2(params.rg.x, params.rg.x);
                    point_color_x2.g = half2(params.rg.y, params.rg.y);
                    point_color_x2.b = half2(params.ba.x, params.ba.x);
                    point_color_x2.a = half2(params.ba.y, params.ba.y);

                    prefetch_point_id = points_in_tile[max(index_in_tile - 1,0)];
                    params = packed_params[prefetch_point_id];

                    half2 grad_r = half2(0, 0);
                    half2 grad_g = half2(0, 0);
                    half2 grad_b = half2(0, 0);
                    half2 err_square = half2(0, 0);
                    half2 grad_a = half2(0, 0);
                    float grad_bxcy = 0;
                    float grad_neg_half_c = 0;
                    float grad_basic = 0;
#pragma unroll
                    for (int i = 0; i < PIXELS_PER_THREAD; i++)
                    {

                        half2 power;
                        power.x = cur_val;
                        cur_val += cur_diff;
                        cur_diff += second_diff;
                        power.y = cur_val;
                        cur_val += cur_diff;
                        cur_diff += second_diff;

                        half2 G = fast_exp_approx(power);
                        half2 alpha = point_color_x2.a * G;
                        alpha = __hmin2(half2(255.0f / 256, 255.0f / 256), alpha);

                        unsigned int valid_mask = 0xffffffffu;
                        //valid_mask &= __hle2_mask(power, half2(1.0f / (1 << 24), 1.0f / (1 << 24)));//1 ULP:2^(-14) * (0 + 1/1024)
                        valid_mask &= __hge2_mask(alpha, half2(1.0f / 256, 1.0f / 256));
                        valid_mask &= __vcmpleu2(index_in_tile << 16 | index_in_tile, shared_last_contributor[i][threadIdx.y * blockDim.x + threadIdx.x]);

                        if (__any_sync(0xffffffff, valid_mask != 0))
                        {
                            reinterpret_cast<unsigned int*>(&alpha)[0] &= valid_mask;
                            reinterpret_cast<unsigned int*>(&G)[0] &= valid_mask;

                            reg_buffer[i].t = __hmin2(half2(SCALER, SCALER), reg_buffer[i].t * h2rcp(half2(1.0f, 1.0f) - alpha));//0-2^(-10)
                            grad_r += alpha * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                            grad_g += alpha * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                            grad_b += alpha * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];


                            half2 d_alpha = half2(0, 0);
                            d_alpha += (point_color_x2.r - reg_buffer[i].r) * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                            d_alpha += (point_color_x2.g - reg_buffer[i].g) * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                            d_alpha += (point_color_x2.b - reg_buffer[i].b) * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];
                            reg_buffer[i].r += alpha * (point_color_x2.r - reg_buffer[i].r);//0-256
                            reg_buffer[i].g += alpha * (point_color_x2.g - reg_buffer[i].g);
                            reg_buffer[i].b += alpha * (point_color_x2.b - reg_buffer[i].b);
                            if (enable_trans_grad)
                            {
                                d_alpha -= shared_trans_grad_buffer[i][threadIdx.y * blockDim.x + threadIdx.x] * h2rcp(half2(1.0f, 1.0f) - alpha);
                            }

                            grad_a += d_alpha * G;
                            half2 d_G = point_color_x2.a * d_alpha;
                            half2 d_power = G * d_G;//G * point_alpha * d_alpha
                            if (enable_statistic)
                            {
                                half2 cur_err = grad_a;
                                //err += cur_err;
                                err_square += (cur_err * half2(INV_SCALER, INV_SCALER) * cur_err);
                            }
                            half2 grad_bxcy_x2 = d_power * half2(2 * i, 2 * i + 1);
                            half2 grad_neg_half_c_x2 = d_power * half2(2 * i, 2 * i + 1) * half2(2 * i, 2 * i + 1);
                            half2 grad_basic_x2 = d_power;
                            grad_bxcy += ((float)grad_bxcy_x2.x + (float)grad_bxcy_x2.y);
                            grad_neg_half_c += ((float)grad_neg_half_c_x2.x + (float)grad_neg_half_c_x2.y);
                            grad_basic += ((float)grad_basic_x2.x + (float)grad_basic_x2.y);
                        }
                    }

                    PackedGrad* grad_addr = packed_grad + point_id;
                    //unsigned mask = __ballot_sync(0xffffffff, grad_opacity!=0);
                    if (__any_sync(0xffffffff, grad_a.x != half(0) || grad_a.y != half(0)))
                    {
                        half2 rg{ grad_r.x + grad_r.y ,grad_g.x + grad_g.y };
                        half2 ba{ grad_b.x + grad_b.y ,grad_a.x + grad_a.y };
                        warp_reduce_sum<half2, false>(rg, 0xffffffff);
                        warp_reduce_sum<half2, false>(ba, 0xffffffff);
                        if (threadIdx.x == 0)
                        {
                            atomicAdd(&grad_addr->r, float(rg.x));
                            atomicAdd(&grad_addr->g, float(rg.y));
                            atomicAdd(&grad_addr->b, float(ba.x));
                            atomicAdd(&grad_addr->a, float(ba.y));
                        }
                        if (enable_statistic)
                        {
                            //float err_sum{ float(err.x + err.y) * INV_SCALER };
                            //warp_reduce_sum<float, false>(err_sum);
                            float err_square_sum{ float(err_square.x + err_square.y) * INV_SCALER };
                            warp_reduce_sum<float, false>(err_square_sum, 0xffffffff);
                            if (threadIdx.x == 0)
                            {
                                atomicAdd(&out_err_square_sum[batch_id][0][point_id], err_square_sum);
                                //atomicAdd(&out_err_sum[batch_id][0][point_id], err_sum);
                            }
                        }

                        float3 grad_invcov{ 0,0,0 };
                        //basic = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                        //bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                        //neg_half_c = -0.5f * params.inv_cov11;
                        grad_invcov.x = -0.5f * d.x * d.x * grad_basic;
                        grad_invcov.y = (-d.x * d.y * grad_basic + d.x * grad_bxcy) * 0.5f;
                        grad_invcov.z = -0.5f * d.y * d.y * grad_basic + d.y * grad_bxcy - 0.5f * grad_neg_half_c;

                        warp_reduce_sum<float, false>(grad_invcov.x, 0xffffffff);
                        warp_reduce_sum<float, false>(grad_invcov.y, 0xffffffff);
                        warp_reduce_sum<float, false>(grad_invcov.z, 0xffffffff);
                        if (threadIdx.x == 0)
                        {
                            atomicAdd(&grad_addr->inv_cov00, grad_invcov.x);
                            atomicAdd(&grad_addr->inv_cov01, grad_invcov.y);
                            atomicAdd(&grad_addr->inv_cov11, grad_invcov.z);
                        }

                        float d_dx = (-params.inv_cov00 * d.x - params.inv_cov01 * d.y) * grad_basic + params.inv_cov01 * grad_bxcy;
                        float d_dy = (-params.inv_cov11 * d.y - params.inv_cov01 * d.x) * grad_basic + params.inv_cov11 * grad_bxcy;
                        float2 d_ndc_xy{ d_dx ,d_dy };
                        warp_reduce_sum<float2, false>(d_ndc_xy, 0xffffffff);
                        if (threadIdx.x == 0)
                        {
                            atomicAdd(&grad_addr->dx, d_ndc_xy.x);
                            atomicAdd(&grad_addr->dy, d_ndc_xy.y);
                        }
                    }
                }
            }
        }
    }
    else if (blockIdx.x < heavy_tiles.size(1))
    {
        const int warp_id = threadIdx.y;
        const int lane_id = threadIdx.x;
        const int tile_id = heavy_tiles[batch_id][blockIdx.x];
        const int subtile_id = (tile_id << 2) + warp_id;

        const int x_in_tile = warp_id * subtile_size_x + lane_id % subtile_size_x;
        const int y_in_tile = lane_id / subtile_size_x;

        const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + x_in_tile;
        const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + y_in_tile;

        float3 pixel_d_img{ d_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile],d_img[batch_id][1][tile_id - 1][y_in_tile][x_in_tile],d_img[batch_id][2][tile_id - 1][y_in_tile][x_in_tile] };
        float trans_grad_tmp = d_trans_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile] * final_transmitance[batch_id][0][tile_id - 1][y_in_tile][x_in_tile];
        if (tile_id != 0)
        {
            int start_index_in_tile = subtile_start[batch_id][subtile_id];

            float transmittance = final_transmitance[batch_id][0][tile_id - 1][y_in_tile][x_in_tile] * SCALER;
            float accum_depth = 0.0f;
            float3 accum_rec{ 0,0,0 };
            int lst_index = max(last_contributor[batch_id][tile_id - 1][y_in_tile][x_in_tile] - 1, 0);
            int warp_lst_index = __reduce_max_sync(0xffffffff, lst_index);

            for (int index_in_tile = warp_lst_index; index_in_tile >= 0; index_in_tile--)
            {
                int point_id = primitives_in_subtile[batch_id][start_index_in_tile + index_in_tile];
                PackedParams params = packed_params[point_id];
                PackedGrad* grad_addr = packed_grad + point_id;
                float3 cur_color{ params.rg.x,params.rg.y,params.ba.x };
                float cur_opacity = params.ba.y;
                float2 xy{ params.pixel_x,params.pixel_y };
                float2 d = { xy.x - pixel_x,xy.y - pixel_y };

                float power = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y) - params.inv_cov01 * d.x * d.y;
                float G = __expf(power);
                float alpha = min(255.0f / 256, cur_opacity * G);
                bool valid = (alpha >= 1.0f / 256) && (index_in_tile <= lst_index);
                alpha = valid * alpha;
                G = valid * G;

                float3 grad_color{ 0,0,0 };
                float3 grad_invcov{ 0,0,0 };
                float2 grad_ndc_xy{ 0,0 };
                float grad_ndc_z = 0;
                float grad_opacity{ 0 };
                if (__any_sync(0xffffffff, alpha != 0))
                {
                    transmittance = min(transmittance/(1 - alpha),1.0f);
                    //color
                    grad_color.x = alpha * transmittance * pixel_d_img.x;
                    grad_color.y = alpha * transmittance * pixel_d_img.y;
                    grad_color.z = alpha * transmittance * pixel_d_img.z;
                    warp_reduce_sum<float, false>(grad_color.x,0xffffffff);
                    warp_reduce_sum<float, false>(grad_color.y, 0xffffffff);
                    warp_reduce_sum<float, false>(grad_color.z, 0xffffffff);
                    if (lane_id == 0)
                    {
                        atomicAdd(&grad_addr->r, grad_color.x);
                        atomicAdd(&grad_addr->g, grad_color.y);
                        atomicAdd(&grad_addr->b, grad_color.z);
                    }
                    /*if (enable_depth_grad)
                    {
                        grad_ndc_z = alpha * transmittance * d_depth_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile];
                        warp_reduce_sum<float, false>(grad_ndc_z, 0xffffffff);
                        if (lane_id == 0)
                        {
                            atomicAdd(&grad_addr->ndc_z, grad_ndc_z);
                        }
                    }*/

                    //alpha
                    float d_alpha = 0;
                    d_alpha += (cur_color.x - accum_rec.x) * transmittance * pixel_d_img.x;
                    d_alpha += (cur_color.y - accum_rec.y) * transmittance * pixel_d_img.y;
                    d_alpha += (cur_color.z - accum_rec.z) * transmittance * pixel_d_img.z;
                    accum_rec.x += alpha * (cur_color.x - accum_rec.x);
                    accum_rec.y += alpha * (cur_color.y - accum_rec.y);
                    accum_rec.z += alpha * (cur_color.z - accum_rec.z);
                    if (enable_trans_grad)
                    {
                        d_alpha -= trans_grad_tmp / (1 - alpha);
                    }
                    /*if (enable_depth_grad)
                    {
                        d_alpha += (params.depth - accum_depth) * transmittance * d_depth_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile];
                        accum_depth = alpha * params.depth + (1.0f - alpha) * accum_depth;
                    }*/

                    //opacity
                    grad_opacity = G * d_alpha;
                    if (enable_statistic)
                    {
                        float cur_err = grad_opacity * INV_SCALER;
                        float err_square = cur_err * cur_err;
                        warp_reduce_sum<float, false>(err_square, 0xffffffff);
                        if (lane_id == 0)
                        {
                            atomicAdd(&out_err_square_sum[batch_id][0][point_id], err_square);
                        }
                    }

                    warp_reduce_sum<float, false>(grad_opacity, 0xffffffff);
                    if (lane_id == 0)
                    {
                        atomicAdd(&grad_addr->a, grad_opacity);
                    }

                    //cov2d_inv
                    float d_G = cur_opacity * d_alpha;
                    float d_power = G * d_G;
                    grad_invcov.x = -0.5f * d.x * d.x * d_power;
                    grad_invcov.y = -0.5f * d.x * d.y * d_power;
                    grad_invcov.z = -0.5f * d.y * d.y * d_power;
                    warp_reduce_sum<float, false>(grad_invcov.x, 0xffffffff);
                    warp_reduce_sum<float, false>(grad_invcov.y, 0xffffffff);
                    warp_reduce_sum<float, false>(grad_invcov.z, 0xffffffff);
                    if (lane_id == 0)
                    {
                        atomicAdd(&grad_addr->inv_cov00, grad_invcov.x);
                        atomicAdd(&grad_addr->inv_cov01, grad_invcov.y);
                        atomicAdd(&grad_addr->inv_cov11, grad_invcov.z);
                    }

                    //mean2d
                    float d_deltax = (-params.inv_cov00 * d.x - params.inv_cov01 * d.y) * d_power;
                    float d_deltay = (-params.inv_cov11 * d.y - params.inv_cov01 * d.x) * d_power;
                    warp_reduce_sum<float, false>(d_deltax, 0xffffffff);
                    warp_reduce_sum<float, false>(d_deltay, 0xffffffff);
                    if (lane_id == 0)
                    {
                        atomicAdd(&grad_addr->dx, d_deltax);
                        atomicAdd(&grad_addr->dy, d_deltay);
                    }

                }
            }
        }
    }
}

__global__ void unpack_gradient(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_grad,//[batch,point_num,property_num]
    const float* grad_inv_scaler,int img_h,int img_w,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity          //[1,point_num]
)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr float INV_SCALER = 1.0f / 128;//transmitance scale for fp16
    if (index < packed_grad.size(1))
    {
        float inv_scaleer = grad_inv_scaler[0] * INV_SCALER;
        PackedGrad* grads = (PackedGrad*)&packed_grad[blockIdx.y][index][0];
        d_ndc[blockIdx.y][0][index] = grads->dx * 0.5f * img_w * inv_scaleer;
        d_ndc[blockIdx.y][1][index] = grads->dy * 0.5f * img_h * inv_scaleer;
        d_cov2d_inv[blockIdx.y][0][0][index] = grads->inv_cov00 * inv_scaleer;
        d_cov2d_inv[blockIdx.y][0][1][index] = grads->inv_cov01 * inv_scaleer;
        d_cov2d_inv[blockIdx.y][1][0][index] = grads->inv_cov01 * inv_scaleer;
        d_cov2d_inv[blockIdx.y][1][1][index] = grads->inv_cov11 * inv_scaleer;
        d_color[blockIdx.y][0][index] = grads->r * inv_scaleer;
        d_color[blockIdx.y][1][index] = grads->g * inv_scaleer;
        d_color[blockIdx.y][2][index] = grads->b * inv_scaleer;
        if (blockIdx.y == 0)//todo fix
        {
            d_opacity[0][index] = grads->a * inv_scaleer;
        }
    }
}


#define RASTER_BACKWARD_PARAMS primitives_in_tile.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
tile_start.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
primitives_in_subtile.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
subtile_start.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
heavy_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
reinterpret_cast<PackedParams*>(packed_params.data_ptr<float>()),\
final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),\
last_contributor.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),\
d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
reinterpret_cast<PackedGrad*>(packed_grad.data_ptr<float>()),\
err_sum.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),\
err_square_sum.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),\
tilesnum_x, img_h, img_w, packed_params.size(1)

#define LAUNCH_RASTER_BACKWARD_KERNEL(TILE_H, TILE_W, STATISTIC, TRANS, DEPTH) \
    raster_backward_kernel<TILE_H, TILE_W, 8, 4, STATISTIC, TRANS, DEPTH> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);

#define DISPATCH_RASTER_BACKWARD_KERNEL(STATISTIC, TRANS, DEPTH) \
    if (tile_h == 8 && tile_w == 16) { \
        LAUNCH_RASTER_BACKWARD_KERNEL(8, 16, STATISTIC, TRANS, DEPTH); } \
    else if (tile_h == 12 && tile_w == 16) { \
        LAUNCH_RASTER_BACKWARD_KERNEL(12, 16, STATISTIC, TRANS, DEPTH); } \
    else if (tile_h == 16 && tile_w == 16) { \
        LAUNCH_RASTER_BACKWARD_KERNEL(16, 16, STATISTIC, TRANS, DEPTH); } \
    else if (tile_h == 8 && tile_w == 8) { \
        LAUNCH_RASTER_BACKWARD_KERNEL(8, 8, STATISTIC, TRANS, DEPTH); }


std::vector<at::Tensor> rasterize_backward(
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, std::optional<at::Tensor>  specific_tiles_arg,
    std::optional<at::Tensor> primitives_in_subtile_arg, std::optional<at::Tensor> subtile_start_arg, std::optional<at::Tensor> subtile_end_arg, std::optional<at::Tensor>  heavy_tiles_arg,
    at::Tensor packed_params,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    std::optional<at::Tensor> d_trans_img_arg,
    std::optional<at::Tensor> d_depth_img_arg,
    std::optional<at::Tensor> grad_inv_sacler_arg,
    int64_t img_h,
    int64_t img_w,
    int64_t tile_h,
    int64_t tile_w,
    bool enable_statistic
)
{
    at::DeviceGuard guard(packed_params.device());

    int64_t viewsnum = tile_start.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    int64_t render_tile_num = tilesnum;
    int64_t heavy_tile_num = 0;
    at::Tensor specific_tiles;
    at::Tensor heavy_tiles;
    at::Tensor primitives_in_subtile;
    at::Tensor subtile_start;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        render_tile_num = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
    }
    if (heavy_tiles_arg.has_value())
    {
        heavy_tiles = *heavy_tiles_arg;
        heavy_tile_num = heavy_tiles.size(1);
        primitives_in_subtile = *primitives_in_subtile_arg;
        subtile_start = *subtile_start_arg;
    }
    else
    {
        heavy_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
        primitives_in_subtile = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
        subtile_start = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
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
    at::Tensor err_square_sum = torch::zeros({ batch_num,1,points_num }, packed_params.options());
    at::Tensor err_sum = torch::zeros({ batch_num,1,points_num }, packed_params.options());
    
    int tiles_per_block = 4;
    dim3 Block3d(std::ceil(render_tile_num / float(tiles_per_block)), viewsnum, 1);
    dim3 Thread3d(32, tiles_per_block);
    
    switch (ENCODE(enable_statistic, d_trans_img_arg.has_value(), d_depth_img_arg.has_value()))
    {
    case ENCODE(false, false, false):
        DISPATCH_RASTER_BACKWARD_KERNEL(false, false, false)
        break;
    case ENCODE(true, false, false):
        DISPATCH_RASTER_BACKWARD_KERNEL(true, false, false)
        break;
    case ENCODE(false, true, false):
        DISPATCH_RASTER_BACKWARD_KERNEL(false, true, false)
        break;
    case ENCODE(false, false, true):
        DISPATCH_RASTER_BACKWARD_KERNEL(false, false, true)
        break;
    case ENCODE(true, true, false):
        DISPATCH_RASTER_BACKWARD_KERNEL(true, true, false)
        break;
    case ENCODE(true, false, true):
        DISPATCH_RASTER_BACKWARD_KERNEL(true, false, true)
        break;
    case ENCODE(false, true, true):
        DISPATCH_RASTER_BACKWARD_KERNEL(false, true, true)
        break;
    case ENCODE(true, true, true):
        DISPATCH_RASTER_BACKWARD_KERNEL(true, true, true)
        break;
    default:
        break;
    }

    CUDA_CHECK_ERRORS;

    dim3 UnpackBlock3d(std::ceil(points_num / 512.0f), batch_num, 1);
    unpack_gradient<<<UnpackBlock3d,512>>>(
        packed_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        (float*)grad_inv_sacler.data_ptr(),img_h,img_w,
        d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),
        d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),
        d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >());
    CUDA_CHECK_ERRORS;

    return { d_ndc ,d_cov2d_inv ,d_color,d_opacity,err_sum,err_square_sum };
}
