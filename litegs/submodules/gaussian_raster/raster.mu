#ifndef __MUSACC__
#define __MUSACC__
#define __NVCC__
#endif
#include "musa_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <musa/atomic>
#include <math.h>
#include <musa_fp16.h>
namespace cg = cooperative_groups;

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


struct RGBA32
{
    float r;
    float g;
    float b;
    float a;
};


struct RegisterBuffer
{
    float r;
    float g;
    float b;
    float t;
    int lst_contributor;
    float alpha;
};


template<class T, bool boardcast>
inline __device__ void warp_reduce_sum(T& data, unsigned int mask=0xffffffff)
{
    data += __shfl_down_sync(mask, data, 16);
    data += __shfl_down_sync(mask, data, 8);
    data += __shfl_down_sync(mask, data, 4);
    data += __shfl_down_sync(mask, data, 2);
    data += __shfl_down_sync(mask, data, 1);
    if (boardcast)
        data = __shfl_sync(mask, data, 0);
}



template <int tile_size_y, int tile_size_x, int subtile_size_y, bool enable_statistic, bool enable_trans, bool enable_depth>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_tile,    //[batch,items] 
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tile_start,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tile_end,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_subtile,    //[batch,items] 
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> subtile_start,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> subtile_end,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> complex_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float/*torch::Half*/, 3, torch::RestrictPtrTraits> packed_params,         //[batch,point_num,sizeof(PackedParams)/4]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_depth,     //[batch,1,tile,tilesize, tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> out_fragment_count,  //[batch,1,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_fragment_weight_sum,  //[batch,1,point_num]
    int tiles_num_x)
{
    //assert blockDim.x==32

    constexpr int VECTOR_SIZE = 1;
    constexpr int PIXELS_PER_THREAD = (tile_size_x * tile_size_y) / (32* VECTOR_SIZE);

    const int batch_id = blockIdx.y;
    int task_id = blockIdx.x*blockDim.y+threadIdx.y;
    int render_subtiles_num = complex_tiles.size(1)*4;

    if (render_subtiles_num<=task_id)
    {
        task_id-=render_subtiles_num;


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

        if (tile_id != 0 && tile_id < tile_start.size(1) )
        {

            int start_index_in_tile = tile_start[batch_id][tile_id];
            int end_index_in_tile = tile_end[batch_id][tile_id];
            RegisterBuffer reg_buffer[PIXELS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD; i++)
            {
                reg_buffer[i].r = 0;
                reg_buffer[i].g = 0;
                reg_buffer[i].b = 0;
                reg_buffer[i].t = 1.0f;
                reg_buffer[i].lst_contributor = 0;
            }



            unsigned int any_active = 0xffffffffu;
            auto points_id_in_tile = &primitives_in_tile[batch_id][start_index_in_tile];
            for (int index_in_tile = 0; (index_in_tile + start_index_in_tile < end_index_in_tile) && (any_active != 0); index_in_tile++)
            {
                int point_id = points_id_in_tile[index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                RGBA32 point_color;
                point_color.r = params.rg.x;
                point_color.g = params.rg.y;
                point_color.b = params.ba.x;
                point_color.a = params.ba.y;
                float2 xy{ params.pixel_x,params.pixel_y };

                const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + threadIdx.x % tile_size_x ;
                const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;
                float2 d { xy.x - pixel_x,xy.y - pixel_y };
                float basic = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                float bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                float neg_half_c = -0.5f * params.inv_cov11;
                //basic+=(cy+bx)*delta - 0.5*c*delta*delta

                any_active = 0;
                unsigned int fragment_count = 0x0;//ushort2
                float weight_sum = 0;
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD; i++)
                {
                    float power = basic + i * bxcy + i * i * neg_half_c;

                    bool bActive = true;
                    bActive = reg_buffer[i].t > 1.0f / 8192;
                    reg_buffer[i].lst_contributor += bActive;
                    any_active |= bActive;

                    reg_buffer[i].alpha = point_color.a * __expf(power);
                    bool alpha_valid = reg_buffer[i].alpha > 1.0f / 256;
                    reg_buffer[i].alpha = (bActive&alpha_valid) * std::min(255.0f / 256, reg_buffer[i].alpha);

                    float weight = reg_buffer[i].t * reg_buffer[i].alpha;
                    if (enable_statistic)
                    {
                        fragment_count += bActive;
                        weight_sum += weight;
                    }

                    reg_buffer[i].r += (point_color.r * weight);
                    reg_buffer[i].g += (point_color.g * weight);
                    reg_buffer[i].b += (point_color.b * weight);
                    reg_buffer[i].t = reg_buffer[i].t * (1.0f - reg_buffer[i].alpha);
                }
                //reg_buffer[1].alpha = (half2(2.0f, 2.0f) * reg_buffer[0].alpha + reg_buffer[3].alpha) * half2(1.0f / 3, 1.0f / 3);
                //reg_buffer[2].alpha = (reg_buffer[0].alpha + half2(2.0f, 2.0f) * reg_buffer[3].alpha) * half2(1.0f / 3, 1.0f / 3);


                //reduce statistic
                if (enable_statistic)
                {
                    warp_reduce_sum<unsigned int, false>(fragment_count);
                    warp_reduce_sum<float, false>(weight_sum);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&out_fragment_count[batch_id][0][point_id], fragment_count);
                        atomicAdd(&out_fragment_weight_sum[batch_id][0][point_id], weight_sum);
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
                const int output_y = threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE + i;

                ourput_r[output_y][output_x] = std::min(reg_buffer[i].r, 1.0f);
                ourput_g[output_y][output_x] = std::min(reg_buffer[i].g, 1.0f);
                ourput_b[output_y][output_x] = std::min(reg_buffer[i].b, 1.0f);
                ourput_t[output_y][output_x] = reg_buffer[i].t;
                output_last_index[output_y][output_x] = reg_buffer[i].lst_contributor;
            }
        }
    }
    else if(task_id<render_subtiles_num)
    {
        const int warp_id = threadIdx.y;
        const int lane_id = threadIdx.x;
        const int tile_id = complex_tiles[batch_id][blockIdx.x];
        const int subtile_id = (tile_id << 2) + warp_id;


        const int x_in_tile = lane_id % tile_size_x;
        const int y_in_tile = warp_id * subtile_size_y + lane_id / tile_size_x;

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

            for (int index_in_tile = 0; (index_in_tile + start_index_in_tile < end_index_in_tile)&&(transmittance > 1.0f / 8192); index_in_tile++)
            {
                int point_id = primitives_in_subtile[batch_id][start_index_in_tile+index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                float3 cur_color{ params.rg.x,params.rg.y,params.ba.x };
                float cur_opacity = params.ba.y;
                float2 xy{ params.pixel_x,params.pixel_y };
                float2 d = { xy.x - pixel_x,xy.y - pixel_y };

                float power = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y) - params.inv_cov01 * d.x * d.y;
                float alpha = min(255.0f / 256, cur_opacity * __expf(power));
                alpha = (alpha < 1.0f / 256) ? 0 : alpha;

                float weight= alpha * transmittance;
                if (enable_statistic)
                {
                    unsigned int active_mask = __activemask();
                    int fragment_count = __popc(active_mask);
                    warp_reduce_sum<float, false>(weight, active_mask);
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
                last_contributor = index_in_tile;
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
complex_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),\
fragment_count.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),\
fragment_weight_sum.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
tilesnum_x

#define ENCODE(STATISTIC, TRANS, DEPTH) (((STATISTIC)*1)<<2)|(((TRANS)*1)<<1)|((DEPTH)*1)

std::vector<at::Tensor> rasterize_forward(
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, std::optional<at::Tensor>  specific_tiles_arg,
    std::optional<at::Tensor> primitives_in_subtile_arg, std::optional<at::Tensor> subtile_start_arg, std::optional<at::Tensor> subtile_end_arg, std::optional<at::Tensor>  complex_tiles_arg,
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
    assert(tile_h == 16 && tile_w == 8);

    int64_t viewsnum = tile_start.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    int64_t render_tile_num = tilesnum;
    int64_t complex_tile_num = 0;
    at::Tensor specific_tiles;
    at::Tensor complex_tiles;
    at::Tensor primitives_in_subtile;
    at::Tensor subtile_start;
    at::Tensor subtile_end;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        render_tile_num = specific_tiles.size(1);
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
    }
    if (complex_tiles_arg.has_value())
    {
        complex_tiles=*complex_tiles_arg;
        complex_tile_num=complex_tiles.size(1);
        primitives_in_subtile = *primitives_in_subtile_arg;
        subtile_start = *subtile_start_arg;
        subtile_end = *subtile_end_arg;
    }
    else
    {
        complex_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
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

    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(tile_start.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty({ viewsnum, tilesnum, tile_h, tile_w }, opt_c);

    at::Tensor fragment_count = torch::zeros({ viewsnum,1,points_num }, packed_params.options().dtype(torch::kI32));
    at::Tensor fragment_weight_sum = torch::zeros({ viewsnum,1,points_num }, packed_params.options().dtype(torch::kFloat32));

    {
        int tiles_per_block = 4;
        dim3 Block3d(complex_tile_num+std::ceil(render_tile_num / float(tiles_per_block)), viewsnum, 1);
        dim3 Thread3d(32, tiles_per_block);
        switch (ENCODE(enable_statistic, enable_trans, enable_depth))
        {
        case ENCODE(false,false,false):
            raster_forward_kernel<16, 8, 4, false, false, false> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(true, false, false):
            raster_forward_kernel<16, 8, 4, true, false, false> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(false, true, false):
            raster_forward_kernel<16, 8, 4, false, true, false> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(false, false, true):
            raster_forward_kernel<16, 8, 4, false, false, true> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(true, true, false):
            raster_forward_kernel<16, 8, 4, true, true, false> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(true, false, true):
            raster_forward_kernel<16, 8, 4, true, false, true> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(false, true, true):
            raster_forward_kernel<16, 8, 4, false, true, true> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(true, true, true):
            raster_forward_kernel<16, 8, 4, true, true, true> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
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
    std::optional<at::Tensor> primitives_in_subtile_arg, std::optional<at::Tensor> subtile_start_arg, std::optional<at::Tensor> subtile_end_arg, std::optional<at::Tensor>  complex_tiles_arg,
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
    assert(tile_h == 16 && tile_w == 8);

    int64_t viewsnum = tile_start.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    int64_t render_tile_num = tilesnum;
    int64_t complex_tile_num = 0;
    at::Tensor specific_tiles;
    at::Tensor complex_tiles;
    at::Tensor primitives_in_subtile;
    at::Tensor subtile_start;
    at::Tensor subtile_end;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        render_tile_num = specific_tiles.size(1);
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
    }
    if (complex_tiles_arg.has_value())
    {
        complex_tiles=*complex_tiles_arg;
        complex_tile_num=complex_tiles.size(1);
        primitives_in_subtile = *primitives_in_subtile_arg;
        subtile_start = *subtile_start_arg;
        subtile_end = *subtile_end_arg;
    }
    else
    {
        complex_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
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

    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(tile_start.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty({ viewsnum, tilesnum, tile_h, tile_w }, opt_c);

    int points_num = packed_params.size(1);
    at::Tensor fragment_count = torch::zeros({ viewsnum,1,points_num }, packed_params.options().dtype(torch::kI32));
    at::Tensor fragment_weight_sum = torch::zeros({ viewsnum,1,points_num }, packed_params.options().dtype(torch::kFloat32));

    {
        int tiles_per_block = 4;
        dim3 Block3d(complex_tile_num+std::ceil(render_tile_num / float(tiles_per_block)), viewsnum, 1);
        dim3 Thread3d(32, tiles_per_block);
        switch (ENCODE(enable_statistic, enable_trans, enable_depth))
        {
        case ENCODE(false, false, false):
            raster_forward_kernel<16, 8, 4, false, false, false> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(true, false, false):
            raster_forward_kernel<16, 8, 4, true, false, false> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(false, true, false):
            raster_forward_kernel<16, 8, 4, false, true, false> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(false, false, true):
            raster_forward_kernel<16, 8, 4, false, false, true> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(true, true, false):
            raster_forward_kernel<16, 8, 4, true, true, false> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(true, false, true):
            raster_forward_kernel<16, 8, 4, true, false, true> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(false, true, true):
            raster_forward_kernel<16, 8, 4, false, true, true> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
            break;
        case ENCODE(true, true, true):
            raster_forward_kernel<16, 8, 4, true, true, true> <<<Block3d, Thread3d >>> (RASTER_FORWARD_PARAMS);
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
    float r;
    float g;
    float b;
    float t;
    float alpha;
};


template <int tile_size_y, int tile_size_x, int subtile_size_y,bool enable_statistic, bool enable_trans_grad, bool enable_depth_grad>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_tile,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> tile_start,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_subtile,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> subtile_start,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> complex_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_params,         // //[batch,point_num,sizeof(PackedParams)/sizeof(float)]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> packed_grad,         //[batch,point_num,9]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_err_sum,  //[batch,1,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_err_square_sum,  //[batch,1,point_num]
    int tiles_num_x, int img_h, int img_w)
{
    constexpr int VECTOR_SIZE = 1;
    constexpr int PIXELS_PER_THREAD = (tile_size_x * tile_size_y) / (32 * VECTOR_SIZE);

    __shared__ float shared_img_grad[3][PIXELS_PER_THREAD][4 * 32];
    __shared__ float shared_trans_grad_buffer[PIXELS_PER_THREAD][4 * 32];
    __shared__ int shared_last_contributor[PIXELS_PER_THREAD][4 * 32];

    const int batch_id = blockIdx.y;
    int task_id = blockIdx.x*blockDim.y+threadIdx.y;
    int render_subtiles_num = complex_tiles.size(1)*4;

    if (render_subtiles_num<=task_id)
    {
        task_id-=render_subtiles_num;
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
        if (tile_id != 0 && tile_id < tile_start.size(1))
        {

            int start_index_in_tile = tile_start[batch_id][tile_id];
            int index_in_tile = 0;
            BackwardRegisterBuffer reg_buffer[PIXELS_PER_THREAD];
            //int lst[pixels_per_thread];
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD; i++)
            {
                reg_buffer[i].r = 0;
                reg_buffer[i].g = 0;
                reg_buffer[i].b = 0;

                const int in_tile_x = threadIdx.x % tile_size_x;
                const int in_tile_y = threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;
                reg_buffer[i].t = final_transmitance[batch_id][0][tile_id - 1][in_tile_y + i][in_tile_x];
                if (enable_trans_grad)
                {
                    shared_trans_grad_buffer[i][threadIdx.y * blockDim.x + threadIdx.x] = final_transmitance[batch_id][0][tile_id - 1][in_tile_y + i][in_tile_x] * d_trans_img[batch_id][0][tile_id - 1][in_tile_y + i][in_tile_x];
                }

                shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x] = 
                    d_img[batch_id][0][tile_id - 1][in_tile_y + i][in_tile_x];
                shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x] =
                    d_img[batch_id][1][tile_id - 1][in_tile_y + i][in_tile_x];
                shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x] =
                    d_img[batch_id][2][tile_id - 1][in_tile_y + i][in_tile_x];
                int last = last_contributor[batch_id][tile_id - 1][in_tile_y + i][in_tile_x] - 1;
                shared_last_contributor[i][threadIdx.y * blockDim.x + threadIdx.x] = last;
                index_in_tile = std::max(last, index_in_tile);
            }
            index_in_tile = __reduce_max_sync(0xffffffff, index_in_tile);

            const int* points_in_tile = &primitives_in_tile[batch_id][start_index_in_tile];
            const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + threadIdx.x % tile_size_x;
            const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;

            for (; (index_in_tile >= 0); index_in_tile--)
            {
                float basic;
                float bxcy;
                float neg_c;
                float2 d{ 0,0 };
                int point_id = points_in_tile[index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                {
                    float2 xy{ params.pixel_x,params.pixel_y};
                    d.x = xy.x - pixel_x;
                    d.y = xy.y - pixel_y;
                    basic = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                    bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                    neg_c = -0.5f * params.inv_cov11;
                }//basic+=(cy+bx)*delta - 0.5*c*delta*delta

                RGBA32 point_color;
                point_color.r = params.rg.x;
                point_color.g = params.rg.y;
                point_color.b = params.ba.x;
                point_color.a = params.ba.y;
                

                float grad_r = 0;
                float grad_g = 0;
                float grad_b = 0;
                float grad_a = 0;
                float err_square = 0;
                float grad_bxcy = 0;
                float grad_neg_c = 0;
                float grad_basic = 0;
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD; i++)
                {
                    float power= basic + i * bxcy + i * i * neg_c;
                    float G = __expf(power);
                    float alpha = point_color.a * G;
                    alpha = std::min(255.0f / 256, alpha);

                    bool valid = true;
                    valid &= alpha >= 1.0f / 256;
                    valid &= index_in_tile <= shared_last_contributor[i][threadIdx.y * blockDim.x + threadIdx.x];

                    if (__any_sync(0xffffffff, valid))
                    {
                        alpha *= valid;
                        G *= valid;

                        reg_buffer[i].t = reg_buffer[i].t/(1.0f - alpha);
                        grad_r += alpha * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                        grad_g += alpha * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                        grad_b += alpha * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];


                        float d_alpha = 0;
                        d_alpha += (point_color.r - reg_buffer[i].r) * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                        d_alpha += (point_color.g - reg_buffer[i].g) * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                        d_alpha += (point_color.b - reg_buffer[i].b) * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];
                        reg_buffer[i].r += alpha * (point_color.r - reg_buffer[i].r);//0-256
                        reg_buffer[i].g += alpha * (point_color.g - reg_buffer[i].g);
                        reg_buffer[i].b += alpha * (point_color.b - reg_buffer[i].b);
                        if (enable_trans_grad)
                        {
                            d_alpha -= shared_trans_grad_buffer[i][threadIdx.y * blockDim.x + threadIdx.x] / (1.0f - alpha);
                        }

                        grad_a += d_alpha * G;
                        float d_G = point_color.a * d_alpha;
                        float d_power = G * d_G;//G * point_a * d_alpha => alpha * d_alpha
                        if (enable_statistic)
                        {
                            float cur_err = grad_a;
                            //err += cur_err;
                            err_square += cur_err * cur_err;
                        }
                        grad_bxcy += d_power * i;
                        grad_neg_c += d_power * i * i;
                        grad_basic += d_power;
                    }
                }
                
                PackedGrad* grad_addr = (PackedGrad*)&packed_grad[batch_id][point_id][0];
                //unsigned mask = __ballot_sync(0xffffffff, grad_opacity!=0);
                if (__any_sync(0xffffffff, grad_a!=0))
                {
                    warp_reduce_sum<float, false>(grad_r);
                    warp_reduce_sum<float, false>(grad_g);
                    warp_reduce_sum<float, false>(grad_b);
                    warp_reduce_sum<float, false>(grad_a);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&grad_addr->r, grad_r);
                        atomicAdd(&grad_addr->g, grad_g);
                        atomicAdd(&grad_addr->b, grad_b);
                        atomicAdd(&grad_addr->a, grad_a);
                    }
                    if (enable_statistic)
                    {
                        //warp_reduce_sum<float, false>(err);
                        warp_reduce_sum<float, false>(err_square);
                        if (threadIdx.x == 0)
                        {
                            atomicAdd(&out_err_square_sum[batch_id][0][point_id], err_square);
                            //atomicAdd(&out_err_sum[batch_id][0][point_id], err);
                        }
                    }

                    float3 grad_invcov{ 0,0,0 };
                    //basic = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y + 2 * params.inv_cov01 * d.x * d.y);
                    //bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                    //neg_half_c = -0.5f * params.inv_cov11;
                    grad_invcov.x = -0.5f * d.x * d.x * grad_basic;
                    grad_invcov.y = (-d.x * d.y * grad_basic + d.x * grad_bxcy) * 0.5f;
                    grad_invcov.z = -0.5f * d.y * d.y * grad_basic + d.y * grad_bxcy - 0.5f * grad_neg_c;

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
                    warp_reduce_sum<float, false>(d_ndc_xy.x);
                    warp_reduce_sum<float, false>(d_ndc_xy.y);
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&grad_addr->ndc_x, d_ndc_xy.x);
                        atomicAdd(&grad_addr->ndc_y, d_ndc_xy.y);
                    }
                }
            }
        }

    }
    else if(task_id<render_subtiles_num)
    {
        const int warp_id = threadIdx.y;
        const int lane_id = threadIdx.x;
        const int tile_id = complex_tiles[batch_id][blockIdx.x];
        const int subtile_id = (tile_id << 2) + warp_id;


        const int x_in_tile = lane_id % tile_size_x;
        const int y_in_tile = warp_id * subtile_size_y + lane_id / tile_size_x;

        const int pixel_x = ((tile_id - 1) % tiles_num_x) * tile_size_x + x_in_tile;
        const int pixel_y = ((tile_id - 1) / tiles_num_x) * tile_size_y + y_in_tile;

        float3 pixel_d_img{d_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile],d_img[batch_id][1][tile_id - 1][y_in_tile][x_in_tile],d_img[batch_id][2][tile_id - 1][y_in_tile][x_in_tile]};
        float trans_grad_tmp=d_trans_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile] * final_transmitance[batch_id][0][tile_id - 1][y_in_tile][x_in_tile];
        if (tile_id != 0)
        {
            int start_index_in_tile = subtile_start[batch_id][subtile_id];

            float transmittance = final_transmitance[batch_id][0][tile_id - 1][y_in_tile][x_in_tile];
            float accum_depth = 0.0f;
            float3 accum_rec{ 0,0,0 };
            int lst_index = max(last_contributor[batch_id][tile_id - 1][y_in_tile][x_in_tile] - 1, 0);
            int warp_lst_index=__reduce_max_sync(0xffffffff, lst_index);

            for (int index_in_tile = warp_lst_index; index_in_tile >= 0; index_in_tile--)
            {
                int point_id = primitives_in_subtile[batch_id][start_index_in_tile + index_in_tile];
                PackedParams params = *((PackedParams*)&packed_params[batch_id][point_id][0]);
                PackedGrad* grad_addr = (PackedGrad*)&packed_grad[batch_id][point_id][0];
                float3 cur_color{ params.rg.x,params.rg.y,params.ba.x };
                float cur_opacity = params.ba.y;
                float2 xy{ params.pixel_x,params.pixel_y };
                float2 d = { xy.x - pixel_x,xy.y - pixel_y };

                float power = -0.5f * (params.inv_cov00 * d.x * d.x + params.inv_cov11 * d.y * d.y) - params.inv_cov01 * d.x * d.y;
                float G = __expf(power);
                float alpha = min(255.0f / 256, cur_opacity * G);
                alpha = (alpha < 1.0f / 256) ? 0 : alpha;
                alpha = (index_in_tile > lst_index) ? 0 : alpha;

                float3 grad_color{ 0,0,0 };
                float3 grad_invcov{ 0,0,0 };
                float2 grad_ndc_xy{ 0,0 };
                float grad_ndc_z = 0;
                float grad_opacity{ 0 };
                if (__any_sync(0xffffffff, alpha != 0))
                {
                    transmittance /= (1 - alpha);
                    //color
                    grad_color.x = alpha * transmittance * pixel_d_img.x;
                    grad_color.y = alpha * transmittance * pixel_d_img.y;
                    grad_color.z = alpha * transmittance * pixel_d_img.z;
                    warp_reduce_sum<float, false>(grad_color.x);
                    warp_reduce_sum<float, false>(grad_color.y);
                    warp_reduce_sum<float, false>(grad_color.z);
                    if (lane_id == 0)
                    {
                        atomicAdd(&grad_addr->r, grad_color.x);
                        atomicAdd(&grad_addr->g, grad_color.y);
                        atomicAdd(&grad_addr->b, grad_color.z);
                    }
                    if (enable_depth_grad)
                    {
                        grad_ndc_z = alpha * transmittance * d_depth_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile];
                        warp_reduce_sum<float, false>(grad_ndc_z);
                        if (lane_id == 0)
                        {
                            //TODO Z
                        }
                    }

                    //alpha
                    float d_alpha = 0;
                    d_alpha += (cur_color.x - accum_rec.x) * transmittance * pixel_d_img.x;
                    d_alpha += (cur_color.y - accum_rec.y) * transmittance * pixel_d_img.y;
                    d_alpha += (cur_color.z - accum_rec.z) * transmittance * pixel_d_img.z;
                    accum_rec.x = alpha * cur_color.x + (1.0f - alpha) * accum_rec.x;
                    accum_rec.y = alpha * cur_color.y + (1.0f - alpha) * accum_rec.y;
                    accum_rec.z = alpha * cur_color.z + (1.0f - alpha) * accum_rec.z;
                    if (enable_trans_grad)
                    {
                        d_alpha -= trans_grad_tmp / (1 - alpha);
                    }
                    if (enable_depth_grad)
                    {
                        d_alpha += (params.depth - accum_depth) * transmittance * d_depth_img[batch_id][0][tile_id - 1][y_in_tile][x_in_tile];
                        accum_depth = alpha * params.depth + (1.0f - alpha) * accum_depth;
                    }

                    //opacity
                    grad_opacity = G * d_alpha;
                    warp_reduce_sum<float, false>(grad_opacity);
                    if (lane_id == 0)
                    {
                        atomicAdd(&grad_addr->a, grad_opacity);
                    }
                    if (enable_statistic)
                    {
                        float cur_err = grad_opacity;
                        float err_square = cur_err * cur_err;
                        warp_reduce_sum<float, false>(err_square);
                        if (lane_id == 0)
                        {
                            atomicAdd(&out_err_square_sum[batch_id][0][point_id], err_square);
                        }
                    }

                    //cov2d_inv
                    float d_G = cur_opacity * d_alpha;
                    float d_power = G * d_G;
                    grad_invcov.x = -0.5f * d.x * d.x * d_power;
                    grad_invcov.y = -0.5f * d.x * d.y * d_power;
                    grad_invcov.z = -0.5f * d.y * d.y * d_power;
                    warp_reduce_sum<float, false>(grad_invcov.x);
                    warp_reduce_sum<float, false>(grad_invcov.y);
                    warp_reduce_sum<float, false>(grad_invcov.z);
                    if (lane_id == 0)
                    {
                        atomicAdd(&grad_addr->inv_cov00, grad_invcov.x);
                        atomicAdd(&grad_addr->inv_cov01, grad_invcov.y);
                        atomicAdd(&grad_addr->inv_cov11, grad_invcov.z);
                    }

                    //mean2d
                    float d_deltax = (-params.inv_cov00 * d.x - params.inv_cov01 * d.y) * d_power;
                    float d_deltay = (-params.inv_cov11 * d.y - params.inv_cov01 * d.x) * d_power;
                    grad_ndc_xy.x = d_deltax;
                    grad_ndc_xy.y = d_deltay;
                    warp_reduce_sum<float, false>(grad_ndc_xy.x);
                    warp_reduce_sum<float, false>(grad_ndc_xy.y);
                    if (lane_id == 0)
                    {
                        atomicAdd(&grad_addr->ndc_x, grad_ndc_xy.x);
                        atomicAdd(&grad_addr->ndc_y, grad_ndc_xy.y);
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


#define RASTER_BACKWARD_PARAMS primitives_in_tile.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
tile_start.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
primitives_in_subtile.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
subtile_start.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
complex_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
packed_params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),\
last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),\
d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
packed_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
err_sum.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),\
err_square_sum.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),\
tilesnum_x, img_h, img_w

std::vector<at::Tensor> rasterize_backward(
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, std::optional<at::Tensor>  specific_tiles_arg,
    std::optional<at::Tensor> primitives_in_subtile_arg, std::optional<at::Tensor> subtile_start_arg, std::optional<at::Tensor> subtile_end_arg, std::optional<at::Tensor>  complex_tiles_arg,
    at::Tensor packed_params,
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
    assert(tilesize_h == 16 && tilesize_w == 8);

    int64_t viewsnum = tile_start.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tilesize_w));
    int tilesnum_y = std::ceil(img_h / float(tilesize_h));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    int64_t render_tile_num = tilesnum;
    int64_t complex_tile_num = 0;
    at::Tensor specific_tiles;
    at::Tensor complex_tiles;
    at::Tensor primitives_in_subtile;
    at::Tensor subtile_start;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        render_tile_num = specific_tiles.size(1);
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
    }
    if (complex_tiles_arg.has_value())
    {
        complex_tiles=*complex_tiles_arg;
        complex_tile_num=complex_tiles.size(1);
        primitives_in_subtile = *primitives_in_subtile_arg;
        subtile_start = *subtile_start_arg;
    }
    else
    {
        complex_tiles = torch::empty({ 0,0 }, packed_params.options().dtype(torch::kInt32));
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
    dim3 Block3d( complex_tile_num+std::ceil(render_tile_num/4.0f), viewsnum, 1);
    dim3 Thread3d(32, tiles_per_block);
    
    switch (ENCODE(enable_statistic, d_trans_img_arg.has_value(), d_depth_img_arg.has_value()))
    {
    case ENCODE(false, false, false):
        raster_backward_kernel<16, 8, 4, false, false, false> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);
        break;
    case ENCODE(true, false, false):
        raster_backward_kernel<16, 8, 4, true, false, false> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);
        break;
    case ENCODE(false, true, false):
        raster_backward_kernel<16, 8, 4, false, true, false> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);
        break;
    case ENCODE(false, false, true):
        raster_backward_kernel<16, 8, 4, false, false, true> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);
        break;
    case ENCODE(true, true, false):
        raster_backward_kernel<16, 8, 4, true, true, false> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);
        break;
    case ENCODE(true, false, true):
        raster_backward_kernel<16, 8, 4, true, false, true> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);
        break;
    case ENCODE(false, true, true):
        raster_backward_kernel<16, 8, 4, false, true, true> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);
        break;
    case ENCODE(true, true, true):
        raster_backward_kernel<16, 8, 4, true, true, true> <<<Block3d, Thread3d >>> (RASTER_BACKWARD_PARAMS);
        break;
    default:
        break;
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

    return { d_ndc ,d_cov2d_inv ,d_color,d_opacity,err_sum,err_square_sum };
}
