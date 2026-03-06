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
    float ddepth;
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
    float d;
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



template <int tile_size_y, int tile_size_x,int subtile_size_y,int subtile_size_x, bool enable_statistic, bool enable_trans, bool enable_depth>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_tile,    //[batch,items] 
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> end_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> tile_pixel_index,    //[batch,tile,2]
    const PackedParams* __restrict__ packed_params,         //[batch,point_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_depth,     //[batch,1,tile,tilesize, tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> out_fragment_count,  //[batch,1,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_fragment_weight_sum,  //[batch,1,point_num]
    int tiles_num_x, int pointsnum)
{
    //assert blockDim.x==32

    constexpr int VECTOR_SIZE = 1;
    constexpr int PIXELS_PER_THREAD = (tile_size_x * tile_size_y) / (32* VECTOR_SIZE);

    const int batch_id = blockIdx.y;
    packed_params += batch_id * pointsnum;

    int tile_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (tile_id < start_index.size(1))
    {

        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = end_index[batch_id][tile_id];
        RegisterBuffer reg_buffer[PIXELS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < PIXELS_PER_THREAD; i++)
        {
            reg_buffer[i].r = 0;
            reg_buffer[i].g = 0;
            reg_buffer[i].b = 0;
            if(enable_depth)
            {
                reg_buffer[i].d=0;
            }
            reg_buffer[i].t = 1.0f;
            reg_buffer[i].lst_contributor = 0;
        }


        if (start_index_in_tile<end_index_in_tile)
        {

            unsigned int any_active = 0xffffffffu;
            int index_in_tile = 0;
            auto points_id_in_tile = &primitives_in_tile[batch_id][start_index_in_tile];
            for (; (index_in_tile + start_index_in_tile < end_index_in_tile) && __any_sync(0xffffffff,any_active != 0); index_in_tile++)
            {
                int point_id = points_id_in_tile[index_in_tile];
                PackedParams params = packed_params[point_id];
                RGBA32 point_color;
                point_color.r = params.rg.x;
                point_color.g = params.rg.y;
                point_color.b = params.ba.x;
                point_color.a = params.ba.y;
                float2 xy{ params.pixel_x,params.pixel_y };

                const int pixel_x = tile_pixel_index[batch_id][tile_id][0] + threadIdx.x % tile_size_x;
                const int pixel_y = tile_pixel_index[batch_id][tile_id][1] + threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;
                float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                float bxcy = params.inv_cov11 * d.y + params.inv_cov01 * d.x;
                float axby = params.inv_cov00 * d.x + params.inv_cov01 * d.y;
                float cur_val = -0.5f * (d.x * axby + d.y * bxcy);
                float cur_diff = bxcy - 0.5f * params.inv_cov11;
                float second_diff = -params.inv_cov11;
                //basic+=(cy+bx)*delta - 0.5*c*delta*delta

                any_active = 0;
                unsigned int fragment_count = 0x0;//ushort2
                float weight_sum = 0;
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD; i++)
                {
                    float power = cur_val;
                    cur_val += cur_diff;
                    cur_diff += second_diff;

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
                    if (enable_depth)
                    {
                        reg_buffer[i].d += (params.depth * weight);
                    }
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
        }
        auto output_r = output_img[batch_id][0][tile_id];
        auto output_g = output_img[batch_id][1][tile_id];
        auto output_b = output_img[batch_id][2][tile_id];
        auto output_d = output_depth[batch_id][0][tile_id];
        auto output_t = output_transmitance[batch_id][0][tile_id];
        auto output_last_index = output_last_contributor[batch_id][tile_id];
        #pragma unroll
        for (int i = 0; i < PIXELS_PER_THREAD; i++)
        {
            const int output_x = threadIdx.x % tile_size_x;
            const int output_y = threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE + i;

            output_r[output_y][output_x] = reg_buffer[i].r;
            output_g[output_y][output_x] = reg_buffer[i].g;
            output_b[output_y][output_x] = reg_buffer[i].b;
            output_d[output_y][output_x] = reg_buffer[i].d;
            output_t[output_y][output_x] = reg_buffer[i].t;
            output_last_index[output_y][output_x] = reg_buffer[i].lst_contributor;
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
tile_pixel_index.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),\
reinterpret_cast<PackedParams*>(packed_params.data_ptr<float>()),\
output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),\
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
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, at::Tensor tile_pixel_index,
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

    int64_t viewsnum = tile_start.size(0);
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tile_start.size(1);

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
        dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
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
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, at::Tensor tile_pixel_index,
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

    int64_t viewsnum = tile_start.size(0);
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tile_start.size(1);
    
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
        dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
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
    float r;
    float g;
    float b;
    float t;
    float alpha;
};


template <int tile_size_y, int tile_size_x, int subtile_size_y, int subtile_size_x, bool enable_statistic, bool enable_trans_grad, bool enable_depth_grad>
__global__ void raster_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> primitives_in_tile,    //[batch,items]  
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> tile_pixel_index,    //[batch,tile,2]
    const PackedParams* __restrict__ packed_params,         //[batch,point_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    PackedGrad* __restrict__ packed_grad,         //[batch,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_err_sum,  //[batch,1,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_err_square_sum,  //[batch,1,point_num]
    int tiles_num_x, int img_h, int img_w, int pointsnum)
{
    constexpr int VECTOR_SIZE = 1;
    constexpr int PIXELS_PER_THREAD = (tile_size_x * tile_size_y) / (32 * VECTOR_SIZE);

    __shared__ float shared_img_grad[3][PIXELS_PER_THREAD][4 * 32];
    __shared__ float shared_depth_grad[PIXELS_PER_THREAD][4 * 32];
    __shared__ float shared_trans_grad_buffer[PIXELS_PER_THREAD][4 * 32];
    __shared__ int shared_last_contributor[PIXELS_PER_THREAD][4 * 32];

    const int batch_id = blockIdx.y;
    packed_params += batch_id * pointsnum;
    packed_grad += batch_id * pointsnum;


    int tile_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (tile_id != 0 && tile_id < start_index.size(1))
    {

        int start_index_in_tile = start_index[batch_id][tile_id];
        int index_in_tile = -1;
        
        if (start_index_in_tile != -1)
        {
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
                reg_buffer[i].t = final_transmitance[batch_id][0][tile_id][in_tile_y + i][in_tile_x];
                if (enable_trans_grad)
                {
                    shared_trans_grad_buffer[i][threadIdx.y * blockDim.x + threadIdx.x] = final_transmitance[batch_id][0][tile_id][in_tile_y + i][in_tile_x] * d_trans_img[batch_id][0][tile_id][in_tile_y + i][in_tile_x];
                }
                if (enable_depth_grad)
                {
                    shared_depth_grad[i][threadIdx.y * blockDim.x + threadIdx.x] = d_depth_img[batch_id][0][tile_id][in_tile_y + i][in_tile_x];
                }

                shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x] = 
                    d_img[batch_id][0][tile_id][in_tile_y + i][in_tile_x];
                shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x] =
                    d_img[batch_id][1][tile_id][in_tile_y + i][in_tile_x];
                shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x] =
                    d_img[batch_id][2][tile_id][in_tile_y + i][in_tile_x];
                int last = last_contributor[batch_id][tile_id][in_tile_y + i][in_tile_x] - 1;
                shared_last_contributor[i][threadIdx.y * blockDim.x + threadIdx.x] = last;
                index_in_tile = std::max(last, index_in_tile);
            }
            index_in_tile = __reduce_max_sync(0xffffffff, index_in_tile);

            const int* points_in_tile = &primitives_in_tile[batch_id][start_index_in_tile];
            const int pixel_x = tile_pixel_index[batch_id][tile_id][0] + threadIdx.x % tile_size_x;
            const int pixel_y = tile_pixel_index[batch_id][tile_id][1] + threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE;

            int point_id = 0;
            int prefetch_point_id = points_in_tile[max(index_in_tile,0)];
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

                RGBA32 point_color;
                point_color.r = params.rg.x;
                point_color.g = params.rg.y;
                point_color.b = params.ba.x;
                point_color.a = params.ba.y;

                prefetch_point_id = points_in_tile[max(index_in_tile - 1,0)];
                params = packed_params[prefetch_point_id];

                float grad_r = 0;
                float grad_g = 0;
                float grad_b = 0;
                float grad_d = 0;
                float grad_a = 0;
                float err_square = 0;
                float grad_bxcy = 0;
                float grad_neg_half_c = 0;
                float grad_basic = 0;
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD; i++)
                {
                    float power = cur_val;
                    cur_val += cur_diff;
                    cur_diff += second_diff;
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

                        reg_buffer[i].t = min(reg_buffer[i].t/(1.0f - alpha),1.0f);
                        grad_r += alpha * reg_buffer[i].t * shared_img_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
                        grad_g += alpha * reg_buffer[i].t * shared_img_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
                        grad_b += alpha * reg_buffer[i].t * shared_img_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];
                        if(enable_depth_grad)
                        {
                            grad_d += alpha * reg_buffer[i].t * shared_depth_grad[i][threadIdx.y * blockDim.x + threadIdx.x];
                        }


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
                        grad_neg_half_c += d_power * i * i;
                        grad_basic += d_power;
                    }
                }
                
                PackedGrad* grad_addr = packed_grad + point_id;
                //unsigned mask = __ballot_sync(0xffffffff, grad_opacity!=0);
                if (__any_sync(0xffffffff, grad_a!=0))
                {
                    warp_reduce_sum<float, false>(grad_r);
                    warp_reduce_sum<float, false>(grad_g);
                    warp_reduce_sum<float, false>(grad_b);
                    warp_reduce_sum<float, false>(grad_a);
                    if(enable_depth_grad)
                    {
                        warp_reduce_sum<float, false>(grad_d);
                    }
                    if (threadIdx.x == 0)
                    {
                        atomicAdd(&grad_addr->r, grad_r);
                        atomicAdd(&grad_addr->g, grad_g);
                        atomicAdd(&grad_addr->b, grad_b);
                        atomicAdd(&grad_addr->a, grad_a);
                        if(enable_depth_grad)
                        {
                            atomicAdd(&grad_addr->ddepth,grad_d);
                        }
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
                    float2 d_ndc_xy{ d_dx, d_dy };
                    warp_reduce_sum<float, false>(d_ndc_xy.x);
                    warp_reduce_sum<float, false>(d_ndc_xy.y);
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
    if (index < packed_grad.size(1))
    {
        PackedGrad* grads = (PackedGrad*)&packed_grad[blockIdx.y][index][0];
        d_ndc[blockIdx.y][0][index] = grads->dx * 0.5f * img_w * grad_inv_scaler[0];
        d_ndc[blockIdx.y][1][index] = grads->dy * 0.5f * img_w * grad_inv_scaler[0];
        d_ndc[blockIdx.y][2][index] = grads->ddepth * grad_inv_scaler[0];
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
tile_pixel_index.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),\
reinterpret_cast<PackedParams*>(packed_params.data_ptr<float>()),\
final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),\
last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),\
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
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, at::Tensor tile_pixel_index,
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

    int64_t viewsnum = tile_start.size(0);
    int tilesnum_x = std::ceil(img_w / float(tile_w));
    int tilesnum_y = std::ceil(img_h / float(tile_h));
    int64_t tilesnum = tile_start.size(1);

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
    dim3 Block3d(std::ceil(tilesnum / float(tiles_per_block)), viewsnum, 1);
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



#define MAX_VTILES_PER_PIXEL 1024 // 共享内存链表最大长度，支持极度密集的场景

// ==============================================================================
// Forward Kernel
// ==============================================================================
__global__ void global_blending_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> virtual_tile_next,
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img, 
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance, 
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance_out, // 必须存下来给 Backward 用
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> final_img, 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> final_T, 
    int img_h, int img_w, int tile_h, int tile_w, int tile_w_num
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int b = blockIdx.z; // batch / view_id

    int pixel_x = tile_x * tile_w + tx;
    int pixel_y = tile_y * tile_h + ty;

    // 核心规则：物理 tile_id 从 1 开始，且 0 是 NULL_NODE
    int physical_tile_id = tile_y * tile_w_num + tile_x + 1;

    // --- 使用 Shared Memory 让整个 Block (同一个物理 Tile) 共享链表遍历 ---
    __shared__ int vtile_seq[MAX_VTILES_PER_PIXEL];
    __shared__ int num_vtiles;

    if (tx == 0 && ty == 0) {
        int count = 0;
        int curr = physical_tile_id;
        while (curr != 0 && count < MAX_VTILES_PER_PIXEL) {
            vtile_seq[count++] = curr;
            curr = virtual_tile_next[b][curr];
        }
        num_vtiles = count;
    }
    __syncthreads(); // 等待链表读取完毕

    // 图像边界检查 (必须放在 syncthreads 之后，否则会死锁)
    if (pixel_x >= img_w || pixel_y >= img_h) return;

    // --- 开始 Alpha Blending ---
    float T_acc = 1.0f;
    float C_r = 0.0f, C_g = 0.0f, C_b = 0.0f;

    for (int i = 0; i < num_vtiles; ++i) {
        int vnode = vtile_seq[i];

        // 极其重要：记录当前 Virtual Tile 之前的累积透射率 (T_less_i)，反向传播强依赖这个值！
        final_transmitance_out[b][0][vnode][ty][tx] = T_acc;

        float c_r = output_img[b][0][vnode][ty][tx];
        float c_g = output_img[b][1][vnode][ty][tx];
        float c_b = output_img[b][2][vnode][ty][tx];
        float t = output_transmitance[b][0][vnode][ty][tx];

        C_r += T_acc * c_r;
        C_g += T_acc * c_g;
        C_b += T_acc * c_b;
        T_acc *= t;
    }

    // 写回整图
    final_img[b][0][pixel_y][pixel_x] = C_r;
    final_img[b][1][pixel_y][pixel_x] = C_g;
    final_img[b][2][pixel_y][pixel_x] = C_b;
    final_T[b][0][pixel_y][pixel_x] = T_acc;
}

// ==============================================================================
// Backward Kernel (逆向遍历，避免除零并实现精确求导)
// ==============================================================================
__global__ void global_blending_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> virtual_tile_next,
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img, 
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance, 
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance, // Forward 存下来的 T_less_i
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> grad_out_img, 
    const float* __restrict__ grad_out_T_ptr, // 可能为空
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img, 
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img, 
    int img_h, int img_w, int tile_h, int tile_w, int tile_w_num,
    bool has_grad_T
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int b = blockIdx.z;

    int pixel_x = tile_x * tile_w + tx;
    int pixel_y = tile_y * tile_h + ty;
    int physical_tile_id = tile_y * tile_w_num + tile_x + 1;

    // --- 依然使用 Shared Memory 读取相同的链表 ---
    __shared__ int vtile_seq[MAX_VTILES_PER_PIXEL];
    __shared__ int num_vtiles;

    if (tx == 0 && ty == 0) {
        int count = 0;
        int curr = physical_tile_id;
        while (curr != 0 && count < MAX_VTILES_PER_PIXEL) {
            vtile_seq[count++] = curr;
            curr = virtual_tile_next[b][curr];
        }
        num_vtiles = count;
    }
    __syncthreads();

    if (pixel_x >= img_w || pixel_y >= img_h) return;

    // 拿到 Loss 对最终图像像素的梯度
    float dL_dCr = grad_out_img[b][0][pixel_y][pixel_x];
    float dL_dCg = grad_out_img[b][1][pixel_y][pixel_x];
    float dL_dCb = grad_out_img[b][2][pixel_y][pixel_x];
    
    // 如果有对背景透射率的梯度需求
    float dL_dT = 0.0f;
    if (has_grad_T) {
        // 计算平铺一维索引
        int flat_idx = b * (img_h * img_w) + pixel_y * img_w + pixel_x;
        dL_dT = grad_out_T_ptr[flat_idx];
    }

    // 后缀和累加器 (Suffix Sums)
    float S_r = 0.0f, S_g = 0.0f, S_b = 0.0f;
    float S_t = 1.0f; 

    // --- 逆向遍历链表！完美计算梯度 ---
    for (int i = num_vtiles - 1; i >= 0; --i) {
        int vnode = vtile_seq[i];

        float c_r = output_img[b][0][vnode][ty][tx];
        float c_g = output_img[b][1][vnode][ty][tx];
        float c_b = output_img[b][2][vnode][ty][tx];
        float t = output_transmitance[b][0][vnode][ty][tx];
        
        // 拿到 Forward 时存好的 T_less_i
        float T_less_i = final_transmitance[b][0][vnode][ty][tx];

        // 1. Color 梯度: dL/dC_i = T_{<i} * dL/dC_{final}
        d_img[b][0][vnode][ty][tx] = T_less_i * dL_dCr;
        d_img[b][1][vnode][ty][tx] = T_less_i * dL_dCg;
        d_img[b][2][vnode][ty][tx] = T_less_i * dL_dCb;

        // 2. Transmittance 梯度: dL/dt_i = T_{<i} * Suffix_Color * dL/dC + T_{<i} * Suffix_T * dL/dT
        float dt_color = T_less_i * (S_r * dL_dCr + S_g * dL_dCg + S_b * dL_dCb);
        float dt_trans = T_less_i * S_t * dL_dT;
        d_trans_img[b][0][vnode][ty][tx] = dt_color + dt_trans;

        // 3. 递归更新后缀和，供给下一次迭代 (也就是前一个节点) 使用
        S_r = c_r + t * S_r;
        S_g = c_g + t * S_g;
        S_b = c_b + t * S_b;
        S_t = t * S_t;
    }
}

// ==============================================================================
// C++ 主机端调用封装
// ==============================================================================
std::vector<at::Tensor> global_blending_forward(
    at::Tensor virtual_tile_next,      // [batch, vtiles_num]
    at::Tensor tile_img,             // [batch, 3, vtiles_num, tile_h, tile_w]
    at::Tensor tile_transmitance,    // [batch, 1, vtiles_num, tile_h, tile_w]
    int img_h, int img_w
) {
    int batch_size = tile_img.size(0);
    int vtiles_num = tile_img.size(2);
    int tile_h=tile_img.size(3);
    int tile_w=tile_img.size(4);
    int tile_w_num = std::ceil(img_w / (float)tile_w);
    int tile_h_num = std::ceil(img_h / (float)tile_h);

    auto opt = tile_img.options();
    
    // 分配最终图像和透射率
    at::Tensor final_img = torch::zeros({batch_size, 3, img_h, img_w}, opt);
    at::Tensor final_T = torch::ones({batch_size, 1, img_h, img_w}, opt);
    
    // 极其重要：分配用于暂存 T_less_i 的 Tensor，供反向传播使用
    at::Tensor T_less_i = torch::ones({batch_size, 1, vtiles_num, tile_h, tile_w}, opt);

    dim3 block(tile_w, tile_h, 1);
    dim3 grid(tile_w_num, tile_h_num, batch_size);

    global_blending_forward_kernel<<<grid, block>>>(
        virtual_tile_next.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        tile_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        tile_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        T_less_i.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        final_img.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        final_T.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        img_h, img_w, tile_h, tile_w, tile_w_num
    );

    return {final_img, final_T, T_less_i};
}

std::vector<at::Tensor> global_blending_backward(
    at::Tensor virtual_tile_next,
    at::Tensor tile_img,
    at::Tensor tile_transmitance,
    at::Tensor T_less_i, // 这是 forward 返回的 final_transmitance_out
    at::Tensor grad_out_img,       // [batch, 3, img_h, img_w]
    std::optional<at::Tensor> grad_out_T_arg
) {
    int batch_size = tile_img.size(0);
    int vtiles_num = tile_img.size(2);
    int tile_h=tile_img.size(3);
    int tile_w=tile_img.size(4);
    int img_h=grad_out_img.size(2);
    int img_w=grad_out_img.size(3);
    int tile_w_num = std::ceil(img_w / (float)tile_w);
    int tile_h_num = std::ceil(img_h / (float)tile_h);

    auto opt = tile_img.options();
    at::Tensor d_img = torch::zeros_like(tile_img);
    at::Tensor d_trans_img = torch::zeros_like(tile_transmitance);

    bool has_grad_T = grad_out_T_arg.has_value() && grad_out_T_arg.value().numel() > 0;
    const float* grad_out_T_ptr = has_grad_T ? grad_out_T_arg.value().data_ptr<float>() : nullptr;

    dim3 block(tile_w, tile_h, 1);
    dim3 grid(tile_w_num, tile_h_num, batch_size);

    global_blending_backward_kernel<<<grid, block>>>(
        virtual_tile_next.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        tile_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        tile_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        T_less_i.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        grad_out_img.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_out_T_ptr,
        d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        img_h, img_w, tile_h, tile_w, tile_w_num,
        has_grad_T
    );

    return {d_img, d_trans_img};
}