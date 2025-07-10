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
#include "binning.h"

 __global__ void duplicate_with_keys_kernel(
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> LU,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> RD,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> prefix_sum,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> depth_sorted_pointid,//viewnum,pointnum
    int TilesNumX,
    torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_pointId
    )
{
    int view_id = blockIdx.y;
    

    if (blockIdx.x * blockDim.x + threadIdx.x < prefix_sum.size(1))
    {
        int point_id = depth_sorted_pointid[view_id][blockIdx.x * blockDim.x + threadIdx.x];
        int end = prefix_sum[view_id][blockIdx.x * blockDim.x + threadIdx.x];

        //int end = prefix_sum[view_id][point_id+1];
        int l = LU[view_id][0][point_id];
        int u = LU[view_id][1][point_id];
        int r = RD[view_id][0][point_id];
        int d = RD[view_id][1][point_id];
        int count = 0;
        if ((r - l) * (d - u) < 32)
        {
            for (int i = u; i < d; i++)
            {
                for (int j = l; j < r; j++)
                {
                    int tile_id = i * TilesNumX + j;
                    table_tileId[view_id][end - 1 - count] = tile_id + 1;// tile_id 0 means invalid!
                    table_pointId[view_id][end - 1 - count] = point_id;
                    count++;
                }
            }
        }
    }
}

 __global__ void large_points_duplicate_with_keys_kernel(
     const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> LU,//viewnum,2,pointnum
     const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> RD,//viewnum,2,pointnum
     const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> prefix_sum,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> depth_sorted_pointid,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> large_index,//tasknum,2
     const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> tensor_ellipse_f,//viewnum,2,2,pointnum
     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> tensor_ellipse_a,//viewnum,pointnum
     int img_h, int img_w, int tilesize_h, int tilesize_w,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_pointId
 )
 {
     auto block = cg::this_thread_block();
     auto warp = cg::tiled_partition<32>(block);
     int task_index = warp.meta_group_size() * block.group_index().x + warp.meta_group_rank();

     if (task_index < large_index.size(0))
     {
         int view_id = large_index[task_index][0];
         int point_index = large_index[task_index][1];
         int point_id = depth_sorted_pointid[view_id][point_index];
         int end = prefix_sum[view_id][point_index];

         //ellipse bounding
         float ellipse_a = tensor_ellipse_a[view_id][point_id];
         float2 ellipse_f[2];
         ellipse_f[0].x = tensor_ellipse_f[view_id][0][0][point_id];
         ellipse_f[0].y = tensor_ellipse_f[view_id][0][1][point_id];
         ellipse_f[1].x = tensor_ellipse_f[view_id][1][0][point_id];
         ellipse_f[1].y = tensor_ellipse_f[view_id][1][1][point_id];
         
         //ROI
         int l = LU[view_id][0][point_id];
         int u = LU[view_id][1][point_id];
         int width = RD[view_id][0][point_id]-l;
         int height = RD[view_id][1][point_id]-u;
         for (int i = warp.thread_rank(); i < width * height; i+=warp.num_threads())
         {
             int tile_col = l + (i % width);
             int tile_row = u + (i / width);
             int tile_center_x = (tile_col + 0.5f) * tilesize_w;
             int tile_center_y = (tile_row + 0.5f) * tilesize_h;

             float square_dist0 = (tile_center_x - ellipse_f[0].x) * (tile_center_x - ellipse_f[0].x) 
                 + (tile_center_y - ellipse_f[0].y) * (tile_center_y - ellipse_f[0].y);
             float square_dist1 = (tile_center_x - ellipse_f[1].x) * (tile_center_x - ellipse_f[1].x)
                 + (tile_center_y - ellipse_f[1].y) * (tile_center_y - ellipse_f[1].y);
             const float radius = sqrt(2.0f)* max(tilesize_w,tilesize_h)/2;
             if (sqrt(square_dist0) + sqrt(square_dist1) - 2 * radius <= 2 * ellipse_a)
             {
                 int tile_id = tile_row * ceil(img_w / (float)tilesize_w) + tile_col;
                 table_tileId[view_id][end - 1 - i] = tile_id + 1;// tile_id 0 means invalid!
                 table_pointId[view_id][end - 1 - i] = point_id;
             }
             else
             {
                 table_tileId[view_id][end - 1 - i] = 0;
             }
             
         }
     }
 }

 std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU, at::Tensor RD, at::Tensor prefix_sum, at::Tensor depth_sorted_pointid,
     at::Tensor large_index, at::Tensor ellipse_f, at::Tensor ellipse_a,
     int64_t allocate_size, int64_t img_h, int64_t img_w, int64_t tilesize_h,int64_t tilesize_w)
{
    at::DeviceGuard guard(LU.device());
    int64_t view_num = LU.sizes()[0];
    int64_t points_num = LU.sizes()[2];

    std::vector<int64_t> output_shape{ view_num, allocate_size };

    auto opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_tileId = torch::zeros(output_shape, opt);
    opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_pointId= torch::zeros(output_shape, opt);

    dim3 Block3d(std::ceil(points_num/512.0f), view_num, 1);
    

    duplicate_with_keys_kernel<<<Block3d ,512>>>(
        LU.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        RD.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        prefix_sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        depth_sorted_pointid.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        int(std::ceil(img_w/(float)tilesize_w)),
        table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        table_pointId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    
    int large_points_num = large_index.size(0);
    int blocksnum = std::ceil((large_points_num * 32) / 512.0f);
    large_points_duplicate_with_keys_kernel << <blocksnum, 512 >> > (
        LU.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        RD.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        prefix_sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        depth_sorted_pointid.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        large_index.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        ellipse_f.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        ellipse_a.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        img_h,img_w, tilesize_h,tilesize_w,
        table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        table_pointId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return { table_tileId ,table_pointId };
    
}

__global__ void tile_range_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> table_tileId,//viewnum,pointnum
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

    dim3 Block3d(std::ceil(table_length / 512.0f), view_num, 1);

    tile_range_kernel<<<Block3d, 512 >>>
        (table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(), table_length, max_tileId, out.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return out;
}

__global__ void create_ROI_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> tensor_ndc,        //viewnum,4,pointnum
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> view_space_z,        //viewnum,pointnum
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> tensor_eigen_val,  //viewnum,2,pointnum
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> tensor_eigen_vec,  //viewnum,2,2,pointnum
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> tensor_opacity,  //viewnum,pointnum
    int img_h,int img_w,
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_left_up,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_right_down,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < float, 4, torch::RestrictPtrTraits> tensor_ellipse_f,//viewnum,2,2,pointnum
    torch::PackedTensorAccessor32 < float, 2, torch::RestrictPtrTraits> tensor_ellipse_a//viewnum,pointnum
)
{
    int view_id = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < tensor_ndc.size(2))
    {
        float4 ndc{ tensor_ndc[view_id][0][index],tensor_ndc[view_id][1][index],
            tensor_ndc[view_id][2][index] ,tensor_ndc[view_id][3][index] };
        bool bVisible = !((ndc.x < -1.3f) || (ndc.x > 1.3f) || (ndc.y < -1.3f) || (ndc.y > 1.3f) || (view_space_z[view_id][index] <= 0.2f));
        if (bVisible)
        {
            float opacity = max(tensor_opacity[view_id][index], 1.0f / 255);
            float coefficient = 2 * log(255 * opacity);
            float axis_length[2]{ 0,0 };
            axis_length[0] = sqrt(coefficient * tensor_eigen_val[view_id][0][index]);
            axis_length[1] = sqrt(coefficient * tensor_eigen_val[view_id][1][index]);
            float2 axis_dir[2];
            axis_dir[0].x = tensor_eigen_vec[view_id][0][0][index];
            axis_dir[0].y = tensor_eigen_vec[view_id][1][0][index];
            axis_dir[1].x = tensor_eigen_vec[view_id][0][1][index];
            axis_dir[1].y = tensor_eigen_vec[view_id][1][1][index];
            float2 axis[2];
            axis[0].x = axis_dir[0].x * axis_length[0];
            axis[0].y = axis_dir[0].y * axis_length[0];
            axis[1].x = axis_dir[1].x * axis_length[1];
            axis[1].y = axis_dir[1].y * axis_length[1];
            float2 screen_uv{ ndc.x * 0.5f + 0.5f,ndc.y * 0.5f + 0.5f };
            float2 coord{ screen_uv.x * img_w - 0.5f,screen_uv.y * img_h - 0.5f };

            {
                //ellipse
                int major_index = (axis_length[0] > axis_length[1]) ? 0 : 1;
                int minor_index = (major_index + 1) % 2;
                float ellipse_a = axis_length[major_index];
                float ellipse_b = axis_length[minor_index];
                float ellipse_c = sqrt(ellipse_a * ellipse_a - ellipse_b * ellipse_b);
                float fscale = ellipse_c / ellipse_a;
                float2 ellipse_f[2];
                ellipse_f[0].x = coord.x + axis[major_index].x * fscale;
                ellipse_f[0].y = coord.y + axis[major_index].y * fscale;
                ellipse_f[1].x = coord.x - axis[major_index].x * fscale;
                ellipse_f[1].y = coord.y - axis[major_index].y * fscale;
                tensor_ellipse_f[view_id][0][0][index] = ellipse_f[0].x;
                tensor_ellipse_f[view_id][0][1][index] = ellipse_f[0].y;
                tensor_ellipse_f[view_id][1][0][index] = ellipse_f[1].x;
                tensor_ellipse_f[view_id][1][1][index] = ellipse_f[1].y;
                tensor_ellipse_a[view_id][index] = ellipse_a;
            }

            {
                //2d AABB
                int min_x = floor(coord.x - abs(axis[0].x) - abs(axis[1].x));
                int max_x = ceil(coord.x + abs(axis[0].x) + abs(axis[1].x));
                int min_y = floor(coord.y - abs(axis[0].y) - abs(axis[1].y));
                int max_y = ceil(coord.y + abs(axis[0].y) + abs(axis[1].y));
                int2 left_up{ min_x ,min_y };
                int2 right_down{ max_x ,max_y };
                tensor_left_up[view_id][0][index] = left_up.x;
                tensor_left_up[view_id][1][index] = left_up.y;
                tensor_right_down[view_id][0][index] = right_down.x;
                tensor_right_down[view_id][1][index] = right_down.y;
            }
            
        }
        else
        {
            tensor_left_up[view_id][0][index] = -1;
            tensor_left_up[view_id][1][index] = -1;
            tensor_right_down[view_id][0][index] = -1;
            tensor_right_down[view_id][1][index] = -1;
        }
    }
}

std::vector<at::Tensor> create_2d_gaussian_ROI(at::Tensor ndc, at::Tensor view_space_z, at::Tensor eigen_val, at::Tensor eigen_vec, at::Tensor opacity,
    int64_t height,int64_t width)
{
    at::DeviceGuard guard(ndc.device());

    int views_num = ndc.size(0);
    int points_num = ndc.size(2);
    at::Tensor left_up = torch::empty({ views_num,2,points_num }, ndc.options().dtype(torch::kInt32));
    at::Tensor right_down = torch::empty({ views_num,2,points_num }, ndc.options().dtype(torch::kInt32));
    at::Tensor ellipse_f = torch::empty({ views_num,2,2,points_num }, ndc.options());
    at::Tensor ellipse_a = torch::empty({ views_num,points_num }, ndc.options());

    dim3 Block3d(std::ceil(points_num / 256.0f), views_num, 1);
    create_ROI_kernel<<<Block3d,256>>>(ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        view_space_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        eigen_val.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        eigen_vec.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        height, width,
        left_up.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        right_down.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        ellipse_f.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        ellipse_a.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );
    CUDA_CHECK_ERRORS;
    return { left_up ,right_down,ellipse_f,ellipse_a };
}

template<int tileH,int tileW>
__global__ void get_allocate_size_kernel(
    const torch::PackedTensorAccessor32< int32_t, 3, torch::RestrictPtrTraits> tensor_pixel_left_up,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32< int32_t, 3, torch::RestrictPtrTraits> tensor_pixel_right_down,//viewnum,2,pointnum
    int max_tileid_y,int max_tileid_x,
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_tile_left_up,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_tile_right_down,//viewnum,2,pointnum
    torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> tensor_allocate//viewnum,pointnum
)
{
    int view_id = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < tensor_pixel_left_up.size(2))
    {
        int pixel_left = tensor_pixel_left_up[view_id][0][index];
        int pixel_up = tensor_pixel_left_up[view_id][1][index];
        int pixel_right = tensor_pixel_right_down[view_id][0][index];
        int pixel_down = tensor_pixel_right_down[view_id][1][index];

        int tile_left = min(max(0, pixel_left / tileW),max_tileid_x);
        int tile_right = min(max(0, (pixel_right + tileW - 1) / tileW), max_tileid_x);
        int tile_up = min(max(0, pixel_up / tileH), max_tileid_y);
        int tile_down = min(max(0, (pixel_down + tileH - 1) / tileH), max_tileid_y);

        tensor_tile_left_up[view_id][0][index] = tile_left;
        tensor_tile_left_up[view_id][1][index] = tile_up;
        tensor_tile_right_down[view_id][0][index] = tile_right;
        tensor_tile_right_down[view_id][1][index] = tile_down;

        int allocate_size = (tile_right - tile_left) * (tile_down - tile_up);
        tensor_allocate[view_id][index] = allocate_size;
    }
}

std::vector<at::Tensor> get_allocate_size(at::Tensor pixel_left_up, at::Tensor pixel_right_down, int64_t tilesize_h, int64_t tilesize_w,
    int64_t max_tileid_y, int64_t max_tileid_x)
{
    at::DeviceGuard guard(pixel_left_up.device());
    assert((tilesize_h == 8) && (tilesize_w == 16));

    int views_num = pixel_left_up.size(0);
    int points_num = pixel_left_up.size(2);
    at::Tensor tile_left_up = torch::empty({ views_num,2,points_num }, pixel_left_up.options());
    at::Tensor tile_right_down = torch::empty({ views_num,2,points_num }, pixel_left_up.options());
    at::Tensor allocate_size = torch::empty({ views_num,points_num }, pixel_left_up.options());

    dim3 Block3d(std::ceil(points_num / 256.0f), views_num, 1);
    get_allocate_size_kernel<8,16> << <Block3d, 256 >> > (
        pixel_left_up.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        pixel_right_down.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        max_tileid_y, max_tileid_x,
        tile_left_up.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        tile_right_down.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        allocate_size.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
    return { tile_left_up,tile_right_down,allocate_size };
}