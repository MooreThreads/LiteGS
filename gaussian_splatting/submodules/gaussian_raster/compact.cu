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
#include"compact.h"


__global__ void compact_visible_params_kernel_forward(
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> visible_mask,    //[chunk_num] 
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible_mask_cumsum,    //[chunk_num] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position,    //[4,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale,    //[3,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation,    //[4,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base,    //[1,3,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest,    //[?,3,chunk_num,chunk_size] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity,    //[1,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_position,    //[4,p] 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_scale,    //[3,p] 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_rotation,    //[4,p] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_sh_base,    //[1,3,p] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_sh_rest,    //[?,3,p] 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_opacity,    //[1,p] 
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> reverse_map     //[visible_chunk_num]
)
{
    if (visible_mask[blockIdx.x] == true)
    {
        int chunksize = position.size(2);
        int compacted_index = (visible_mask_cumsum[blockIdx.x]-1)* chunksize;
        if (threadIdx.x == 0)
        {
            reverse_map[compacted_index/ chunksize] = blockIdx.x;
        }

        for (int index = threadIdx.x; index < chunksize; index += blockDim.x)
        {
            //copy
            compacted_position[0][compacted_index + index] = position[0][blockIdx.x][index];
            compacted_position[1][compacted_index + index] = position[1][blockIdx.x][index];
            compacted_position[2][compacted_index + index] = position[2][blockIdx.x][index];
            compacted_position[3][compacted_index + index] = position[3][blockIdx.x][index];
            compacted_scale[0][compacted_index + index] = scale[0][blockIdx.x][index];
            compacted_scale[1][compacted_index + index] = scale[1][blockIdx.x][index];
            compacted_scale[2][compacted_index + index] = scale[2][blockIdx.x][index];
            compacted_rotation[0][compacted_index + index] = rotation[0][blockIdx.x][index];
            compacted_rotation[1][compacted_index + index] = rotation[1][blockIdx.x][index];
            compacted_rotation[2][compacted_index + index] = rotation[2][blockIdx.x][index];
            compacted_rotation[3][compacted_index + index] = rotation[3][blockIdx.x][index];
            compacted_sh_base[0][0][compacted_index + index] = sh_base[0][0][blockIdx.x][index];
            compacted_sh_base[0][1][compacted_index + index] = sh_base[0][1][blockIdx.x][index];
            compacted_sh_base[0][2][compacted_index + index] = sh_base[0][2][blockIdx.x][index];
            for (int i = 0; i < sh_rest.size(0); i++)
            {
                compacted_sh_rest[i][0][compacted_index + index] = sh_rest[i][0][blockIdx.x][index];
                compacted_sh_rest[i][1][compacted_index + index] = sh_rest[i][1][blockIdx.x][index];
                compacted_sh_rest[i][2][compacted_index + index] = sh_rest[i][2][blockIdx.x][index];
            }
            compacted_opacity[0][compacted_index + index] = opacity[0][blockIdx.x][index];
        }
    }
}


std::vector<at::Tensor> compact_visible_params_forward(int64_t visible_num, at::Tensor visible_mask, at::Tensor visible_mask_cumsum, at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity)
{
    int64_t chunknum = position.size(1);
    int64_t chunksize = position.size(2);

    auto tensor_shape = position.sizes();
    at::Tensor compacted_position = torch::empty({ tensor_shape[0],chunksize * visible_num }, position.options());
    tensor_shape = scale.sizes();
    at::Tensor compacted_scale = torch::empty({ tensor_shape[0],chunksize * visible_num }, scale.options());
    tensor_shape = rotation.sizes();
    at::Tensor compacted_rotation = torch::empty({ tensor_shape[0],chunksize * visible_num }, rotation.options());

    tensor_shape = sh_base.sizes();
    at::Tensor compacted_sh_base = torch::empty({ tensor_shape[0],tensor_shape[1],chunksize * visible_num }, sh_base.options());
    tensor_shape = sh_rest.sizes();
    at::Tensor compacted_sh_rest = torch::empty({ tensor_shape[0],tensor_shape[1],chunksize * visible_num }, sh_rest.options());

    tensor_shape = opacity.sizes();
    at::Tensor compacted_opacity = torch::empty({ tensor_shape[0],chunksize * visible_num }, opacity.options());

    at::Tensor reverse_map = torch::empty({ visible_num }, visible_mask_cumsum.options().requires_grad(false));

    //dim3 Block3d(32, 1, 1);
    compact_visible_params_kernel_forward<<<chunknum, 256 >>>(
        visible_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        visible_mask_cumsum.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rotation.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sh_base.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        sh_rest.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_position.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_scale.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_rotation.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_sh_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_sh_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        reverse_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>()
    );
    CUDA_CHECK_ERRORS;
    return { compacted_position,compacted_scale,compacted_rotation,compacted_sh_base,compacted_sh_rest,compacted_opacity,reverse_map };
}


__global__ void compact_visible_params_kernel_backward(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> reverse_map,     //[visible_chunk_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_position_grad,    //[4,P] 
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_scale_grad,    //[3,P] 
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_rotation_grad,    //[4,P] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_sh_base_grad,    //[1,3,P] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> compacted_sh_rest_grad,    //[?,3,P] 
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> compacted_opacity_grad,    //[1,P] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> position_grad,    //[4,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> scale_grad,    //[3,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> rotation_grad,    //[4,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_base_grad,    //[1,3,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> sh_rest_grad,    //[?,3,chunk_num,chunk_size] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> opacity_grad    //[1,chunk_num,chunk_size] 
)
{
    //assert blockDim.x==chunksize

    if (blockIdx.x<reverse_map.size(0))
    {
        int chunk_index = reverse_map[blockIdx.x];
        int chunk_size = position_grad.size(2);
        int offset = blockIdx.x * chunk_size;

        //copy
        for (int index = threadIdx.x; index < chunk_size; index += blockDim.x)
        {
            position_grad[0][chunk_index][index] = compacted_position_grad[0][offset+index];
            position_grad[1][chunk_index][index] = compacted_position_grad[1][offset+index];
            position_grad[2][chunk_index][index] = compacted_position_grad[2][offset+index];

            scale_grad[0][chunk_index][index] = compacted_scale_grad[0][offset+index];
            scale_grad[1][chunk_index][index] = compacted_scale_grad[1][offset+index];
            scale_grad[2][chunk_index][index] = compacted_scale_grad[2][offset+index];

            rotation_grad[0][chunk_index][index] = compacted_rotation_grad[0][offset+index];
            rotation_grad[1][chunk_index][index] = compacted_rotation_grad[1][offset+index];
            rotation_grad[2][chunk_index][index] = compacted_rotation_grad[2][offset+index];
            rotation_grad[3][chunk_index][index] = compacted_rotation_grad[3][offset+index];

            sh_base_grad[0][0][chunk_index][index] = compacted_sh_base_grad[0][0][offset+index];
            sh_base_grad[0][1][chunk_index][index] = compacted_sh_base_grad[0][1][offset+index];
            sh_base_grad[0][2][chunk_index][index] = compacted_sh_base_grad[0][2][offset+index];

            for (int i = 0; i < compacted_sh_rest_grad.size(0); i++)
            {
                sh_rest_grad[i][0][chunk_index][index] = compacted_sh_rest_grad[i][0][offset+index];
                sh_rest_grad[i][1][chunk_index][index] = compacted_sh_rest_grad[i][1][offset+index];
                sh_rest_grad[i][2][chunk_index][index] = compacted_sh_rest_grad[i][2][offset+index];
            }

            opacity_grad[0][chunk_index][index] = compacted_opacity_grad[0][offset+index];
        }
    }
}

std::vector<at::Tensor> compact_visible_params_backward(int64_t chunk_num, int64_t chunk_size, at::Tensor reverse_map, 
    at::Tensor compacted_position_grad, at::Tensor compacted_scale_grad, at::Tensor compacted_rotation_grad, 
    at::Tensor compacted_sh_base_grad, at::Tensor compacted_sh_rest_grad, at::Tensor compacted_opacity_grad)
{

    auto tensor_shape = compacted_position_grad.sizes();
    at::Tensor position_grad = torch::zeros({ tensor_shape[0], chunk_num,chunk_size }, compacted_position_grad.options());
    
    tensor_shape = compacted_scale_grad.sizes();
    at::Tensor scale_grad = torch::zeros({  tensor_shape[0],chunk_num,chunk_size }, compacted_scale_grad.options());

    tensor_shape = compacted_rotation_grad.sizes();
    at::Tensor rotation_grad = torch::zeros({  tensor_shape[0],chunk_num,chunk_size }, compacted_rotation_grad.options());

    tensor_shape = compacted_sh_base_grad.sizes();
    at::Tensor sh_base_grad = torch::zeros({  tensor_shape[0],tensor_shape[1],chunk_num,chunk_size }, compacted_sh_base_grad.options());

    tensor_shape = compacted_sh_rest_grad.sizes();
    at::Tensor sh_rest_grad = torch::zeros({  tensor_shape[0],tensor_shape[1],chunk_num,chunk_size }, compacted_sh_rest_grad.options());

    tensor_shape = compacted_opacity_grad.sizes();
    at::Tensor opacity_grad = torch::zeros({  tensor_shape[0],chunk_num,chunk_size }, compacted_opacity_grad.options());

    int visible_chunknum = compacted_position_grad.size(1) / chunk_size;
    compact_visible_params_kernel_backward<<<visible_chunknum,256>>>(
        reverse_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        compacted_position_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_scale_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_rotation_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        compacted_sh_base_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_sh_rest_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        compacted_opacity_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        position_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        scale_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rotation_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sh_base_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        sh_rest_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
        );
    CUDA_CHECK_ERRORS;
    return { position_grad,scale_grad,rotation_grad,sh_base_grad,sh_rest_grad,opacity_grad };
}