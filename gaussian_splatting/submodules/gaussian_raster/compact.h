#pragma once
#include <torch/extension.h>
std::vector<at::Tensor> compact_visible_params_forward(int64_t visible_num,at::Tensor visible_mask, at::Tensor visible_mask_cumsum, 
    at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity);
std::vector<at::Tensor> compact_visible_params_backward(int64_t chunk_num, int64_t chunk_size, at::Tensor reverse_map,
    at::Tensor compacted_position_grad, at::Tensor compacted_scale_grad, at::Tensor compacted_rotation_grad,
    at::Tensor compacted_sh_base_grad, at::Tensor compacted_sh_rest_grad, at::Tensor compacted_opacity_grad);