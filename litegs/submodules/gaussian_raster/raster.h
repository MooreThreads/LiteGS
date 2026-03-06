#pragma once
#include <torch/extension.h>
std::vector<at::Tensor> rasterize_forward(
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, at::Tensor tile_pixel_index,
    at::Tensor ndc,
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity, 
    int64_t img_h,
    int64_t img_w,
    int64_t tilesize_h,
    int64_t tilesize_w,
    bool enable_statistic,
    bool enable_trans,
    bool enable_depth
);

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
);

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
    int64_t tilesize_h,
    int64_t tilesize_w,
    bool enable_statistic
);

std::vector<at::Tensor> global_blending_forward(
    at::Tensor virtual_tile_next,      // [batch, vtiles_num]
    at::Tensor tile_img,             // [batch, 3, vtiles_num, tile_h, tile_w]
    at::Tensor tile_transmitance,    // [batch, 1, vtiles_num, tile_h, tile_w]
    int img_h, int img_w
);

std::vector<at::Tensor> global_blending_backward(
    at::Tensor virtual_tile_next,
    at::Tensor tile_img,
    at::Tensor tile_transmitance,
    at::Tensor T_less_i, // forward return
    at::Tensor grad_out_img,       // [batch, 3, img_h, img_w]
    std::optional<at::Tensor> grad_out_T_arg
);
