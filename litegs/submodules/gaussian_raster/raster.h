#pragma once
#include <torch/extension.h>
std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor tile_start, at::Tensor tile_end,
    at::Tensor ndc,
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity, 
    std::optional<at::Tensor> specific_tiles,
    int64_t img_h,
    int64_t img_w,
    int64_t tilesize_h,
    int64_t tilesize_w,
    bool enable_statistic,
    bool enable_trans,
    bool enable_depth
);

std::vector<at::Tensor> rasterize_forward_packed(
    at::Tensor sorted_points,
    at::Tensor tile_start, at::Tensor tile_end,
    at::Tensor packed_params,
    std::optional<at::Tensor>  specific_tiles_arg,
    int64_t img_h,
    int64_t img_w,
    int64_t tile_h,
    int64_t tile_w,
    bool enable_statistic,
    bool enable_trans,
    bool enable_depth
);

std::vector<at::Tensor> rasterize_backward(
    at::Tensor sorted_points,
    at::Tensor tile_start, at::Tensor tile_end,
    at::Tensor packed_params,
    std::optional<at::Tensor> specific_tiles,
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
