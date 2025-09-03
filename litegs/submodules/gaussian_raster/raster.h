#pragma once
#include <torch/extension.h>
std::vector<at::Tensor> rasterize_forward(
    at::Tensor primitives_in_tile, at::Tensor tile_start, at::Tensor tile_end, std::optional<at::Tensor>  specific_tiles_arg,
    std::optional<at::Tensor> primitives_in_subtile_arg, std::optional<at::Tensor> subtile_start_arg, std::optional<at::Tensor> subtile_end_arg, std::optional<at::Tensor>  complex_tiles_arg,
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
);

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
);
