#pragma once
#include <torch/extension.h>
std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor ndc,// 
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    at::Tensor specific_tiles,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w
);

std::vector<at::Tensor> rasterize_backward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor ndc,
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    at::Tensor specific_tiles,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w
);
