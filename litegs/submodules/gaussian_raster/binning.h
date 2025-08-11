#pragma once
#include <torch/extension.h>


std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU,at::Tensor RD,at::Tensor prefix_sum, at::Tensor depth_sorted_pointid,
	int64_t allocate_size, int64_t img_h,int64_t img_w, int64_t tilesize_h, int64_t tilesize_w);
at::Tensor tileRange(at::Tensor table_tileId, int64_t table_length, int64_t max_tileId);
std::vector<at::Tensor> create_2d_gaussian_ROI(at::Tensor ndc,at::Tensor view_space_z, at::Tensor inv_cov2d, at::Tensor opacity,int64_t height, int64_t width);
std::vector<at::Tensor> get_allocate_size(at::Tensor pixel_left_up, at::Tensor pixel_right_down, int64_t tilesize_h, int64_t tilesize_w,
	int64_t max_tileid_y, int64_t max_tileid_x);
