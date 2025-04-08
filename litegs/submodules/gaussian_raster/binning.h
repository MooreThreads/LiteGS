#pragma once
#include <torch/extension.h>


std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU,at::Tensor RD,at::Tensor prefix_sum, at::Tensor depth_sorted_pointid,
	at::Tensor large_index, at::Tensor ellipse_f, at::Tensor ellipse_a,
	int64_t allocate_size, int64_t img_h,int64_t img_w,int64_t tilesize);
at::Tensor tileRange(at::Tensor table_tileId, int64_t table_length, int64_t max_tileId);
std::vector<at::Tensor> create_2d_gaussian_ROI(at::Tensor ndc, at::Tensor eigen_val, at::Tensor eigen_vec, at::Tensor opacity,
	int64_t height, int64_t width);
