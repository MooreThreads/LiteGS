#pragma once
#include <torch/extension.h>


std::vector<at::Tensor> create_table(
	at::Tensor ndc, at::Tensor inv_cov2d, at::Tensor opacity, at::Tensor offset, at::Tensor depth_sorted_pointid,
	std::optional<at::Tensor> feedback_buffer_cpu, std::optional<at::Tensor> idx_tensor_cpu,
	int64_t height, int64_t width, int64_t tile_size_h, int64_t tile_size_w
);
at::Tensor tileRange(at::Tensor table_tileId, int64_t max_tileId);
std::vector<at::Tensor> get_allocate_size(
	at::Tensor ndc,at::Tensor view_space_z, at::Tensor inv_cov2d, at::Tensor opacity,
	int64_t height, int64_t width, int64_t tilesize_h, int64_t tilesize_w,
	std::optional<at::Tensor> valid_length
);
