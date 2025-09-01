#pragma once
#include <torch/extension.h>

std::vector<at::Tensor> create_subtile(at::Tensor primitive_list, at::Tensor tile_start, at::Tensor tile_end, at::Tensor complex_tileid, int64_t allocate_size,
	at::Tensor ndc, at::Tensor inv_cov2d, at::Tensor opacity, int64_t height, int64_t width, int64_t tile_size_h, int64_t tile_size_w);
std::vector<at::Tensor> create_table(at::Tensor ndc, at::Tensor inv_cov2d, at::Tensor opacity, at::Tensor offset, at::Tensor depth_sorted_pointid,
	int64_t allocate_size, int64_t height, int64_t width, int64_t tile_size_h, int64_t tile_size_w);
std::vector<at::Tensor> tileRange(at::Tensor table_tileId, int64_t table_length, int64_t max_tileId);
std::vector<at::Tensor> get_allocate_size(at::Tensor ndc,at::Tensor view_space_z, at::Tensor inv_cov2d, at::Tensor opacity,
	int64_t height, int64_t width, int64_t tilesize_h, int64_t tilesize_w);
