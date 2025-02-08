#pragma once
#include <torch/extension.h>


std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU,at::Tensor RD,at::Tensor prefix_sum, at::Tensor depth_sorted_pointid,
	at::Tensor large_index,int64_t allocate_size, int64_t TilesSizeX);
at::Tensor tileRange(at::Tensor table_tileId, int64_t table_length, int64_t max_tileId);
