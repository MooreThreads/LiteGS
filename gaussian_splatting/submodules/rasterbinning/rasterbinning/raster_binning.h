#include <torch/extension.h>
using namespace at;

std::vector<Tensor> duplicateWithKeys(Tensor L,Tensor U,Tensor R,Tensor D,Tensor ValidPointNum,Tensor prefix_sum, int64_t allocate_size, int64_t TilesSizeX);
Tensor tileRange(Tensor table_tileId, int64_t table_length, int64_t max_tileId);