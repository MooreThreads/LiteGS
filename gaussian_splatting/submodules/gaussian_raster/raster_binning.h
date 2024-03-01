#include <torch/extension.h>
using namespace at;

std::vector<Tensor> duplicateWithKeys(Tensor L,Tensor U,Tensor R,Tensor D,Tensor ValidPointNum,Tensor prefix_sum, int64_t allocate_size, int64_t TilesSizeX);
Tensor tileRange(Tensor table_tileId, int64_t table_length, int64_t max_tileId);

std::vector<Tensor> rasterize_forward(
    Tensor sorted_points,
    Tensor start_index,
    Tensor gathered_mean2d,// 
    Tensor gathered_cov2d_inv,
    Tensor gathered_color,
    Tensor gathered_opacity,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y
);

std::vector<Tensor> rasterize_backward(
    Tensor sorted_points,
    Tensor start_index,
    Tensor mean2d,
    Tensor cov2d_inv,
    Tensor color,
    Tensor opacity,
    Tensor final_transmitance,
    Tensor last_contributor,
    Tensor d_img,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y
);