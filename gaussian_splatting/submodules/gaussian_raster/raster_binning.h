#include <torch/extension.h>
using namespace at;

std::vector<Tensor> duplicateWithKeys(Tensor L,Tensor U,Tensor R,Tensor D,Tensor ValidPointNum,Tensor prefix_sum, int64_t allocate_size, int64_t TilesSizeX);
Tensor tileRange(Tensor table_tileId, int64_t table_length, int64_t max_tileId);

std::vector<Tensor> rasterize_forward(
    Tensor sorted_points,
    Tensor start_index,
    Tensor mean2d,// 
    Tensor cov2d_inv,
    Tensor color,
    Tensor opacity,
    Tensor tiles,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y,
    int64_t img_h,
    int64_t img_w
);

std::vector<Tensor> rasterize_backward(
    Tensor sorted_points,
    Tensor start_index,
    Tensor mean2d,
    Tensor cov2d_inv,
    Tensor color,
    Tensor opacity,
    Tensor tiles,
    Tensor final_transmitance,
    Tensor last_contributor,
    Tensor d_img,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y,
    int64_t img_h,
    int64_t img_w
);

Tensor jacobianRayspace(Tensor translate_position,Tensor camera_focal);

Tensor createTransformMatrix_forward(Tensor quaternion, Tensor scale);
std::vector<Tensor> createTransformMatrix_backward(Tensor transform_matrix_grad, Tensor quaternion, Tensor scale);

Tensor world2ndc_backword(Tensor view_project_matrix, Tensor position, Tensor repc_hom_w, Tensor grad_ndcpos);