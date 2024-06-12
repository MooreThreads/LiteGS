#include <torch/extension.h>


std::vector<at::Tensor> duplicateWithKeys(at::Tensor L,at::Tensor U,at::Tensor R,at::Tensor D,at::Tensor ValidPointNum,at::Tensor prefix_sum, int64_t allocate_size, int64_t TilesSizeX);
at::Tensor tileRange(at::Tensor table_tileId, int64_t table_length, int64_t max_tileId);

std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor mean2d,// 
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    at::Tensor tiles,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y,
    int64_t img_h,
    int64_t img_w
);

std::vector<at::Tensor> rasterize_backward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor mean2d,
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    at::Tensor tiles,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    int64_t tilesize,
    int64_t tilesnum_x,
    int64_t tilesnum_y,
    int64_t img_h,
    int64_t img_w
);

at::Tensor jacobianRayspace(at::Tensor translate_position,at::Tensor camera_focal,bool bTranspose);

at::Tensor createTransformMatrix_forward(at::Tensor quaternion, at::Tensor scale);
std::vector<at::Tensor> createTransformMatrix_backward(at::Tensor transform_matrix_grad, at::Tensor quaternion, at::Tensor scale);

at::Tensor world2ndc_backword(at::Tensor view_project_matrix, at::Tensor position, at::Tensor repc_hom_w, at::Tensor grad_ndcpos);

at::Tensor createCov2dDirectly_forward(at::Tensor J, at::Tensor view_matrix,at::Tensor transform_matrix);

at::Tensor createCov2dDirectly_backward(at::Tensor cov2d_grad, at::Tensor J, at::Tensor view_matrix, at::Tensor transform_matrix);

at::Tensor sh2rgb_forward(int64_t degree, at::Tensor sh, at::Tensor dir);

at::Tensor sh2rgb_backward(int64_t degree, at::Tensor rgb_grad, at::Tensor sh, at::Tensor dir);
