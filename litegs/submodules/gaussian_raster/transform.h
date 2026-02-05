#pragma once
#include <torch/extension.h>
at::Tensor jacobianRayspace(at::Tensor translate_position,at::Tensor proj_matrix,int64_t output_h,int64_t output_w, std::optional<at::Tensor> valid_length);

at::Tensor createTransformMatrix_forward(at::Tensor quaternion, at::Tensor scale, std::optional<at::Tensor> valid_length);
std::vector<at::Tensor> createTransformMatrix_backward(at::Tensor transform_matrix_grad, at::Tensor quaternion, at::Tensor scale, std::optional<at::Tensor> valid_length);

std::vector<at::Tensor> world2ndc_forward(at::Tensor world_position,at::Tensor view_project_matrix);
at::Tensor world2ndc_backword(at::Tensor view_project_matrix, at::Tensor position, at::Tensor repc_hom_w, at::Tensor grad_ndcpos);

std::vector<at::Tensor> mvp_transform_forward(at::Tensor world_position, at::Tensor view_matrix, at::Tensor proj_matrix, std::optional<at::Tensor> valid_length);
at::Tensor mvp_transform_backward(
    at::Tensor grad_ndc_pos, at::Tensor grad_view_pos,
    at::Tensor view_matrix, at::Tensor proj_matrix, at::Tensor view_pos,
    std::optional<at::Tensor> valid_length
);


at::Tensor createCov2dDirectly_forward(at::Tensor J, at::Tensor view_matrix,at::Tensor transform_matrix, std::optional<at::Tensor> valid_length);
at::Tensor createCov2dDirectly_backward(at::Tensor cov2d_grad, at::Tensor J, at::Tensor view_matrix, at::Tensor transform_matrix, std::optional<at::Tensor> valid_length);

at::Tensor sh2rgb_forward(int64_t degree, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor dir);
std::vector<at::Tensor> sh2rgb_backward(int64_t degree, at::Tensor rgb_grad, int64_t sh_rest_dim, at::Tensor dir, at::Tensor SH_base, at::Tensor SH_rest);

std::vector<at::Tensor> eigh_and_inv_2x2matrix_forward(at::Tensor input, std::optional<at::Tensor> valid_length);
at::Tensor inv_2x2matrix_backward(at::Tensor inv_matrix, at::Tensor dL_dInvMatrix, std::optional<at::Tensor> valid_length);