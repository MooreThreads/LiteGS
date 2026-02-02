#pragma once
#include <torch/extension.h>
std::vector<at::Tensor> cull_compact_activate(
	int sh_degree,
	at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
	at::Tensor view_matrix,
	at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity
);

std::vector<at::Tensor> activate_backward(
	int sh_degree,
	at::Tensor visible_chunk_id, at::Tensor visible_chunks_num,
	at::Tensor view_matrix, 
	at::Tensor position, at::Tensor scale, at::Tensor rotation, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity,
	at::Tensor activated_position_grad, at::Tensor activated_scale_grad, at::Tensor activated_rotation_grad, at::Tensor color_grad, at::Tensor activated_opacity_grad
);

void adamUpdate(
	torch::Tensor param, torch::Tensor param_grad,
	torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
	torch::Tensor visible_index, std::optional<torch::Tensor> valid_length,
	const double lr, const double b1, const double b2, const double eps
);
std::vector<at::Tensor> create_viewproj_forward(at::Tensor view_params, at::Tensor recp_tan_half_fov_x, int img_h, int img_w, float z_near, float z_far);
std::vector<at::Tensor> create_viewproj_backward(at::Tensor view_matrix_grad, at::Tensor proj_matrix_grad, at::Tensor viewproj_matrix_grad, 
	at::Tensor view_params, at::Tensor recp_tan_half_fov_x,int img_h, int img_w, float z_near, float z_far);

std::vector<at::Tensor> frustum_culling_aabb(at::Tensor aabb_origin, at::Tensor aabb_ext, at::Tensor frustumplane,std::optional<at::Tensor> feedback_buffer_arg, std::optional<at::Tensor> data_idx_arg);