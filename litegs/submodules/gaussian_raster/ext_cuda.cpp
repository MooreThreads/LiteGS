#include <torch/extension.h>
#include "binning.h"
#include "compact.h"
#include "raster.h"
#include "transform.h"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("create_viewproj_forward", &create_viewproj_forward);
	m.def("create_viewproj_backward", &create_viewproj_backward);
	m.def("create_table", &create_table);
	m.def("tileRange", &tileRange);
	m.def("get_allocate_size", &get_allocate_size);
	m.def("rasterize_forward", &rasterize_forward);
	m.def("rasterize_forward_packed", &rasterize_forward_packed);
	m.def("rasterize_backward", &rasterize_backward);
	m.def("jacobianRayspace", &jacobianRayspace);
	m.def("createTransformMatrix_forward", &createTransformMatrix_forward);
	m.def("createTransformMatrix_backward", &createTransformMatrix_backward);
	m.def("world2ndc_forward", &world2ndc_forward);
	m.def("world2ndc_backword", &world2ndc_backword);
	m.def("createCov2dDirectly_forward", &createCov2dDirectly_forward);
	m.def("createCov2dDirectly_backward", &createCov2dDirectly_backward);
	m.def("sh2rgb_forward", &sh2rgb_forward);
	m.def("sh2rgb_backward", &sh2rgb_backward);
	m.def("eigh_and_inv_2x2matrix_forward", &eigh_and_inv_2x2matrix_forward);
	m.def("inv_2x2matrix_backward", &inv_2x2matrix_backward);
	m.def("cull_compact_activate", &cull_compact_activate);
	m.def("activate_backward", &activate_backward);
	m.def("adamUpdate", &adamUpdate);
	m.def("frustum_culling_aabb_cuda", &frustum_culling_aabb_cuda);
}