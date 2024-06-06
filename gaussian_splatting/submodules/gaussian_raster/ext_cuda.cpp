#include <torch/extension.h>
#include "raster_binning.h"


TORCH_LIBRARY(RasterBinning, m) {
  m.def("duplicateWithKeys", duplicateWithKeys);
  m.def("tileRange", tileRange);
  m.def("rasterize_forward", rasterize_forward);
  m.def("rasterize_backward", rasterize_backward);
  m.def("jacobianRayspace", jacobianRayspace);
  m.def("createTransformMatrix_forward", createTransformMatrix_forward);
  m.def("createTransformMatrix_backward", createTransformMatrix_backward);
  m.def("world2ndc_backword", world2ndc_backword);
  m.def("createCov2dDirectly_forward", createCov2dDirectly_forward);
  m.def("createCov2dDirectly_backward", createCov2dDirectly_backward);
}
