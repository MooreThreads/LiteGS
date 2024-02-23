#include <torch/extension.h>
#include "raster_binning.h"


TORCH_LIBRARY(RasterBinning, m) {
  m.def("duplicateWithKeys", duplicateWithKeys);
  m.def("tileRange", tileRange);
  m.def("rasterize_forward_gathered", rasterize_gathered_forward);
  m.def("rasterize_backward_gathered", rasterize_gathered_backward);
  m.def("rasterize_forward", rasterize_forward);
  m.def("rasterize_backward", rasterize_backward);
}
