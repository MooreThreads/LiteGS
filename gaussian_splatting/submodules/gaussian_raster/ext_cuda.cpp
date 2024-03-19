#include <torch/extension.h>
#include "raster_binning.h"


TORCH_LIBRARY(RasterBinning, m) {
  m.def("duplicateWithKeys", duplicateWithKeys);
  m.def("tileRange", tileRange);
  m.def("rasterize_forward", rasterize_forward);
  m.def("rasterize_backward", rasterize_backward);
}