#include <torch/extension.h>
#include "raster_binning.h"


TORCH_LIBRARY(RasterBinning, m) {
  m.def("duplicateWithKeys", duplicateWithKeys);
  m.def("tileRange", tileRange);
}
