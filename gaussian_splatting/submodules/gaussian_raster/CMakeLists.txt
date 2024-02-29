cmake_minimum_required (VERSION 3.9)

project(gaussian_splatting_pytorch_lib LANGUAGES CXX CUDA)

set( CMAKE_CXX_STANDARD 17)
set( CMAKE_CUDA_STANDARD 17 )
set( CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

include_directories (${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set( INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" )


add_library(GaussianRaster SHARED
  ext_cuda.cpp
  raster_binning.cu
)

target_include_directories(GaussianRaster PRIVATE "${INCLUDE_DIR}")

target_link_libraries(GaussianRaster PRIVATE ${TORCH_LIBRARIES} Python::Python)

#cmake -DCMAKE_PREFIX_PATH="C:\Users\admin\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\share\cmake" ..