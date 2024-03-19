# A better implementation of Gaussian Splatting
Welcome to this repository, a refactored codebase aimed at enhancing the functionality and flexibility of the Gaussian splatting. 

## Background
Gaussian splatting is a powerful technique used in various computer graphics and vision applications. It involves mapping a set of points onto a grid using Gaussian distributions, allowing for efficient and accurate representation of spatial data. However, the original implementation of Gaussian splatting in PyTorch posed certain challenges: 

1. Both the forward and backward passes were encapsulated within a single PyTorch operator, making it difficult to access intermediate variables without delving into the C code.
2. Even if one attempted to modify the C code, they would have to derive the gradient formula manually and subsequently implement those formulas in the backward pass.

## Features

1. **Modular Design**: The refactored Gaussian splatting implementation separates the forward and backward into multiple PyTorch operators, allowing for greater modularity and accessibility. Besides, by leveraging PyTorch Autograd, the refactored codebase eliminates the need for deriving the gradient formula manually so that we can add loss term for intermediate variables easily.

2. **Fewer Atomic Operation**: Instead of using a lot of AtomicAdd in backward, using reduction in shared memory is much more efficient.

## Getting Started

### build and install submodules.
1. Install simple-knn

    `cd gaussian_splatting/submodules/simple-knn`

    `python setup.py build_ext --inplace -j8`

    `python setup.py build_ext --inplace -j8`

2. Install gaussian_raster

    `cd gaussian_splatting/submodules/gaussian_raster`

    `mkdir ./build`

    `cd ./build`

    `cmake -DCMAKE_PREFIX_PATH=@1/share/cmake ../` replace @1 with the installation path of your PyTorch, which is like "\$PYTHONHOME\$/Lib/site-packages/torch"

    `cmake --build . --config Release`

### train

`./train.py --random_background --sh_degree 0 -s dataset/garden -i images_4 -m output/garden`