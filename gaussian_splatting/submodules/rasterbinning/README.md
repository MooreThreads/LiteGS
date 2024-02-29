# RasterBinning: PyTorch CUDA Extension

This repository contains a RasterBinning implementation as a PyTorch CUDA extension. 
It is used by our training code.

## Installation
1. Build Dynamic Library

    `mkdir ./rasterbinning/build`

    `cd ./rasterbinning/build`
    
    `cmake -DCMAKE_PREFIX_PATH=@1\share\cmake ../` replace @1 with the installation path of your PyTorch, which is like "\$PYTHONHOME\$/lib/site-packages/torch"

    `cmake --build . --config Release`

2. Load Dynamic Library in Code

    modify the path in REPO_ROOT/gaussian_splatting/model.py LINE 10.