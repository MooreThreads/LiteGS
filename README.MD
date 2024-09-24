# A better implementation of Gaussian Splatting
Welcome to this repository, a refactored codebase aimed at enhancing the  flexibility and performance of the Gaussian splatting.

## Background
Gaussian splatting is a powerful technique used in various computer graphics and vision applications. It involves mapping a set of points onto a grid using Gaussian distributions, allowing for efficient and accurate representation of spatial data. However, the original implementation of Gaussian splatting in PyTorch posed certain challenges: 

1. Both the forward and backward passes were encapsulated within a single PyTorch operator, making it difficult to access intermediate variables without delving into the C code.
2. Even if one attempted to modify the C code, they would have to derive the gradient formula manually and subsequently implement those formulas in the backward pass.

## Features

1. **Modular Design**: The refactored Gaussian splatting implementation separates the forward and backward into multiple PyTorch operators, allowing for greater modularity and accessibility. Besides, by leveraging PyTorch Autograd, the refactored codebase eliminates the need for deriving the gradient formula manually so that we can add loss term for intermediate variables easily.

2. **Flexible**: PytorchGS supply two sets of modular API. One is implement by cuda, another is pruely implement by python scripts. You can change the caculate logit simply in python scripts API without modifing C code and rapidly implement your idea. If you are concerned about performance, you can create or modify the cuda API. ~~Besides, a unit test framework is implemented in PytorchGS in order to check these two API getting the same result.~~

3. **Better Performance Fewer Resources**: The PytorchGS is not only more flexible and readable than the official code but also much more faster. We do a lot of work to improve the performance of training and get **50%** acceleration. Besides, PytorchGS takes less gpu memory.

4. **No New**: We do not change the GS algorithm. Only a few training logic is changed due to we create the cluster of gaussian points.


## Getting Started

### build and install submodules.
1. Install simple-knn

    `cd gaussian_splatting/submodules/simple-knn`

    `python setup.py build_ext --inplace -j8`

    `python setup.py install`

2. Install gaussian_raster

    `cd gaussian_splatting/submodules/gaussian_raster`

    `mkdir ./build`

    `cd ./build`

    `cmake -DCMAKE_PREFIX_PATH=@1/share/cmake ../` replace @1 with the installation path of your PyTorch, which is like "\$PYTHONHOME\$/Lib/site-packages/torch"

    `cmake --build . --config Release`

### train

`./train.py --random_background --sh_degree 0 -s dataset/garden -i images_4 -m output/garden`

## Result

|            |Bicycle  |flowers  |garden   |stump    |treehill |room     |counter  |kitchen  |bonsai   |
|   ----     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     |
| Take Times | 19:46   | 15:41   | 19:34   | 19:19   | 17:36   | 11:26   | 11:50   | 12:44   | 12:17   |
|SSIM(Train) | 0.8391  | 0.751   | 0.905   | 0.907   | 0.796   | 0.935   | 0.921   | 0.936   | 0.947   |
|PSNR(Train) | 26.213  | 24.384  | 30.076  | 31.413  | 24.876  | 35.021  | 30.641  | 33.710  | 34.508  |
|LPIPS(Train)| 0.185   | 0.241   | 0.090   | 0.151   | 0.231   | 0.170   | 0.162   | 0.101   | 0.161   |
|SSIM(Test)  | 0.764   | 0.612   | 0.858   | 0.787   | 0.637   | 0.919   | 0.906   | 0.920   | 0.940   |
|PSNR(Test)  | 25.454  | 22.038  | 27.622  | 27.174  | 22.768  | 32.187  | 29.307  | 32.100  | 33.225  |
|LPIPS(Test) | 0.218   | 0.335   | 0.112   | 0.218   | 0.318   | 0.186   | 0.179   | 0.113   | 0.166   |



## Modular

cluster culling

cluster compact

3DGS projection

Visibility Table

Rasterization

## PythonScript or CudaExtension
...


## Performance Optimization Log

...

Instead of using a lot of AtomicAdd in backward, using reduction in shared memory is much more efficient.