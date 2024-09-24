# A better implementation of Gaussian Splatting
Welcome to this repository, a refactored codebase aimed at enhancing the  flexibility and performance of the Gaussian splatting.

## Background
Gaussian splatting is a powerful technique used in various computer graphics and vision applications. It involves mapping a set of points onto a grid using Gaussian distributions, allowing for efficient and accurate representation of spatial data. However, the original implementation of Gaussian splatting in PyTorch posed certain challenges: 

1. Both the forward and backward passes were encapsulated within a single PyTorch operator, making it difficult to access intermediate variables without delving into the C code.
2. Even if one attempted to modify the C code, they would have to derive the gradient formula manually and subsequently implement those formulas in the backward pass.

## Features

1. **Modular Design**: The refactored Gaussian splatting implementation separates the forward and backward into multiple PyTorch operators, allowing for greater modularity and accessibility. Besides, by leveraging PyTorch Autograd, the refactored codebase eliminates the need for deriving the gradient formula manually so that we can add loss term for intermediate variables easily.

2. **Flexible**: PytorchGS supply two sets of modular API. One is implement by cuda, another is pruely implement by python scripts. You can change the caculate logit simply in python scripts API without modifing C code and rapidly implement your idea. If you are concerned about performance, you can create or modify the cuda API. Besides, a unit test framework is implemented in PytorchGS in order to check these two API getting the same result.

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

|           |Bicycle(Ours)  |Bicycle(Official)  |
|   ----    | ----          | ---               |
| #Point    | 5.3M          |                   |
| #Iter     | 33800         | 30000             |
| Take Times| 24:06         |                   |
|PSNR(Train)| 26.33         |                   |
|PSNR(Test) | 25.50         | 25.24             |

|           |Flowers(Ours)  |Flowers(Official)  |
|   ----    | ----          | ---               |
| #Point    | 3.4M          |                   |
| #Iter     | 30200         | 30000             |
| Take Times| 16:50         |                   |
|PSNR(Train)| 24.47         |                   |
|PSNR(Test) | 22.07         | 21.52             |

|           |Garden(Ours)   |Garden(Official)   |
|   ----    | ----          | ---               |
| #Point    | 4.5M          |                   |
| #Iter     | 32200         | 30000             |
| Take Times| 22:06         | 31:34             |
|PSNR(Train)| 30.12         | 29.86             |
|PSNR(Test) | 27.58         | 27.42             |

|           | Stump(Ours)   | Stump(Official)   |
|   ----    | ----          | ---               |
| #Point    | 5.1M          |                   |
| #Iter     | 21800         | 30000             |
| Take Times| 14:40         |                   |
|PSNR(Train)| 30.85         |                   |
|PSNR(Test) | 27.18         | 26.55             |

|           | Treehill(Ours)| Treehill(Official)|
|   ----    | ----          | ---               |
| #Point    | 4.2M          |                   |
| #Iter     | 24600         | 30000             |
| Take Times| 15:16         |                   |
|PSNR(Train)| 24.59         |                   |
|PSNR(Test) | 22.87         | 22.49             |

Room

140EPOCH-38352iter   35.02 31.70 16:44

200EPOCH-54400iter   35.45 31.67 22:38

Counter

140EPOCH-29610iter  30.60 29.21 13:00

200EPOCH-42000iter  30.90 29.40 17:42

Kitchen

140EPOCH-34404iter 33.87 31.67 16:12

200EPOCH-48800iter 34.18 31.75 21:55

Bonsai

140EPOCH-35955iter 34.33 33.31 15:41

200EPOCH-51000iter 34.85 33.61 21:44



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