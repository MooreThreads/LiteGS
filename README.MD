# LiteGS

**Fastest(about 50 seconds), Modular, and Available in Pure Python or CUDA**

This repository provides a refactored codebase aimed at improving the flexibility and performance of Gaussian splatting. 

## Training 3DGS in about 50 seconds

`python ./scripts/full_eval_aggressive.py --mipnerf360 SOURCE_PATH1 --tanksandtemples SOURCE_PATH2 --deepblending SOURCE_PATH3`

| Scene | Primitives | Time_3090(s) | Time_4090(s) | SSIM | PSNR | LPIPS |
|-------|------------|---------------|---------------|------|------|-------|
| bicycle | 1,000,000 | 50 | 33 | 0.7467 | 25.10 | 0.2531 |
| bonsai | 1,000,000 | 71 | 45 | 0.9430 | 32.02 | 0.1986 |
| counter | 1,000,000 | 84 | 51 | 0.9103 | 28.98 | 0.1952 |
| drjohnson | 1,000,000 | 47 | 28 | 0.9054 | 29.55 | 0.2596 |
| flowers | 1,000,000 | 60 | 40 | 0.6011 | 21.75 | 0.3500 |
| garden | 1,000,000 | 53 | 33 | 0.8471 | 27.23 | 0.1485 |
| kitchen | 1,000,000 | 82 | 50 | 0.9249 | 31.33 | 0.1321 |
| playroom | 1,000,000 | 48 | 29 | 0.9123 | 30.82 | 0.2434 |
| room | 1,000,000 | 64 | 39 | 0.9220 | 31.63 | 0.2130 |
| stump | 1,000,000 | 53 | 34 | 0.7847 | 27.10 | 0.2278 |
| train | 1,000,000 | 70 | 47 | 0.7956 | 21.50 | 0.2316 |
| treehill | 1,000,000 | 57 | 37 | 0.6382 | 23.17 | 0.3479 |
| truck | 1,000,000 | 63 | 43 | 0.8831 | 25.78 | 0.1368 |

| Dataset | Primitives | Time_3090(s) | Time_4090(s) | SSIM | PSNR | LPIPS |
|-------|------------|---------------|---------------|------|------|-------|
| mipnerf360 | 1,000,000 | 64 | 40 | 0.8131 | 27.59 | 0.2296 |
| tat | 1,000,000 | 67 | 45 | 0.8394 | 23.64 | 0.1842 |
| db | 1,000,000 | 47 | 29 | 0.9088 | 30.19 | 0.2515 |

## Background
Gaussian splatting is a powerful technique used in various computer graphics and vision applications. It involves representing 3D data as Gaussian distributions in space, allowing for efficient and accurate representation of spatial data. However, the original implementation (https://github.com/graphdeco-inria/gaussian-splatting) of Gaussian splatting in PyTorch faced several limitations: 

1. The forward and backward computations were encapsulated in two distinct PyTorch extension functions. Although this design significantly accelerated training, it restricted access to intermediate variables unless the underlying C code was modified.
2. Modifying any step of the algorithm required manually deriving gradient formulas and implementing them in the backward pass, adding considerable complexity.

## Features

1. **Modular Design**: The refactored codebase breaks forward and backward into multiple PyTorch extension functions, significantly improving modularity and enabling easier access to intermediate variables. Additionally, in some cases, leveraging PyTorch Autograd eliminates the need to manually derive gradient formulas.

2. **Flexible**: LiteGS provides two modular APIs—one implemented in CUDA and the other in Python. The Python-based API facilitates straightforward modifications to calculation logic without requiring expertise in C code, enabling rapid prototyping. Additionally, tensor dimensions are permuted to maintain competitive training speeds for the Python API. For performance-critical tasks, the CUDA-based API is fully customizable.

3. **Better Performance and Fewer Resources**: LiteGS achieves an 4.7x speed improvement over the original 3DGS implementation while reducing GPU memory usage by around 30%. These optimizations enhance training efficiency without compromising flexibility or readability.

4. **Algorithm Preservation**: LiteGS retains the core 3DGS algorithm, making only minor adjustments to the training logic due to culstering.


## Getting Started

1. Install simple-knn

    ```bash
    pip install litegs/submodules/simple-knn
    ```

2. Install fused-ssim

    ```bash
    pip install litegs/submodules/fused_ssim
    ```
    

3. Install litegs_fused
    
    ```bash
    pip install litegs/submodules/gaussian_raster
    ```

    If you need the cmake project(e.g. CUDA Debug in Visual Studio)：

    ```bash
    cd litegs/submodules/gaussian_raster
    mkdir ./build
    cd ./build
    #for Windows PowerShell: $env:CMAKE_PREFIX_PATH = (python -c "import torch; print(torch.utils.cmake_prefix_path)")
    export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
    cmake ../
    cmake --build . --config Release
    ```

4. Install requirments

    ```bash
    pip install -r requirement.txt
    ```
    
### Train
Begin training with the following command:

```bash
python example_train.py --model.sh-degree 3 -s DATA_SOURCE -i IMAGE_FOLDER -m OUTPUT_PATH --dataset.eval --densify.target-primitives 1000000
```

**Resume from checkpoint:**

```bash
python example_train.py --config-path OUTPUT_PATH/config.json --resume OUTPUT_PATH/checkpoint.pth
```

### Evaluation

```bash
python example_metrics.py --config-path OUTPUT_PATH/finish/eval_config.json --save-image
```

## Faster

The training results of LiteGS using the Mip-NeRF 360 dataset on an RTX 3090 are presented below. The training and evaluation command used is:

LiteGS:
`python ./full_eval.py --mipnerf360 SOURCE_PATH1 --tanksandtemples SOURCE_PATH2 --deepblending SOURCE_PATH3`

![image](doc_img/LiteGS_ParamScale.png)



## Modular

Unlike the original 3DGS, which encapsulates nearly the entire rendering process into a single PyTorch extension function, LiteGS divides the process into multiple modular functions. This design allows users to access intermediate variables and integrate custom computation logic using Python scripts, eliminating the need to modify C code. The rendering process in LiteGS is broken down into the following steps:

1. Cluster Culling
    
    LiteGS divides the Gaussian points into several chunks, with each chunk containing 1,024 points. The first step in the rendering pipeline is frustum culling, where points outside the camera's view are filtered out.

2. Cluster Compact
    
    Similar to mesh rendering, LiteGS compacts visible primitives after frustum culling. Each property of the visible points is reorganized into sequential memory to improve processing efficiency.

3. 3DGS Projection
    
    Gaussian points are projected into screen space in this step, with no modifications made compared to the original 3DGS implementation.

4. Create Visibility Table
    
    A visibility table is created in this step, mapping tiles to their visible primitives, enabling efficient parallel processing in subsequent stages.

5. Rasterization
    
    In the final step, each tile rasterizes its visible primitives in parallel, ensuring high computational efficiency.

LiteGS makes slight adjustments to density control to accommodate its clustering-based approach. 

## Flexible

The gaussian_splatting/wrapper.py file contains two sets of APIs, offering flexibility in choosing between Python-based and CUDA-based implementations. The Python-based API is invoked using call_script(), while the CUDA-based API is available via call_fused(). While the CUDA-based API delivers significant performance improvements, it lacks the flexibility. The choice between these implementations depends on the specific use case:

* python-based api: Provides greater flexibility, making it ideal for rapid prototyping and development where training speed is less critical.

* cuda-based api: Offers the highest performance and is recommended for production environments where training speed is a priority.

Additionally, an interface validate() and the accompanying check_wrapper.py script are provided to verify that both APIs produce consistent gradients.

Here is an example that demonstrates the flexibility of LiteGS. In this instance, our goal is to create a more precise bounding box for a 2D Gaussian when generating visibility tables. In the original 3DGS implementation, the bounding box is determined as three times the length of the major axis of the Gaussian. However, incorporating opacity can allow for a smaller bounding box.

To implement this change in the original 3DGS, the following steps are required:
* Modify the C++ function declarations and definitions
* Update the CUDA global function
* Recompile

In LiteGS, the same change can be achieved by simply editing a Python script.

original:
```python
axis_length=(3.0*eigen_val.abs()).sqrt().ceil()
```
modified:
```python
coefficient=2*((255*opacity).log())
axis_length=(coefficient*eigen_val.abs()).sqrt().ceil()
```

## Performance Optimization Detail

Comming soon.
