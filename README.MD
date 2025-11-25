# LiteGS

**Fastest(about 50 seconds), Modular, and Available in Pure Python or CUDA**

This repository provides a refactored codebase aimed at improving the flexibility and performance of Gaussian splatting. 

## Training 3DGS in about 50 seconds

`python ./scripts/full_eval_fast.py --mipnerf360 SOURCE_PATH1 --tanksandtemples SOURCE_PATH2 --deepblending SOURCE_PATH3`

|scene|primitives|takes(RTX 3090)|takes(RTX 4090)|SSIM_train|PSNR_train|LPIPS_train|SSIM_test|PSNR_test|LPIPS_test|
|---|---|---|---|---|---|---|---|---|---|
|bicycle|1360000|68.90290451049805|**41.72691751**|0.7696665|23.9320354|0.2501733|0.7585854|25.1978111|0.2335179|
|flowers|1220000|78.57123351097107|**47.9060986**|0.7225419|23.1023922|0.2838508|0.6053527|21.7142277|0.3396027|
|garden|1460000|78.55829739570618|**46.76381636**|0.8845506|28.7087154|0.1193212|0.8556744|27.3950653|0.1323125|
|stump|1340000|71.96337366104126|**44.23456001**|0.8554742|28.5616417|0.2056226|0.7926696|27.2177505|0.2127895|
|treehill|1160000|74.03676867485046|**46.34997225**|0.7261154|22.7449684|0.3029707|0.6390569|22.8510647|0.3379321|
|room|800000|70.27747488|**41.05782461**|0.9351271|33.6474419|0.2011591|0.9221826|31.5905571|0.2149457|
|counter|800000|89.81700301170349|**51.63395095**|0.9220967|29.9026985|0.1850652|0.9086227|28.8188915|0.2000091|
|kitchen|1200000|105.24096488952637|**63.1258564**|0.9389035|32.4555969|0.1205899|0.9264742|31.4408417|0.1294753|
|bonsai|1200000|91.47094392776489|**56.54596996**|0.9501833|32.8006744|0.1922134|0.9444824|31.9543724|0.1964863|
|truck|680000|61.99935531616211|**40.03852582**|0.8910525|26.4909763|0.1484901|0.8747544|25.3993835|0.1531686|
|train|720000|69.77397298812866|**46.26339579**|0.8102806|23.553236|0.2373699|0.7795711|21.1193752|0.2538348|
|drjohnson|1600000|66.38695478439331|**39.18569803**|0.9353783|34.0604973|0.2259799|0.9080961|29.6174507|0.2518271|
|playroom|980000|56.96188426017761|**32.52357793**|0.9448828|35.7014847|0.2159965|0.9150408|30.9205952|0.2443539|

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
    pip install litegs/submodules/fussed_ssim
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

`./example_train.py --sh_degree 3 -s DATA_SOURCE -i IMAGE_FOLDER -m OUTPUT_PATH`

## Faster

The training results of LiteGS using the Mip-NeRF 360 dataset on an RTX 3090 are presented below. The training and evaluation command used is:

LiteGS-turbo:
`python ./scripts/full_eval_fast.py --mipnerf360 SOURCE_PATH1 --tanksandtemples SOURCE_PATH2 --deepblending SOURCE_PATH3`

LiteGS:
`python ./full_eval.py --mipnerf360 SOURCE_PATH1 --tanksandtemples SOURCE_PATH2 --deepblending SOURCE_PATH3`

![image](doc_img/results.png)

![image](doc_img/primitive_vs_psnr.png)



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
