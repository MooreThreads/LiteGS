# LiteGS

**3.5x Acceleration, Modular, and Available in Pure Python or CUDA**

This repository provides a refactored codebase aimed at improving the flexibility and performance of Gaussian splatting. 

## Update
一个全新的重构版本已经发布，避免了大量碎片化的接口，并支持pip安装。部分主流的3DGS新特性将在1-2周内放出，敬请期待。

## Background
Gaussian splatting is a powerful technique used in various computer graphics and vision applications. It involves representing 3D data as Gaussian distributions in space, allowing for efficient and accurate representation of spatial data. However, the original implementation (https://github.com/graphdeco-inria/gaussian-splatting) of Gaussian splatting in PyTorch faced several limitations: 

1. The forward and backward computations were encapsulated in two distinct PyTorch extension functions. Although this design significantly accelerated training, it restricted access to intermediate variables unless the underlying C code was modified.
2. Modifying any step of the algorithm required manually deriving gradient formulas and implementing them in the backward pass, adding considerable complexity.

## Features

1. **Modular Design**: The refactored codebase breaks forward and backward into multiple PyTorch extension functions, significantly improving modularity and enabling easier access to intermediate variables. Additionally, in some cases, leveraging PyTorch Autograd eliminates the need to manually derive gradient formulas.

2. **Flexible**: LiteGS provides two modular APIs—one implemented in CUDA and the other in Python. The Python-based API facilitates straightforward modifications to calculation logic without requiring expertise in C code, enabling rapid prototyping. Additionally, tensor dimensions are permuted to maintain competitive training speeds for the Python API. For performance-critical tasks, the CUDA-based API is fully customizable.

3. **Better Performance and Fewer Resources**: LiteGS achieves an 3.5x speed improvement over the original 3DGS implementation while reducing GPU memory usage by around 30%. These optimizations enhance training efficiency without compromising flexibility or readability.

4. **Algorithm Preservation**: LiteGS retains the core 3DGS algorithm, making only minor adjustments to the training logic due to culstering.


## Getting Started

### Install from pip

安装包已上传至Pypi，您可以通过以下指令完成安装
`
pip install lite-gaussian-splatting
`

### Or Clone&Build

有编辑需求的用户，我们更推荐您采用clone&build的方式完成安装，这样可以为核心模块构建cmake工程以便调试:

1. Install simple-knn

    `cd gaussian_splatting/submodules/simple-knn`

    `python install .`

2. Install fused-ssim

    `cd gaussian_splatting/submodules/fussed_ssim`

    `python intsall .`
    

3. Install litegs_fused
    `cd gaussian_splatting/submodules/gaussian_raster`

    `python install .`

    在开发中，为了方便调试，您可能需要cmake project。你可以通过以下命令使用cmake构建litegs_fused模块：

    `cd gaussian_splatting/submodules/gaussian_raster`

    `mkdir ./build`

    `cd ./build`

    Windows Powershell:
    ` $env:CMAKE_PREFIX_PATH = (python -c "import torch; print(torch.utils.cmake_prefix_path)") `
    
    Linux:
    ` export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") `

    `cmake ../`

    `cmake --build . --config Release`

    Begin training with the following command:

    `./example_train.py --random_background --sh_degree 0 -s dataset/garden -i images_4 -m output/garden`

## Faster

The performance metrics are heavily dependent on the COLMAP results. we use the same processed data as RetinaGS( https://github.com/MooreThreads/RetinaGS ).

The training results of LiteGS using the Mip-NeRF 360 dataset on an NVIDIA A100 GPU are presented below. The training and evaluation command used is:

`python ./full_eval.py --mipnerf360 SOURCE_PATH`

|metric\|scene  |Bicycle  |flowers  |garden   |stump    |treehill |room     |counter  |kitchen  |bonsai   |
|   ----        | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     |
|SSIM(Test)     | 0.767   | 0.615   | 0.864   | 0.793   | 0.649   | 0.932   | 0.916   | 0.934   | 0.952   |
|PSNR(Test)     | 25.42   | 22.08   | 27.61   | 27.14   | 22.91   | 32.17   | 29.24   | 32.04   | 33.19   |
|LPIPS(Test)    | 0.223   | 0.338   | 0.115   | 0.218   | 0.323   | 0.189   | 0.183   | 0.115   | 0.168   |


For comparison, the metrics of the original 3DGS implementation are as follows. These results demonstrate that the refactored LiteGS code maintains accuracy while significantly improving performance.

|metric\|scene  |Bicycle  |flowers  |garden   |stump    |treehill |room     |counter  |kitchen  |bonsai   |
|   ----        | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     |
|SSIM(Test)     | 0.770   | 0.623   | 0.866   | 0.771   | 0.641   | 0.931   | 0.919   | 0.933   | 0.950   |
|PSNR(Test)     | 25.33   | 21.85   | 27.58   | 26.74   | 22.46   | 31.93   | 29.54   | 31.52   | 33.10   |
|LPIPS(Test)    | 0.200   | 0.320   | 0.107   | 0.223   | 0.317   | 0.220   | 0.204   | 0.129   | 0.205   |


Below is a summary of the training times for LiteGS and the original 3DGS implementations on NVIDIA A100 and NVIDIA RTX3090 GPUs. In the current setup, the original 3DGS implementation includes an integrated acceleration scheme from Taming-GS that can be enabled via training parameters. For comparison purposes, we have also included this TamingGS-integrated version in our evaluation.

Our experimental results demonstrate that even with kernel splitting to achieve modularity, LiteGS still delivers remarkable performance improvements. Specifically, compared to the original 3DGS implementation, LiteGS achieves a 3.5x speedup on RTX3090. Moreover, when compared to the TamingGS-integrated version of 3DGS, LiteGS still offers a 1.5x speedup. These results clearly underscore the effectiveness of our approach in enhancing training efficiency while maintaining high performance.

|   |Bicycle  |flowers  |garden   |stump    |treehill |room     |counter  |kitchen  |bonsai   |
|   ----     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     |
| Original-d9fad7b(A100)    | 32:09   | 24:37   | 31:34   | 25:58   | 24:08   | 21:59   | 21:45   | 25:41   | 19:10   |
| Taming-Intergrated(A100)  | 13:29   | 10:53   | 13:02   | 10:48   | 10:19   | 10:04   | 10:11   | 14:16   | 13:48   |
| LiteGS(A100)              | 08:04   | 07:32   | 08:09   | 07:39   | 07:40   | 07:24   | 07:17   | 07:32   | 07:07   |
| Original-d9fad7b(RTX3090) | 40:06   | 28:46   | 42:18   | 32:40   | 29:39   | 24:53   | 24:16   | 29:30   | 20:15   |
|Taming-Intergrated(RTX3090)| 15:29   | 11:42   | 16:24   | 11:49   | 11:48   | 10:06   | 12:00   | 16:38   | 09:01   |
| LiteGS(RTX3090)           | 09:59   | 08:23   | 10:46   | 09:03   | 09:06   | 06:33   | 07:26   | 08:32   | 06:35   |

![image](doc_img/takes_time.png)



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

## Citation

If you find this project useful in your research, please consider cite:

```
@misc{LiteGS,
    title={LiteGS},
    author={LiteGS Contributors},
    howpublished = {\url{https://github.com/MooreThreads/LiteGS}},
    year={2024}
}
```

## Performance Optimization

This section outlines the key optimizations implemented at each stage to enhance performance.

### high level

At a high level, **tensor dimensions are permuted** from [points,property] to [property,points]. This change ensures that tensor computations implemented in pure Python load and store data sequentially, improving memory access efficiency.

### DensityControl

To maintain spatial locality during density control, we now employ Morton code partitioning. Unlike the previous approach where new points were simply appended to the end of tensors—resulting in reduced spatial locality—our updated method leverages Morton codes to map 3D points into a one-dimensional space while preserving their spatial proximity. This reordering ensures that points which are close in the 3D space remain adjacent in memory, thereby improving memory access efficiency and overall computational performance.

### Cull and Compact

The original implementation performed direct culling of primitives, which could become time-consuming in large scenes. We optimized this by performing **frustum culling on cluster AABB**(Axis-Aligned Bounding Box) instead, significantly reducing computational overhead.

Additionally, we implemented **compaction** of Gaussian point parameters to improve GPU utilization. While the original implementation skipped computations for culled Gaussian points, it left the GPU scheduler underutilized due to warp-based execution (32 threads). By compacting Gaussian point parameters, GPU resource utilization is improved.

It is important to note that compaction should be applied to Gaussian point clusters rather than individual primitives. Applying it at the primitive level can slow down the backward pass due to out-of-order memory writes.

### Sparse Gradients and Fused Optimizer

Another closely related optimization is the use of sparse gradients. After visibility culling, the gradient back-propagated to each parameter can be represented as a sparse tensor whose indices are defined by visible primitives only. In higher-order spherical-harmonics (3DGS) training, utilizing sparse tensors can provide significant performance benefits.

Furthermore, conventional optimizers may not be optimal for 3DGS training. For example, the default Adam implementation often relies on a multitensor-based algorithm (invoked via multiple multitensor_add, multitensor_mul, multitensor_div steps in Python), which is generally tailored for large numbers of small tensors in typical neural networks. By contrast, 3DGS involves fewer—but often larger—tensors. Specifying the fused parameter triggers a kernel-based Adam implementation for each tensor, which can improve efficiency in 3DGS workflows.

### Projection

The projection step remains unchanged from the original implementation. We highly recommend using the cuda implemented function instead of the python implemented function. The essential different between these two functions is whether we write the intermediate data into the gpu global memory. The projection step involves several consecutive 4x4 matrix multiplications. Writing intermediate data to GPU global memory significantly increases data transfer overhead.

### Create Visibility Table

Same as the "binning" step in compute raster, the purpose of this step is to map the Gaussian points and tiles. The critical optimization of this step is creating a more accurate bounding box for each primitive. An accurate visibility table will reduce computation in following step. Original 3DGS use the major axis of 2D Gaussian to create a bounding square which is a rough approximation. **Taking the alpha and minor axis into consideration**, LiteGS generates a more accurate bounding box for the 2D Gaussian.

There is a trade-off between accuracy and speed. The bounding box composed of major axis and minor axis need 2 pass to create the visibility table. One pass to allocate memory size and another to fill the table. 2D AABB is easy to get the allocate size so it only need one pass. LiteGS opts to use the 2D AABB for efficiency.

### Rasterization

The original implementation of rasterization contains a significant efficiency issue. During the backward of rasterization, gradients of properties are computed for each pixel, and these gradients need to be summed. The original implementation uses AtomicAdd directly on global memory, which is inefficient for large-scale computations.

A more efficient approach involves using a reduction algorithm. Below is the pseudo-code for such an implementation:

```
for(int i=element_num;i > 16;i/=2)
{
    if(threadidx.x < i)
    {
        property0[threadidx.x]+=property0[threadidx.x+i];
        property1[threadidx.x]+=property1[threadidx.x+i];
        property2[threadidx.x]+=property2[threadidx.x+i];
        ...
        __syncthreads();
    }
}  
float grad0=property0[threadidx.x];
float grad1=property1[threadidx.x];
...
for(int i=16;i > 0;i/=2)
{
    grad0 += __shfl_down_sync(0xffffffff, grad0, i);
    grad1 += __shfl_down_sync(0xffffffff, grad1, i);
    ...
} 
```

The backward pass for rasterizing a 2D Gaussian plane is one of the most computationally expensive steps in the training process, requiring careful optimization.

#### Warp-level Reduction
In some scenarios, it can be more efficient to bypass shared-memory reduction and directly use warp-level reduction followed by a global AtomicAdd. This approach works particularly well on devices that have sufficient L2 atomic throughput, making shared-memory accumulation unnecessary. Although NVIDIA’s official documentation does not specify the exact number of “L2 Atomic Pipelines,” numerous microbenchmark studies indicate that L2 caches are physically partitioned per memory controller, and each partition provides a dedicated atomic path. Because different GPU models have varying numbers of memory partitions, the ratio of L2 partitions to compute cores ultimately determines whether warp-level reduction plus a global atomic add will sustain high performance or become a bottleneck.

In summary, warp-level reduction is a straightforward, efficient implementation that can be optimal for certain hardware configurations.

#### Multibatch Reduction
For reductions in shared memory, further optimizations are necessary. Each pixel typically involves 9 floating-point gradients, meaning the reduction process must handle multiple batches (multibatch reduction) instead of a simple one. Assigning threads to batches instead of relying on for-loops can significantly accelerate the reduction. Below is an optimized implementation:
```
for (int property_id = threadidx / 32; property_id < property_num ; property_id+= wraps_num_in_block)
{
    float gradient_sum = 0;
    for (int offset = threadid_in_warp; offset < tilesize * tilesize; reduction_i+=32)
    {
        gradient_sum += gradient_buffer[property_id * tilesize * tilesize + offset];
    }
    for (int offset = 16; offset > 0; offset /= 2)
    {
        gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, offset);
    }
    if (threadid_in_warp == 0)
    {
        shared_gradient_sum[property_id] = gradient_sum;
    }
}
```

Empty fragments are a notable problem in this process. Although the primitives visible to currrent tile are listed, they are often too small to fully occupy entire tile, creating empty fragments. Using smaller tiles (e.g., reducing the size from 16x16 to 8x8) can mitigate this issue but increases the task size in step 4: Create Visibility Table. Based on our tests, reducing tile size from 16x16 to 8x8 improves performance.

Compacting the gradient in shared memory and skipping the empty fragments in gradient summation is a further solution. By compacting valid fragments and storing their gradients in shared memory sequentially, followed by a block synchronization step, the gradients can be efficiently summed. Below is the implementation:
```
__shared__ float gradient[9][tilesize*tilesize];
__shared__ int valid_pix_num;
for(point:points)
{
    float alpha=CalcAlpha(threadIdx.xy,point);
    if(alpha>1.0f/255)
    {
        int index=AtomicAdd(&valid_pix_num);
        GradientPack grad_pack=CalcGradient(point);
        gradient[0][index]=grad_pack.ndc[0];
        gradient[1][index]=grad_pack.ndc[1];
        gradient[2][index]=grad_pack.cov_inv[0][0];
        gradient[3][index]=grad_pack.cov_inv[0][1];
        gradient[4][index]=grad_pack.cov_inv[1][1];
        gradient[5][index]=grad_pack.rgb[0];
        gradient[6][index]=grad_pack.rgb[1];
        gradient[7][index]=grad_pack.rgb[2];
        gradient[8][index]=grad_pack.alpha;
    }
    __syncthreads();
    MultibatchSum(gradient);
}
```
