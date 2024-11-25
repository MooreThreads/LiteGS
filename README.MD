# LiteGS

**80% Faster, Modular, and Available in Pure Python or CUDA**

This repository provides a refactored codebase aimed at improving the flexibility and performance of Gaussian splatting.

## Background
Gaussian splatting is a powerful technique used in various computer graphics and vision applications. It involves representing 3D data as Gaussian distributions in space, allowing for efficient and accurate representation of spatial data. However, the original implementation (https://github.com/graphdeco-inria/gaussian-splatting) of Gaussian splatting in PyTorch faced several limitations: 

1. The forward and backward computations were encapsulated in two distinct PyTorch extension functions. Although this design significantly accelerated training, it restricted access to intermediate variables unless the underlying C code was modified.
2. Modifying any step of the algorithm required manually deriving gradient formulas and implementing them in the backward pass, adding considerable complexity.

## Features

1. **Modular Design**: The refactored codebase breaks forward and backward into multiple PyTorch extension functions, significantly improving modularity and enabling easier access to intermediate variables. Additionally, in some cases, leveraging PyTorch Autograd eliminates the need to manually derive gradient formulas.

2. **Flexible**: LiteGS provides two modular APIs—one implemented in CUDA and the other in Python. The Python-based API facilitates straightforward modifications to calculation logic without requiring expertise in C code, enabling rapid prototyping. Additionally, tensor dimensions are permuted to maintain competitive training speeds for the Python API. For performance-critical tasks, the CUDA-based API is fully customizable.

3. **Better Performance and Fewer Resources**: LiteGS achieves an 80% speed improvement over the original 3DGS implementation while reducing GPU memory usage. These optimizations enhance training efficiency without compromising flexibility or readability.

4. **Algorithm Preservation**: LiteGS retains the core 3DGS algorithm, making only minor adjustments to the training logic due to BVH creation.


## Getting Started

### build and install submodules.

To get started, you’ll need to install the required submodules:

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

Begin training with the following command:

`./train.py --random_background --sh_degree 0 -s dataset/garden -i images_4 -m output/garden`

## Faster

The performance metrics are heavily dependent on the COLMAP results. we use the same processed data as RetinaGS( https://github.com/MooreThreads/RetinaGS ).

The training results of LiteGS using the Mip-NeRF 360 dataset on an NVIDIA A100 GPU are presented below. The training and evaluation command used is:

`python ./full_eval.py --mipnerf360 SOURCE_PATH`

|  metric\|scene    |Bicycle  |flowers  |garden   |stump    |treehill |room     |counter  |kitchen  |bonsai   |
|   ----     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     |
|SSIM(Test)  | 0.764   | 0.612   | 0.858   | 0.787   | 0.637   | 0.919   | 0.906   | 0.920   | 0.940   |
|PSNR(Test)  | 25.454  | 22.038  | 27.622  | 27.174  | 22.768  | 32.027  | 29.307  | 32.100  | 33.073  |
|LPIPS(Test) | 0.218   | 0.335   | 0.112   | 0.218   | 0.318   | 0.186   | 0.179   | 0.113   | 0.166   |


For comparison, the metrics of the original 3DGS implementation are as follows. These results demonstrate that the refactored LiteGS code maintains accuracy while significantly improving performance.

|metric\|scene  |Bicycle  |flowers  |garden   |stump    |treehill |room     |counter  |kitchen  |bonsai   |
|   ----     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     |
|SSIM(Test)  | 0.770   | 0.623   | 0.866   | 0.771   | 0.641   | 0.931   | 0.919   | 0.933   | 0.950   |
|PSNR(Test)  | 25.33  | 21.85  | 27.58  | 26.74  | 22.46  | 31.93  | 29.54  | 31.52  | 33.10  |
|LPIPS(Test) | 0.200   | 0.320   | 0.107   | 0.223   | 0.317   | 0.220   | 0.204   | 0.129   | 0.205   |


The training times for LiteGS and the original 3DGS implementation on an NVIDIA A100 GPU are summarized below, illustrating an approximately **80%** speedup achieved by LiteGS.

| repo\|scene  |Bicycle  |flowers  |garden   |stump    |treehill |room     |counter  |kitchen  |bonsai   |
|   ----     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     | ---     |
| Original(d9fad7b) | 32:09   | 24:37   | 31:34   | 25:58   | 24:08   | 21:59   | 21:45   | 25:41   | 19:10   |
| LiteGS | 17:19   | 13:45   | 17:24   | 16:56   | 15:26   | 9:31   | 9:40   | 10:34   | 10:02   |

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

LiteGS makes slight adjustments to density control to accommodate its clustering-based approach. Points are cloned and split at each epoch, then reorganized into new chunks without the need to rebuild the BVH. Every 10 epochs, point opacity is reset, and invalid points are removed, followed by a BVH rebuild. Compared to the original 3DGS, LiteGS resets opacity more frequently and introduces an opacity decay mechanism instead of resetting it to near zero. These minor changes maintain compatibility with LiteGS's clustering-based approach.

## Flexible

The gaussian_splatting/wrapper.py file contains two sets of APIs, offering flexibility in choosing between Python-based and CUDA-based implementations. Functions with the suffix “v1” represent the Python-based API, while those with the suffix “v2” correspond to the CUDA-based API. While the CUDA-based API delivers significant performance improvements, it lacks the flexibility. The choice between these implementations depends on the specific use case:

* python-based api: Provides greater flexibility, making it ideal for rapid prototyping and development where training speed is less critical.

* cuda-based api: Offers the highest performance and is recommended for production environments where training speed is a priority.

Switching between the two implementations is straightforward: simply modify the gaussian_splatting/wrapper.py file.

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

To maintain spatial locality during density control, we implemented **a simple BVH** (Bounding Volume Hierarchy). Unlike the original implementation, where new points are appended to the end of tensors—resulting in reduced spatial locality—our approach organizes points to preserve spatial coherence.

### Cull and Compact

The original implementation performed direct culling of primitives, which could become time-consuming in large scenes. We optimized this by performing **frustum culling on cluster AABB**(Axis-Aligned Bounding Box) instead, significantly reducing computational overhead.

Additionally, we implemented **compaction** of Gaussian point parameters to improve GPU utilization. While the original implementation skipped computations for culled Gaussian points, it left the GPU scheduler underutilized due to warp-based execution (32 threads). By compacting Gaussian point parameters, GPU resource utilization is improved.

It is important to note that compaction should be applied to Gaussian point clusters rather than individual primitives. Applying it at the primitive level can slow down the backward pass due to out-of-order memory writes.

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
In some cases, skipping shared memory reduction and directly performing warp-level reduction followed by AtomicAdd is the most efficient solution. Specifically, if the device has sufficient ROP (Raster Operations) units, shared memory reduction may be unnecessary. ROP unit is a hardware architecture in L2 cache to accelerate the atomic operation in NVIDIA gpu. While some GPUs may lack ROP units, all GPUs have hardware responsible for global atomic operations. For simplicity, we refer to these as ROP units. If the ratio of compute units to ROP units is favorable (e.g., Nvidia Turing architecture has 36 SMs and 96 ROP units, while Nvidia Ampere architecture has 84 SMs and 112 ROP units), AtomicAdd does not become a bottleneck.

In summary, warp-level reduction is a straightforward and efficient implementation that can be optimal for certain hardware configurations.

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
}
__syncthreads();
MultibatchSum(gradient);
```
