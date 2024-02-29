# A Better implementation of Gaussian Splatting
Welcome to this repository, a refactored codebase aimed at enhancing the functionality and flexibility of the Gaussian splatting. 

## Background
Gaussian splatting is a powerful technique used in various computer graphics and vision applications. It involves mapping a set of points onto a grid using Gaussian distributions, allowing for efficient and accurate representation of spatial data. However, the original implementation of Gaussian splatting in PyTorch presented a challenge: 

1. both the forward and backward passes were encapsulated within a single PyTorch operator, making it difficult to access intermediate variables without delving into the C code.
2. even we try to modify the C code, we must caculate the gradient formula manually and implement these formula in backward.

## Features

1. Modular Design: The refactored Gaussian splatting implementation separates the forward and backward into individual PyTorch operators, allowing for greater modularity and accessibility.

2. Fewer Atomic Operation: Instead of using a lot of AtomicAdd in backward, using reduction in shared memory is much more efficient.