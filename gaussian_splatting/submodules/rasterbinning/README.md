# RasterBinning: PyTorch CUDA Extension

This repository contains a RasterBinning implementation as a PyTorch CUDA extension. 
It is used by our training code.

## Installation
`pip install rasterbinning`

## Documentation
Please see todo

## Troubleshooting
If you get SIGSEGV upon importing,
check that your CUDA runtime and PyTorch CUDA versions match.  That is,
`nvcc --version`
should match (Python)
`torch.version.cuda`
