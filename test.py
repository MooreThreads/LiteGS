import torch


M=torch.randn([ 2, 2],device='cuda',requires_grad=True)
result=M@M.transpose(0,1)
result.sum().backward()


n=0
