import torch

J_transposed=torch.zeros([2, 69132, 3, 3],device='cuda',requires_grad=True)
view_matrix=torch.zeros([2, 1, 3, 3],device='cuda',requires_grad=False)
cov3d=torch.zeros([2, 69132, 3, 3],device='cuda',requires_grad=True)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],use_cuda=True) as prof:
    for i in range(93):
        T_trans=torch.matmul(J_transposed,view_matrix)
        temp=torch.matmul(T_trans,cov3d)
        cov2d=torch.matmul(temp,temp.transpose(2,3))
        cov2d.sum().backward()
print(prof.key_averages().table(sort_by="cuda_time_total"))