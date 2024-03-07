import torch

def mse(img1:torch.Tensor, img2:torch.Tensor , keep_channel=True):
    if keep_channel:
        return (((img1 - img2)) ** 2).view(img1.shape[0],img1.shape[1], -1).mean(2)
    return (((img1 - img2)) ** 2).view(img1.shape[0],img1.shape[1], -1).mean(2).mean(1)

def psnr(img1:torch.Tensor, img2:torch.Tensor):
    mse_result = mse(img1,img2,True)
    return (20 * torch.log10(1.0 / torch.sqrt(mse_result))).mean(1)