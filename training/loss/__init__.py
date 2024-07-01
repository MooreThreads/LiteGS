import torch
import math
import torch.nn.functional as F

def l1_loss(network_output:torch.tensor, gt:torch.tensor):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output:torch.tensor, gt:torch.tensor):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size:int, sigma:float):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _ssim(img1,img2,window,window_size,channel,size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class LossSSIM(torch.nn.Module):
    def __init__(self,window_size:int=11,img_channel:int=3):
        super(LossSSIM, self).__init__()
        self.window_size=window_size
        self.channel=img_channel
        gaussian_kernel=self.__create_window()
        self.register_buffer('gaussian_kernel',gaussian_kernel)
        return


    def forward(self,img1,img2,size_average=True):
        result=_ssim(img1,img2,self.gaussian_kernel,self.window_size,self.channel,size_average)
        return result
    
    def __create_window(self):
        _1D_window = gaussian(self.window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(_2D_window.expand(self.channel, 1, self.window_size, self.window_size).contiguous())
        return window
    