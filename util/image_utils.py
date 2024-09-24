import torch
import math

def mse(img1:torch.Tensor, img2:torch.Tensor , keep_channel=True):
    if keep_channel:
        return (((img1 - img2)) ** 2).view(img1.shape[0],img1.shape[1], -1).mean(2)
    return (((img1 - img2)) ** 2).view(img1.shape[0],img1.shape[1], -1).mean(2).mean(1)

def psnr(img1:torch.Tensor, img2:torch.Tensor):
    mse_result = mse(img1,img2,True)
    return (20 * torch.log10(1.0 / torch.sqrt(mse_result))).mean(1)



__cached_window:dict[tuple[int,int],torch.Tensor]={}


def ssim(img1, img2, window_size=11, size_average=True):
    def create_window(window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    channel = img1.size(-3)
    if __cached_window.get((window_size, channel),None) is None:
        window = create_window(window_size, channel)
        __cached_window.clear()
        __cached_window[(window_size, channel)]=window
    else:
        window = __cached_window[(window_size, channel)]

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)