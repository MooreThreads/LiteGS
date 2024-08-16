'''
code copy from
https://github.com/One-sixth/ms_ssim_pytorch/blob/master/ssim.py
'''

import torch
import torch.jit
import torch.nn.functional as F
from util.platform import platform_torch_compile


def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


@platform_torch_compile
def ssim(X, Y, window, data_range: float, use_padding: bool=False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    def _gaussian_filter(x, window_1d, use_padding: bool):
        '''
        Blur input with 1-D kernel
        :param x: batch of tensors to be blured
        :param window_1d: 1-D gauss kernel
        :param use_padding: padding image before conv
        :return: blured tensors
        '''
        C = x.shape[1]
        padding = 0
        if use_padding:
            window_size = window_1d.shape[3]
            padding = window_size // 2
        out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
        out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
        return out

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = sigma1_sq - mu1_sq
    sigma2_sq = sigma2_sq - mu2_sq
    sigma12 = sigma12 - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW

    return ssim_val


class SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding

    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return r
