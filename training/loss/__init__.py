import torch
import math
import torch.nn.functional as F

def l1_loss(network_output:torch.tensor, gt:torch.tensor):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output:torch.tensor, gt:torch.tensor):
    return ((network_output - gt) ** 2).mean()