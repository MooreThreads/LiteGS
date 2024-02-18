from loader import InfoLoader
from loader import PointCloudLoader
import os
from PIL import Image
from gaussian_splatting.gaussian_util import GaussianScene,RGB2SH
import torch
import numpy as np
from simple_knn._C import distCUDA2

def BasicScene2GaussianScene(basic_scene:PointCloudLoader.BasicPointCloud,sh_degree:int)->GaussianScene:
    fused_point_cloud = torch.tensor(basic_scene.positions).float()
    fused_color = RGB2SH(torch.tensor(basic_scene.colors).float())
    features = torch.zeros((fused_color.shape[0], 3, (sh_degree + 1) ** 2)).float()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(basic_scene.positions)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3).cpu()
    rots = torch.zeros((fused_point_cloud.shape[0], 4))
    rots[:, 0] = 1

    temp=0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float)
    opacities = torch.log(temp/(1-temp))#inverse_sigmoid

    scene=GaussianScene(sh_degree,fused_point_cloud.numpy(),scales.numpy(),rots.numpy(),features.numpy(),opacities.numpy())
    return scene

def load(data_path:str,img_dir:str,sh_degree:int):
    cameras_info,images_info,NerfNormTrans,NerfNormRadius=InfoLoader.load(data_path,img_dir)
    basic_scene=PointCloudLoader.load_pointcloud(data_path)
    print("Number of points at initialisation : ", basic_scene.positions.shape[0])

    scene=BasicScene2GaussianScene(basic_scene,sh_degree)
    return cameras_info,images_info,scene,NerfNormTrans,NerfNormRadius
    
