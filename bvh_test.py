from loader import TrainingDataLoader
from argparse import ArgumentParser
import typing
from loader.InfoLoader import CameraInfo,ImageInfo
from gaussian_splatting.scene import GaussianScene
from gaussian_splatting.model import GaussianSplattingModel
import training.loss
from training.loss.ssim import SSIM
from training.training import ViewManager
from training.densitycontroller import DensityControllerOurs
import torch
from util import tiles2img_torch,statistic_helper
from matplotlib import pyplot as plt 
from training.arguments import OptimizationParams
import time
import numpy as np


#load training data
cameras_info:typing.Dict[int,CameraInfo]=None
images_info:typing.List[ImageInfo]=None
scene:GaussianScene=None
cameras_info,images_info,scene,_,NerfNormRadius=TrainingDataLoader.load('./dataset/garden','images',2,-1)
image_size=images_info[0].image.size

args = OptimizationParams(ArgumentParser(description="Training script parameters"))


gaussian_model=GaussianSplattingModel(scene,NerfNormRadius)
gaussian_model.update_tiles_coord(image_size,16)
(model_params,op_state_dict, first_iter,actived_sh_degree) = torch.load('output/chkpnt200_baseline.pth')
gaussian_model.load_params(model_params)
gaussian_model.actived_sh_degree=actived_sh_degree


trainingset=[c for idx, c in enumerate(images_info) if idx % 8 != 0]
testset=[c for idx, c in enumerate(images_info) if idx % 8 == 0]
view_manager=ViewManager(trainingset,cameras_info)
view_manager_testset=ViewManager(testset,cameras_info)

view_matrix=torch.Tensor(view_manager.view_matrix_tensor).cuda()
view_project_matrix=view_matrix@(torch.Tensor(view_manager.proj_matrix_tensor).cuda())
camera_center=torch.Tensor(view_manager.camera_center_tensor).cuda()
camera_focal=torch.Tensor(view_manager.camera_focal_tensor).cuda()
ground_truth=torch.Tensor(view_manager.view_gt_tensor).cuda()
total_views_num=view_matrix.shape[0]
ssim_module=training.loss.LossSSIM().cuda()
statistic_helper.StatisticsHelperInst.reset(gaussian_model._xyz.shape[0])
statistic_helper.StatisticsHelperInst.start()
density_controller=DensityControllerOurs(args)

#bvh
from util.BVH.Object import GSpointBatch
from util.BVH.PytorchBVH import BVH
with torch.no_grad():
    chunksize=1024

    point_id=torch.arange(gaussian_model._xyz.shape[0],device='cuda')
    scale=gaussian_model._scaling.exp()
    roator=torch.nn.functional.normalize(gaussian_model._rotation,dim=-1)
    cov=gaussian_model.transform_to_cov3d(scale.unsqueeze(0),roator.unsqueeze(0))[0][0]
    points_batch=GSpointBatch(point_id,gaussian_model._xyz[:,:3],cov)
    bvh=BVH([points_batch,])
    bvh.build(chunksize)

    points_in_chunk=[]
    origin_list=[]
    extend_list=[]
    for node in bvh.leaf_nodes:
        point_id=torch.zeros(chunksize,device='cuda',dtype=torch.long)
        points_num=node.objs.shape[0]
        point_id[:points_num]=node.objs
        points_in_chunk.append(point_id)
        origin_list.append(node.origin.unsqueeze(0))
        extend_list.append(node.extend.unsqueeze(0))

    chunk_AABB_origin=torch.cat(origin_list)
    chunk_AABB_extend=torch.cat(extend_list)

    chunk_num=len(points_in_chunk)
    new_xyz=torch.zeros((chunk_num,chunksize,4),device='cuda')
    new_features_dc=torch.zeros((chunk_num,chunksize,1,3),device='cuda')
    new_features_rest=torch.zeros((chunk_num,chunksize,8,3),device='cuda')
    new_opacity=torch.zeros((chunk_num,chunksize,1),device='cuda')
    new_scaling=torch.zeros((chunk_num,chunksize,3),device='cuda')
    new_rotation=torch.zeros((chunk_num,chunksize,4),device='cuda')
    for chunk_index,point_ids in enumerate(points_in_chunk):
        #if chunk_index*chunksize+chunksize<gaussian_model._xyz.shape[0]:
        #    new_xyz[chunk_index]=gaussian_model._xyz[chunk_index*chunksize:chunk_index*chunksize+chunksize]
        new_xyz[chunk_index]=gaussian_model._xyz[point_ids]
        new_features_dc[chunk_index]=gaussian_model._features_dc[point_ids]
        new_features_rest[chunk_index]=gaussian_model._features_rest[point_ids]
        new_opacity[chunk_index]=gaussian_model._opacity[point_ids]
        new_scaling[chunk_index]=gaussian_model._scaling[point_ids]
        new_rotation[chunk_index]=gaussian_model._rotation[point_ids]

gaussian_model.chunk_AABB_origin=chunk_AABB_origin
gaussian_model.chunk_AABB_extend=chunk_AABB_extend
gaussian_model._xyz=torch.nn.Parameter(new_xyz)
gaussian_model._features_dc=torch.nn.Parameter(new_features_dc)
gaussian_model._features_rest=torch.nn.Parameter(new_features_rest)
gaussian_model._opacity=torch.nn.Parameter(new_opacity)
gaussian_model._scaling=torch.nn.Parameter(new_scaling)
gaussian_model._rotation=torch.nn.Parameter(new_rotation)
l = [
            {'params': [gaussian_model._xyz], 'lr': args.position_lr_init * 0.1, "name": "xyz"},
            {'params': [gaussian_model._features_dc], 'lr': args.feature_lr, "name": "f_dc"},
            {'params': [gaussian_model._features_rest], 'lr': args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [gaussian_model._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            {'params': [gaussian_model._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [gaussian_model._rotation], 'lr': args.rotation_lr, "name": "rotation"}
        ]
optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
optimizer.load_state_dict(op_state_dict)

start=time.monotonic()
for i in range(0,view_matrix.shape[0],1):
    with torch.no_grad():
        view_matrix_batch=view_matrix[i:i+1]
        view_project_matrix_batch=view_project_matrix[i:i+1]
        camera_focal_batch=camera_focal[i:i+1]
        camera_center_batch=camera_center[i:i+1]
        ground_truth_batch=ground_truth[i:i+1]

    tile_img,tile_transmitance=gaussian_model.render(None,None,
                              view_matrix_batch,view_project_matrix_batch,camera_focal_batch,camera_center_batch,None,
                              None)
    img=tiles2img_torch(tile_img,gaussian_model.cached_tiles_size[0],gaussian_model.cached_tiles_size[1])[...,:image_size[1],:image_size[0]].contiguous()

    l1_loss=training.loss.l1_loss(img,ground_truth_batch)
    ssim_loss=ssim_module(img,ground_truth_batch)
    loss=(1.0-0.2)*l1_loss+0.2*(1-ssim_loss)
    loss.backward()
    optimizer.zero_grad(set_to_none = True)


torch.cuda.synchronize()
end=time.monotonic()
print("takes:{0} ms".format(end-start))
