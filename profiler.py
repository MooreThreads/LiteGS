import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
import fused_ssim
from argparse import ArgumentParser
import sys
import matplotlib.pyplot as plt

import litegs
import litegs.config
from litegs.utils.wrapper import litegs_fused
from litegs.training import optimizer
from litegs import render,scene


def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

if __name__ == "__main__":

    scene_name="garden"
    parser = ArgumentParser(description="Training script parameters")
    args = parser.parse_args(sys.argv[1:])

    cameras_info:dict[int,litegs.data.CameraInfo]=None
    camera_frames:list[litegs.data.CameraFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=litegs.io_manager.load_colmap_result('./dataset/mipnerf360/{}'.format(scene_name),'images_4')

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image()

    #Dataset
    training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
    test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    trainingset=litegs.data.CameraFrameDataset(cameras_info,training_frames,-1,True)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=False)
    testset=litegs.data.CameraFrameDataset(cameras_info,test_frames,-1,True)
    test_loader = DataLoader(testset, batch_size=1,shuffle=False)

    xyz,scale,rot,sh_0,sh_rest,opacity=litegs.io_manager.load_ply('output/{}-500k/point_cloud/finish/point_cloud.ply'.format(scene_name),3)
    xyz=torch.Tensor(xyz).cuda()
    scale=torch.Tensor(scale).cuda()
    rot=torch.Tensor(rot).cuda()
    sh_0=torch.Tensor(sh_0).cuda()
    sh_rest=torch.Tensor(sh_rest).cuda()
    opacity=torch.Tensor(opacity).cuda()
    lp,op,pp,dp=litegs.config.get_default_arg()
    if pp.cluster_size:
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
    xyz=torch.nn.Parameter(xyz.contiguous())
    scale=torch.nn.Parameter(scale.contiguous())
    rot=torch.nn.Parameter(rot.contiguous())
    sh_0=torch.nn.Parameter(sh_0.contiguous())
    sh_rest=torch.nn.Parameter(sh_rest.contiguous())
    opacity=torch.nn.Parameter(opacity.contiguous())
    op.feature_lr=0
    op.opacity_lr=0
    op.scaling_lr=0
    op.position_lr_final=0
    op.position_lr_init=0
    op.rotation_lr=0

    opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,0,op,pp)

    cluster_origin=None
    cluster_extend=None
    if pp.cluster_size:
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.spatial_refine(pp.cluster_size!=0,opt,xyz)
        cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz[:3],scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        for i in range(10):
            for view_matrix,proj_matrix,frustumplane,gt_image in train_loader:
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0

                #cluster culling.
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                                xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                img,transmitance,depth,normal,_=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                            3,gt_image.shape[2:],pp)
                
                # l1_loss=__l1_loss(img,gt_image)
                # ssim_loss:torch.Tensor=1-fused_ssim.fused_ssim(img,gt_image)
                # loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*ssim_loss
                # loss+=(culled_scale).square().mean()*op.reg_weight
                # loss.backward()
                # opt.step(visible_chunkid)
                # opt.zero_grad(set_to_none = True)
                img.sum().backward()
                opt.zero_grad(set_to_none = True)
    print(prof.key_averages(group_by_input_shape=False).table(sort_by='self_cuda_time_total', row_limit=10))
    pass