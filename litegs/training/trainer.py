import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr
from tqdm import tqdm
import numpy as np
import math
import os
import torch.cuda.nvtx as nvtx
import matplotlib.pyplot as plt
import json

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from ..utils.statistic_helper import StatisticsHelperInst
from . import densify
from .. import utils

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint:str=None):
    
    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.ImageFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    if lp.eval:
        if os.path.exists(os.path.join(lp.source_path,"train_test_split.json")):
            with open(os.path.join(lp.source_path,"train_test_split.json"), "r") as file:
                train_test_split = json.load(file)
                training_frames=[c for c in camera_frames if c.name in train_test_split["train"]]
                test_frames=[c for c in camera_frames if c.name in train_test_split["test"]]
        else:
            training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames=camera_frames
        test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    init_points_num=init_xyz.shape[0]
    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        if pp.cluster_size:
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op,pp)
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
    if pp.cluster_size:
        cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    actived_sh_degree=0

    #learnable view matrix
    if op.learnable_viewproj:
        noise_extr=torch.cat([frame.extr_params[None,:] for frame in trainingset.frames])
        denoised_training_extr=torch.nn.Embedding(noise_extr.shape[0],noise_extr.shape[1],_weight=noise_extr.clone(),sparse=True)
        noise_intr=torch.tensor(list(trainingset.cameras.values())[0].intr_params,dtype=torch.float32,device='cuda').unsqueeze(0)
        denoised_training_intr=torch.nn.Parameter(torch.tensor(list(trainingset.cameras.values())[0].intr_params,dtype=torch.float32,device='cuda').unsqueeze(0))#todo fix multi cameras
        view_opt=torch.optim.SparseAdam(denoised_training_extr.parameters(),lr=1e-4)
        proj_opt=torch.optim.Adam([denoised_training_intr,],lr=1e-5)

    #init
    total_epoch=int(op.iterations/len(trainingset))
    if dp.densify_until<0:
        dp.densify_until=int(total_epoch*0.8/dp.opacity_reset_interval)*dp.opacity_reset_interval+1
    density_controller=densify.DensityControllerTamingGS(norm_radius,dp,pp.cluster_size>0,init_points_num)
    StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)

    for epoch in range(start_epoch,total_epoch):

        with torch.no_grad():
            if pp.cluster_size>0 and (epoch-1)%dp.densification_interval==0:
                xyz,scale,rot,sh_0,sh_rest,opacity=scene.spatial_refine(pp.cluster_size>0,opt,xyz)
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

        with StatisticsHelperInst.try_start(epoch):
            for view_matrix,proj_matrix,frustumplane,gt_image,idx in train_loader:
                nvtx.range_push("Iter Init")
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                idx=idx.cuda()
                if op.learnable_viewproj:
                    #fix view matrix
                    extr=denoised_training_extr(idx)
                    intr=denoised_training_intr
                    view_matrix,proj_matrix,viewproj_matrix,frustumplane=utils.wrapper.CreateViewProj.apply(extr,intr,gt_image.shape[2],gt_image.shape[3],0.01,5000)
                nvtx.range_pop()

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,view_matrix,xyz,scale,rot,sh_0,sh_rest,opacity,op,pp,actived_sh_degree)
                img,transmitance,depth,normal,primitive_visible=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp)
                
                l1_loss=__l1_loss(img,gt_image)
                ssim_loss:torch.Tensor=1-fused_ssim.fused_ssim(img,gt_image)
                loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*ssim_loss
                loss+=(culled_scale).square().mean()*op.reg_weight
                if pp.enable_transmitance:
                    loss+=(1-transmitance).abs().mean()
                loss.backward()
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if pp.sparse_grad:
                    opt.step(visible_chunkid,primitive_visible)
                else:
                    opt.step()
                opt.zero_grad(set_to_none = True)
                if op.learnable_viewproj:
                    view_opt.step()
                    view_opt.zero_grad()
                    # proj_opt.step()
                    # proj_opt.zero_grad()
                schedular.step()

        if epoch in test_epochs:
            with torch.no_grad():
                _cluster_origin=None
                _cluster_extend=None
                if pp.cluster_size:
                    _cluster_origin,_cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
                psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
                loaders={"Trainingset":train_loader}
                if lp.eval:
                    loaders["Testset"]=test_loader
                for name,loader in loaders.items():
                    psnr_list=[]
                    for view_matrix,proj_matrix,frustumplane,gt_image,idx in loader:
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        frustumplane=frustumplane.cuda()
                        gt_image=gt_image.cuda()/255.0
                        idx=idx.cuda()
                        if op.learnable_viewproj:
                            if name=="Trainingset":
                                #fix view matrix
                                extr=denoised_training_extr(idx)
                                intr=denoised_training_intr
                            else:
                                nearest_idx=(extr-denoised_training_extr._parameters['weight']).abs().sum(dim=1).argmin()
                                delta=denoised_training_extr(nearest_idx)-noise_extr[nearest_idx]
                                extr=extr+delta
                            view_matrix,proj_matrix,viewproj_matrix,frustumplane=utils.wrapper.CreateViewProj.apply(extr,intr,gt_image.shape[2],gt_image.shape[3],0.01,5000)

                        #cluster culling
                        visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,view_matrix,xyz,scale,rot,sh_0,sh_rest,opacity,op,pp,actived_sh_degree)
                        img,transmitance,depth,normal,primitive_visible=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,
                                                                    actived_sh_degree,gt_image.shape[2:],pp)
                        psnr_list.append(psnr_metrics(img,gt_image).unsqueeze(0))
                    tqdm.write("\n[EPOCH {}] {} Evaluating: PSNR {}".format(epoch,name,torch.concat(psnr_list,dim=0).mean()))

        xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch)
        progress_bar.update()  

        if epoch in save_ply or epoch==total_epoch-1:
            if epoch==total_epoch-1:
                progress_bar.close()
                print("{} takes: {}".format(lp.model_path,progress_bar.format_dict['elapsed']))
                save_path=os.path.join(lp.model_path,"point_cloud","finish")
            else:
                save_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch))    

            if pp.cluster_size:
                tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            io_manager.save_ply(os.path.join(save_path,"point_cloud.ply"),*param_nyp)
            if op.learnable_viewproj:
                torch.save(list(denoised_training_extr.parameters())+[denoised_training_intr],os.path.join(save_path,"viewproj.pth"))

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)
    
    return