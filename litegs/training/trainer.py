import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr
from tqdm import tqdm
from matplotlib import pyplot as plt 
import numpy as np
import os

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from .optimizer import SparseGaussianAdam
from ..utils import wrapper
from ..utils.statistic_helper import StatisticsHelperInst
from . import densify

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor):
    return torch.abs((network_output - gt)).mean()

def render_preprocess(cluster_origin:torch.Tensor,cluster_extend:torch.Tensor,view_matrix:torch.Tensor,proj_matrix:torch.Tensor,
                      xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,sh_0:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor,
                      op:arguments.OptimizationParams,pp:arguments.PipelineParams):
    if pp.cluster_size:
        if cluster_origin is None or cluster_extend is None:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
        visible_chunkid=scene.cluster.get_visible_cluster(cluster_origin,cluster_extend,view_matrix,proj_matrix)
        if op.sparse_grad:#enable sparse gradient
            culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=wrapper.CompactVisibleWithSparseGrad.apply(visible_chunkid,xyz,scale,rot,sh_0,sh_rest,opacity)
        else:
            culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=scene.cluster.culling(visible_chunkid,xyz,scale,rot,sh_0,sh_rest,opacity)
        culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=scene.cluster.uncluster(culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity)
        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.set_compact_mask(visible_chunkid)
    else:
        culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=xyz,scale,rot,sh_0,sh_rest,opacity
        visible_chunkid=None
    return visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint:str=None):
    
    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.CameraFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image()

    #Dataset
    if lp.eval:
        training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
        test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames=camera_frames
        test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=True)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution)
        test_loader = DataLoader(testset, batch_size=1,shuffle=True,pin_memory=True)
    norm_trans,norm_radius=trainingset.get_norm()

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        #xyz,scale,rot,sh_0,sh_rest,opacity=scene.spatial_refine(False,None,xyz,scale,rot,sh_0,sh_rest,opacity)
        if pp.cluster_size:
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
            #cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op)
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
        if pp.cluster_size:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    actived_sh_degree=0

    #init
    density_controller=densify.DensityControllerOfficial(norm_radius,dp,pp.cluster_size>0)
    StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, op.iterations), desc="Training progress")
    progress_bar.update(0)

    for epoch in range(start_epoch,op.iterations):

        with torch.no_grad():
            if epoch%op.spatial_refine_interval==0:#spatial refine
                scene.spatial_refine(pp.cluster_size>0,opt,xyz)
            if pp.cluster_size>0 and (epoch%op.spatial_refine_interval==0 or density_controller.is_densify_actived(epoch-1)):
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=int(epoch/5)

        with StatisticsHelperInst.try_start(epoch):
            for view_matrix,proj_matrix,gt_image in train_loader:
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                gt_image=gt_image.cuda()

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render_preprocess(cluster_origin,cluster_extend,view_matrix,proj_matrix,
                                                                                                               xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                img,transmitance,depth,normal=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp)
                
                l1_loss:torch.Tensor=__l1_loss(img,gt_image)
                ssim_loss:torch.Tensor=fused_ssim.fused_ssim(img,gt_image)
                loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*(1-ssim_loss)
                loss.backward()
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if op.sparse_grad:
                    opt.step(visible_chunkid)
                else:
                    opt.step()
                opt.zero_grad(set_to_none = True)
        schedular.step()

        if epoch in test_epochs:
            with torch.no_grad():
                psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
                loaders={"Trainingset":train_loader}
                if lp.eval:
                    loaders["Testset"]=test_loader
                for name,loader in loaders.items():
                    psnr_list=[]
                    for view_matrix,proj_matrix,gt_image in loader:
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        gt_image=gt_image.cuda()
                        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render_preprocess(cluster_origin,cluster_extend,view_matrix,proj_matrix,
                                                                                                                xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                        img,transmitance,depth,normal=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                                    actived_sh_degree,gt_image.shape[2:],pp)
                        psnr_list.append(psnr_metrics(img,gt_image).unsqueeze(0))
                    tqdm.write("\n[EPOCH {}] {} Evaluating: PSNR {}".format(epoch,name,torch.concat(psnr_list,dim=0).mean()))

        xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch)

        progress_bar.update()  

        if epoch in save_ply or epoch==op.iterations-1:
            if pp.cluster_size:
                tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            if epoch==op.iterations-1:
                ply_path=os.path.join(lp.model_path,"point_cloud","finish","point_cloud.ply")
            else:
                ply_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch),"point_cloud.ply")
            io_manager.save_ply(ply_path,*param_nyp)
            pass

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)
    
    return