import torch
from torch.utils.data import DataLoader
import fused_ssim

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor):
    return torch.abs((network_output - gt)).mean()

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,start_checkpoint=None):
    
    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.CameraFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #Dataset
    if lp.eval:
        training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
        test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames=camera_frames
        test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,num_workers=1)
    test_loader=None
    if test_frames is not None:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution)
        test_loader = DataLoader(testset, batch_size=1,shuffle=True,num_workers=1)

    #torch parameter
    bClustered=False
    cluster_origin=None
    cluster_extend=None
    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        if pp.cluster_interval>0:
            bClustered=True
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale,rot)
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity=io_manager.load_checkpoint(start_checkpoint)

    norm_trans,norm_radius=trainingset.get_norm()
    opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op)

    actived_sh_degree=0
    for epoch in range(op.iterations):
        with torch.no_grad():
            scene.refine_spatial(bClustered,opt,xyz)
            if pp.cluster_interval>0 and epoch%pp.cluster_interval==0:
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale,rot)

        for view_matrix,proj_matrix,image in train_loader:
            view_matrix=view_matrix.cuda()
            proj_matrix=proj_matrix.cuda()
            image=image.cuda()
            if bClustered:
                visible_chunkid=scene.cluster.get_visible_cluster(cluster_origin,cluster_extend,view_matrix,proj_matrix)
                xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.culling(visible_chunkid,xyz,scale,rot,sh_0,sh_rest,opacity)
                xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            img,transmitance,depth,normal=render.render(view_matrix,proj_matrix,actived_sh_degree,xyz,scale,rot,sh_0,sh_rest,opacity,image.shape[2:],pp)
            l1_loss=__l1_loss(img,image)
            ssim_loss=fused_ssim.fused_ssim(img,image)
            loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*(1-ssim_loss)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none = True)
        schedular.step()
    
    #io_manager.save
    return