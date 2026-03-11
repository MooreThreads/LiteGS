from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import psnr,ssim,lpip
import sys
import os
import matplotlib.pyplot as plt
import json
import shutil

import litegs
import litegs.arguments
import litegs.utils
from litegs.scene.model import GaussianSplattingModel
from litegs.render.pipeline import RenderPipeline

if __name__ == "__main__":
    cfg:litegs.arguments.EvalConfig=litegs.arguments.get_config(litegs.arguments.EvalConfig)

    cameras_info:dict[int,litegs.data.CameraInfo]=None
    camera_frames:list[litegs.data.ImageFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=litegs.io_manager.load_colmap_result(cfg.dataset.source_path,cfg.dataset.images)#lp.sh_degree,lp.resolution


    if cfg.save_image:
        try:
            shutil.rmtree(os.path.join(cfg.dataset.model_path,"Trainingset"))
            shutil.rmtree(os.path.join(cfg.dataset.model_path,"Testset"))
        except:
            pass
        os.makedirs(os.path.join(cfg.dataset.model_path,"Trainingset"),exist_ok=True)
        os.makedirs(os.path.join(cfg.dataset.model_path,"Testset"),exist_ok=True)

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(cfg.dataset.resolution)

    #Dataset
    if cfg.dataset.eval:
        if os.path.exists(os.path.join(cfg.dataset.source_path,"train_test_split.json")):
            with open(os.path.join(cfg.dataset.source_path,"train_test_split.json"), "r") as file:
                train_test_split = json.load(file)
                training_frames=[c for c in camera_frames if c.name in train_test_split["train"]]
                test_frames=[c for c in camera_frames if c.name in train_test_split["test"]]
        else:
            training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
        trainingset=litegs.data.CameraFrameDataset(cameras_info,training_frames,cfg.dataset.resolution,cfg.dataset.device_preload)
        train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not cfg.dataset.device_preload)
        testset=litegs.data.CameraFrameDataset(cameras_info,test_frames,cfg.dataset.resolution,cfg.dataset.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not cfg.dataset.device_preload)
    else:
        trainingset=litegs.data.CameraFrameDataset(cameras_info,camera_frames,cfg.dataset.resolution,cfg.dataset.device_preload)
        train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not cfg.dataset.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #model
    ply_path=os.path.join(cfg.dataset.model_path,"finish","point_cloud.ply")
    model=GaussianSplattingModel.from_ply(ply_path,cfg.model,None,None,None)
    pipeline = RenderPipeline(cfg.pipeline, model ,trainingset)
    if cfg.pipeline.learnable_viewproj:
        pipeline.learnable_viewproj.load_state_dict(os.path.join(cfg.dataset.model_path,"finish","viewproj.pt"))

    #metrics
    ssim_metrics=ssim.StructuralSimilarityIndexMeasure(data_range=(0.0,1.0)).cuda()
    psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
    lpip_metrics=lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    #iter
    if cfg.dataset.eval:
        loaders={"Trainingset":train_loader,"Testset":test_loader}
    else:
        loaders={"Trainingset":train_loader}

    pipeline.eval()
    with torch.no_grad():
        for loader_name,loader in loaders.items():
            ssim_list=[]
            psnr_list=[]
            lpips_list=[]
            for index,(view_matrix,proj_matrix,frustumplane,gt_image,idx_tensor) in enumerate(loader):
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                
                img,transmitance,depth,normal=pipeline(view_matrix, proj_matrix, frustumplane, idx_tensor, gt_image.shape[2:])
                
                psnr_value=psnr_metrics(img,gt_image)
                ssim_list.append(ssim_metrics(img,gt_image).unsqueeze(0))
                psnr_list.append(psnr_value.unsqueeze(0))
                lpips_list.append(lpip_metrics(img,gt_image).unsqueeze(0))
                if loader_name=="Testset" and cfg.save_image:
                    plt.imsave(os.path.join(cfg.dataset.model_path,loader_name,"{}-{:.2f}-rd.png".format(index,float(psnr_value))),img.detach().cpu()[0].permute(1,2,0).numpy())
                    plt.imsave(os.path.join(cfg.dataset.model_path,loader_name,"{}-{:.2f}-gt.png".format(index,float(psnr_value))),gt_image.detach().cpu()[0].permute(1,2,0).numpy())
            ssim_mean=torch.concat(ssim_list,dim=0).mean()
            psnr_mean=torch.concat(psnr_list,dim=0).mean()
            lpips_mean=torch.concat(lpips_list,dim=0).mean()

            print("  Scene:{0}".format(cfg.dataset.model_path+" "+loader_name))
            print("  SSIM : {:>12.7f}".format(float(ssim_mean)))
            print("  PSNR : {:>12.7f}".format(float(psnr_mean)))
            print("  LPIPS: {:>12.7f}".format(float(lpips_mean)))
            print("")
