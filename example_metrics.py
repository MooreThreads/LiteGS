from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import psnr,ssim,lpip
import sys
import os
import matplotlib.pyplot as plt

import litegs
import litegs.config
import litegs.utils
import shutil

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp_cdo,op_cdo,pp_cdo,dp_cdo=litegs.config.get_default_arg()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo,parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_cdo,parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo,parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_cdo,parser)
    
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--save_image", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    lp=litegs.arguments.ModelParams.extract(args)
    op=litegs.arguments.OptimizationParams.extract(args)
    pp=litegs.arguments.PipelineParams.extract(args)
    dp=litegs.arguments.DensifyParams.extract(args)

    cameras_info:dict[int,litegs.data.CameraInfo]=None
    camera_frames:list[litegs.data.ImageFrame]=None
    if lp.source_type=="colmap":
        cameras_info,camera_frames,init_xyz,init_color=litegs.io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution
    elif lp.source_type=="slam":
        cameras_info,camera_frames,init_xyz,init_color=litegs.io_manager.load_slam_result(lp.source_path)#lp.sh_degree,lp.resolution

    if args.save_image:
        try:
            shutil.rmtree(os.path.join(lp.model_path,"Trainingset"))
            shutil.rmtree(os.path.join(lp.model_path,"Testset"))
        except:
            pass
        os.makedirs(os.path.join(lp.model_path,"Trainingset"),exist_ok=True)
        os.makedirs(os.path.join(lp.model_path,"Testset"),exist_ok=True)

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    if lp.eval:
        training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
        test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
        trainingset=litegs.data.CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
        train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
        testset=litegs.data.CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    else:
        trainingset=litegs.data.CameraFrameDataset(cameras_info,camera_frames,lp.resolution,pp.device_preload)
        train_loader = DataLoader(trainingset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #model
    xyz,scale,rot,sh_0,sh_rest,opacity=litegs.io_manager.load_ply(os.path.join(lp.model_path,"point_cloud","finish","point_cloud.ply"),lp.sh_degree)
    xyz=torch.Tensor(xyz).cuda()
    scale=torch.Tensor(scale).cuda()
    rot=torch.Tensor(rot).cuda()
    sh_0=torch.Tensor(sh_0).cuda()
    sh_rest=torch.Tensor(sh_rest).cuda()
    opacity=torch.Tensor(opacity).cuda()
    cluster_origin=None
    cluster_extend=None
    if pp.cluster_size>0:
        xyz,scale,rot,sh_0,sh_rest,opacity=litegs.scene.point.spatial_refine(False,None,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz,scale,rot,sh_0,sh_rest,opacity=litegs.scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        cluster_origin,cluster_extend=litegs.scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    if op.learnable_viewproj:
        noise_extr=torch.cat([frame.extr_params[None,:] for frame in trainingset.frames])
        noise_intr=torch.tensor(trainingset.cameras[0].intr_params,dtype=torch.float32,device='cuda').unsqueeze(0)
        denoised_training_extr,denoised_training_intr=torch.load(os.path.join(lp.model_path,"point_cloud","finish","viewproj.pth"))

    #metrics
    ssim_metrics=ssim.StructuralSimilarityIndexMeasure(data_range=(0.0,1.0)).cuda()
    psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
    lpip_metrics=lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    #iter
    if lp.eval:
        loaders={"Trainingset":train_loader,"Testset":test_loader}
    else:
        loaders={"Trainingset":train_loader}

    with torch.no_grad():
        for loader_name,loader in loaders.items():
            ssim_list=[]
            psnr_list=[]
            lpips_list=[]
            for index,(extr,intr,gt_image,idx) in enumerate(loader):
                extr=extr.cuda()
                intr=intr.cuda()
                gt_image=gt_image.cuda()/255.0
                idx=idx.cuda()
                if op.learnable_viewproj:
                    if loader_name=="Trainingset":
                        #fix view matrix
                        extr=denoised_training_extr[idx]
                        intr=denoised_training_intr
                    else:
                        nearest_idx=(extr-denoised_training_extr).abs().sum(dim=1).argmin()
                        delta=denoised_training_extr[nearest_idx]-noise_extr[nearest_idx]
                        extr=extr+delta
                view_matrix,proj_matrix,viewproj_matrix,frustumplane=litegs.utils.wrapper.CreateViewProj.apply(extr,intr,gt_image.shape[2],gt_image.shape[3],0.01,5000)

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity=litegs.render.render_preprocess(cluster_origin,cluster_extend,frustumplane,view_matrix,xyz,scale,rot,sh_0,sh_rest,opacity,op,pp,lp.sh_degree)
                img,transmitance,depth,normal,primitive_visible=litegs.render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,
                                                            lp.sh_degree,gt_image.shape[2:],pp)
                psnr_value=psnr_metrics(img,gt_image)
                ssim_list.append(ssim_metrics(img,gt_image).unsqueeze(0))
                psnr_list.append(psnr_value.unsqueeze(0))
                lpips_list.append(lpip_metrics(img,gt_image).unsqueeze(0))
                if args.save_image:
                    plt.imsave(os.path.join(lp.model_path,loader_name,"{}-{:.2f}-rd.png".format(index,float(psnr_value))),img.detach().cpu()[0].permute(1,2,0).numpy())
                    plt.imsave(os.path.join(lp.model_path,loader_name,"{}-{:.2f}-gt.png".format(index,float(psnr_value))),gt_image.detach().cpu()[0].permute(1,2,0).numpy())
            ssim_mean=torch.concat(ssim_list,dim=0).mean()
            psnr_mean=torch.concat(psnr_list,dim=0).mean()
            lpips_mean=torch.concat(lpips_list,dim=0).mean()

            print("  Scene:{0}".format(lp.model_path+" "+loader_name))
            print("  SSIM : {:>12.7f}".format(float(ssim_mean)))
            print("  PSNR : {:>12.7f}".format(float(psnr_mean)))
            print("  LPIPS: {:>12.7f}".format(float(lpips_mean)))
            print("")
