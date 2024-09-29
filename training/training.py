from gaussian_splatting.model import GaussianSplattingModel
from training.arguments import OptimizationParams,ModelParams
from gaussian_splatting.scene import GaussianScene
from loader.InfoLoader import CameraInfo,ImageInfo,PinHoleCameraInfo
import training.loss.ssim
from training.view_manager import ViewManager
import training.loss
from training.densitycontroller import DensityControllerOfficial
from util.statistic_helper import StatisticsHelperInst
from util import cg_torch,image_utils,tiles2img_torch,img2tiles_torch

import torch
import typing
from tqdm import tqdm
import numpy as np
import math
import random
from matplotlib import pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision


    
    

class GaussianTrainer:

    def __init__(self,gaussian_model:GaussianSplattingModel,lp:ModelParams,op:OptimizationParams,NerfNormRadius:int,image_list:list[ImageInfo],camera_dict:dict[int,PinHoleCameraInfo]):
        self.spatial_lr_scale=NerfNormRadius
        self.image_list=image_list
        self.camera_dict=camera_dict
        if lp.eval:
            trainingset=[c for idx, c in enumerate(self.image_list) if idx % 8 != 0]
            testset=[c for idx, c in enumerate(self.image_list) if idx % 8 == 0]
            self.view_manager=ViewManager(trainingset,self.camera_dict)
            self.view_manager_testset=ViewManager(testset,self.camera_dict)

        else:
            self.view_manager=ViewManager(self.image_list,self.camera_dict)
            self.view_manager_testset=None
        self.model=gaussian_model
        self.output_path=lp.model_path
        self.__training_setup(gaussian_model,op)
        self.opt_params=op

        self.image_size=self.image_list[0].image.size
        self.tile_size=8
        self.iter_start=0

        screen_size_threshold=None#20
        opacity_threshold=0.005
        self.density_controller=DensityControllerOfficial(op.densify_grad_threshold,opacity_threshold,screen_size_threshold,op.percent_dense,torch.Tensor(self.view_manager.view_matrix_tensor).cuda(),self.opt_params)

        return
    
    def save(self,iteration):
        model_params=self.model.get_params()
        torch.save((model_params,self.optimizer.state_dict(), iteration,self.model.actived_sh_degree), self.output_path + "/chkpnt" + str(iteration) + ".pth")
        return
    
    def restore(self,checkpoint):
        (model_params,op_state_dict, first_iter,actived_sh_degree) = torch.load(checkpoint)
        self.model.load_params(model_params)
        self.__training_setup(self.model,self.opt_params)
        self.optimizer.load_state_dict(op_state_dict)
        self.iter_start=first_iter+1
        self.model.actived_sh_degree=actived_sh_degree
        self.model.rebuild_AABB()
        return
    
    def __training_setup(self,gaussian_model:GaussianSplattingModel,args:OptimizationParams):
        def get_expon_lr_func(
            lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
        ):
            """
            Copied from Plenoxels

            Continuous learning rate decay function. Adapted from JaxNeRF
            The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
            is log-linearly interpolated elsewhere (equivalent to exponential decay).
            If lr_delay_steps>0 then the learning rate will be scaled by some smooth
            function of lr_delay_mult, such that the initial learning rate is
            lr_init*lr_delay_mult at the beginning of optimization but will be eased back
            to the normal learning rate when steps>lr_delay_steps.
            :param conf: config subtree 'lr' or similar
            :param max_steps: int, the number of steps during optimization.
            :return HoF which takes step as input
            """

            def helper(step):
                if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                    # Disable this parameter
                    return 0.0
                if lr_delay_steps > 0:
                    # A kind of reverse cosine decay.
                    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                    )
                else:
                    delay_rate = 1.0
                t = np.clip(step / max_steps, 0, 1)
                log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
                return delay_rate * log_lerp

            return helper


        l = [
            {'params': [gaussian_model._xyz], 'lr': args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [gaussian_model._features_dc], 'lr': args.feature_lr, "name": "f_dc"},
            {'params': [gaussian_model._features_rest], 'lr': args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [gaussian_model._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            {'params': [gaussian_model._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [gaussian_model._rotation], 'lr': args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=args.position_lr_delay_mult,
                                                    max_steps=args.position_lr_max_steps)
        
        #self.tb_writer = SummaryWriter(self.output_path)
        return
    
    def __update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    @staticmethod
    def __regularization_loss_backward(positions,scales,rotators,sh_base,sh_rest,opacities):
        #regularization_loss=(1-visible_opacities).mean()*0.001+visible_scales.var(2).mean()*0.1
        regularization_loss=scales.var(1).mean()*0.1
        regularization_loss.backward(retain_graph=True)
        return    
    
    def __iter(self,epoch_i:int,batch_size:int,
                            view_matrix:torch.Tensor,view_project_matrix:torch.Tensor,camera_center:torch.Tensor,camera_focal:torch.Tensor,ground_truth:torch.Tensor):
        

        
        with torch.no_grad():
            total_views_num=view_matrix.shape[0]
        
        log_loss=0
        counter=0
        iter_range=list(range(0,total_views_num,batch_size))
        random.shuffle(iter_range)
        ssim_module=training.loss.ssim.SSIM(data_range=1.0).cuda()

        ### iter batch ###
        for iter_i,i in enumerate(iter_range):
            batch_tail=min(i+batch_size,total_views_num)
            self.__update_learning_rate(epoch_i*total_views_num+iter_i+1)

            ### gather batch data ###
            with torch.no_grad():
                view_matrix_batch=view_matrix[i:batch_tail]
                view_project_matrix_batch=view_project_matrix[i:batch_tail]
                camera_focal_batch=camera_focal[i:batch_tail]
                camera_center_batch=camera_center[i:batch_tail]
                ground_truth_batch=ground_truth[i:batch_tail].contiguous()

            ### render ###
            tile_img,tile_transmitance=self.model.render(view_matrix_batch,view_project_matrix_batch,camera_focal_batch,camera_center_batch,
                                                         None,None)
            img=tiles2img_torch(tile_img,self.model.cached_tiles_size[0],self.model.cached_tiles_size[1])[...,:self.image_size[1],:self.image_size[0]].contiguous()
            #transmitance=tiles2img_torch(tile_transmitance,self.model.cached_tiles_size[0],self.model.cached_tiles_size[1])[...,:self.image_size[1],:self.image_size[0]].contiguous()

            #### loss ###
            l1_loss=training.loss.l1_loss(img,ground_truth_batch)
            ssim_loss=ssim_module(img,ground_truth_batch)
            loss=(1.0-self.opt_params.lambda_dssim)*l1_loss+self.opt_params.lambda_dssim*(1-ssim_loss)
            loss.backward()
            log_loss+=l1_loss.detach()
            counter+=1

            self.optimizer.step()
            if StatisticsHelperInst.bStart:
                StatisticsHelperInst.backward_callback()
            self.optimizer.zero_grad(set_to_none = True)

        return

    @torch.no_grad()
    def inference(gs_model:GaussianSplattingModel,view_manager:ViewManager=None,bLog:bool=False,
                  iter_callback:typing.Callable[[int,str,torch.Tensor,torch.Tensor],typing.Any]=None):
        
        view_matrix=torch.Tensor(view_manager.view_matrix_tensor).cuda()
        view_project_matrix=view_matrix@(torch.Tensor(view_manager.proj_matrix_tensor).cuda())
        camera_center=torch.Tensor(view_manager.camera_center_tensor).cuda()
        camera_focal=torch.Tensor(view_manager.camera_focal_tensor).cuda()
        ground_truth=torch.Tensor(view_manager.view_gt_tensor).cuda()
        total_views_num=view_matrix.shape[0]

        callback_result_list=[]
        Visibility={}

        for i in range(total_views_num):

            ### gather batch data ###
            with torch.no_grad():
                view_matrix_batch=view_matrix[i:i+1]
                view_project_matrix_batch=view_project_matrix[i:i+1]
                camera_focal_batch=camera_focal[i:i+1]
                camera_center_batch=camera_center[i:i+1]
                ground_truth_bath=ground_truth[i:i+1]
                img_name=view_manager.view_list[i].image_name

            ### render ###
            tile_img,tile_transmitance=gs_model.render(view_matrix_batch,view_project_matrix_batch,camera_focal_batch,camera_center_batch)
            img=tiles2img_torch(tile_img,gs_model.cached_tiles_size[0],gs_model.cached_tiles_size[1])[...,:view_manager.image_size[1],:view_manager.image_size[0]]
            #transmitance=tiles2img_torch(tile_transmitance,self.model.cached_tiles_size[0],self.model.cached_tiles_size[1])[...,:self.image_size[1],:self.image_size[0]]
            callback_result_list.append(iter_callback(i,img_name,img,ground_truth_bath))

        return callback_result_list

    @torch.no_grad()
    def report_psnr(self,epoch_i:int):
        def get_psnr_internal(iter_i:int,img_name:str,out_img:torch.Tensor,ground_truth:torch.Tensor)->torch.Tensor:
            psnr=image_utils.psnr(out_img,ground_truth)
            return psnr
        psnr_list=GaussianTrainer.inference(self.model,self.view_manager,False,get_psnr_internal)
        psnr=torch.concat(psnr_list,dim=0)
        tqdm.write("\n[EPOCH {}] Trainingset Evaluating: PSNR {}".format(epoch_i, psnr.mean()))
        torch.cuda.empty_cache()

        if self.view_manager_testset is not None:
            psnr_list=GaussianTrainer.inference(self.model,self.view_manager_testset,False,get_psnr_internal)
            psnr=torch.concat(psnr_list,dim=0)
            tqdm.write("[EPOCH {}] Testingset Evaluating: PSNR {}".format(epoch_i, psnr.mean()))
            torch.cuda.empty_cache()
        return

    def start(self,iterations:int,load_checkpoint:str=None,checkpoint_epochs:typing.List=[],saving_epochs:typing.List=[],test_epochs:typing.List=[]):
        if load_checkpoint is not None:
            self.restore(load_checkpoint)

        batch_size=1

        with torch.no_grad():
            self.model.update_tiles_coord(self.image_size,self.tile_size,batch_size)
            view_matrix=torch.Tensor(self.view_manager.view_matrix_tensor).cuda()
            view_project_matrix=view_matrix@(torch.Tensor(self.view_manager.proj_matrix_tensor).cuda())
            camera_center=torch.Tensor(self.view_manager.camera_center_tensor).cuda()
            camera_focal=torch.Tensor(self.view_manager.camera_focal_tensor).cuda()
            ground_truth=torch.Tensor(self.view_manager.view_gt_tensor).cuda()
            total_views_num=view_matrix.shape[0]

        StatisticsHelperInst.reset(self.model._xyz.shape[-2],self.model._xyz.shape[-1])
        torch.cuda.empty_cache()
        sample_num=len(self.view_manager.view_list)
        epoch=int(math.ceil(iterations/sample_num))
        progress_bar = tqdm(range(self.iter_start*sample_num, epoch*sample_num), desc="Training progress")
        progress_bar.update(0)
        
        for epoch_i in range(self.iter_start,epoch):
            if (epoch_i+1)%5==0:
                self.model.oneupSHdegree()

            if StatisticsHelperInst.bStart==False and self.density_controller.IsDensify(epoch_i):
                StatisticsHelperInst.start()
            
            self.__iter(epoch_i,batch_size,view_matrix,view_project_matrix,camera_center,camera_focal,ground_truth)
            progress_bar.update(total_views_num)
            
            if epoch_i in checkpoint_epochs:
                tqdm.write("\n[EPOCH {}] Saving Checkpoint".format(epoch_i))
                self.save(epoch_i)

            if epoch_i in test_epochs:
                self.report_psnr(epoch_i)
            
            if epoch_i in saving_epochs:
                tqdm.write("\n[EPOCH {}] Saving Gaussians".format(epoch_i))
                scene=GaussianScene()
                self.model.save_to_scene(scene)
                dir=os.path.join(self.output_path,"point_cloud/iteration_{}".format(epoch_i))
                scene.save_ply(os.path.join(dir,"point_cloud.ply"))

            self.density_controller.step(self.model,self.optimizer,epoch_i)

        #finish
        progress_bar.close()
        scene=GaussianScene()
        self.save(epoch)
        self.model.save_to_scene(scene)
        dir=os.path.join(self.output_path,"point_cloud/finish")
        scene.save_ply(os.path.join(dir,"point_cloud.ply"))

        return
    