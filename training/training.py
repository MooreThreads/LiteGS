from gaussian_splatting.model import GaussianSplattingModel
from training.arguments import OptimizationParams,ModelParams
from gaussian_splatting.gaussian_util import GaussianScene,get_expon_lr_func
from loader.InfoLoader import CameraInfo,ImageInfo,PinHoleCameraInfo
from gaussian_splatting.gaussian_util import View
from training import cache
from gaussian_splatting.gaussian_util import GaussianScene
from training.utils import loss_utils,image_utils

import torch
import typing
from tqdm import tqdm
import numpy as np
import math
import random
from matplotlib import pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
import os

class ViewManager:
    '''
    cache visibility of points in views
    cache tiles info for views
    '''
    def __init__(self,image_list:typing.List[ImageInfo],camera_dict:typing.Dict[int,PinHoleCameraInfo]):
        self.view_list:typing.List[View]=[]
        self.view_matrix_tensor=None
        self.proj_matrix_tensor=None
        self.camera_center_tensor=None

        view_matrix_list=[]
        proj_matrix_list=[]
        camera_center_list=[]
        camera_focal_list=[]
        gt_list=[]
        for img in image_list:
            camera_info=camera_dict[img.camera_id]
            cur_view=View(img.viewtransform_position,img.viewtransform_rotation,camera_info.fovX,camera_info.fovY,np.array(img.image),img.name)
            self.view_list.append(cur_view)
            view_matrix_list.append(np.expand_dims(cur_view.world2view_matrix,0))
            proj_matrix_list.append(np.expand_dims(cur_view.project_matrix,0))
            camera_center_list.append(np.expand_dims(cur_view.camera_center,0))
            camera_focal_list.append(np.expand_dims((cur_view.focal_x,cur_view.focal_y),0))
            gt_list.append(np.expand_dims(cur_view.image,0))
        self.view_matrix_tensor=np.concatenate(view_matrix_list)
        self.proj_matrix_tensor=np.concatenate(proj_matrix_list)
        self.camera_center_tensor=np.concatenate(camera_center_list)
        self.camera_focal_tensor=np.concatenate(camera_focal_list)
        self.view_gt_tensor=np.concatenate(gt_list)/255.0
        self.view_gt_tensor=self.view_gt_tensor.transpose(0,3,1,2)

        self.cache:typing.Dict[int,cache.CachedData]={}
        self.cached_view_index=None
        self.data_generation=0
        return
    
    def update_cached_visibility_info(self,batch_index:int,visible_points_for_views:torch.Tensor,visible_points_num:torch.Tensor):
        if batch_index not in self.cache.keys():
            self.cache[batch_index]=cache.CachedData(batch_index,self.data_generation+1)
        self.cache[batch_index].visible_info=cache.VisibleInfo(batch_index,visible_points_for_views,visible_points_num,self.data_generation+1)
        return
    
    def update_cached_binning_info(self,batch_index:int,tile_start_index:torch.Tensor,sorted_pointId:torch.Tensor,sorted_tileId:torch.Tensor):
        if batch_index not in self.cache.keys():
            self.cache[batch_index]=cache.CachedData(batch_index,self.data_generation+1)
        self.cache[batch_index].binning_info=cache.BinningInfo(batch_index,tile_start_index,sorted_pointId,sorted_tileId,self.data_generation+1)
        return
    
    def update_cache_data_generation(self,cur_view_index:torch.Tensor):
        self.cached_view_index=cur_view_index
        self.data_generation+=1
        return
    
    

class GaussianTrain:

    def __init__(self,gaussian_model:GaussianSplattingModel,lp:ModelParams,op:OptimizationParams,NerfNormRadius:int,image_list:typing.List[ImageInfo],camera_dict:typing.Dict[int,PinHoleCameraInfo]):
        self.spatial_lr_scale=NerfNormRadius
        self.image_list=image_list
        self.camera_dict=camera_dict
        self.view_manager=ViewManager(self.image_list,self.camera_dict)
        self.model=gaussian_model
        self.output_path=lp.model_path
        self.__training_setup(gaussian_model,op)
        self.opt_params=op

        self.image_size=self.image_list[0].image.size
        self.tile_size=16
        self.iter_start=0
        return
    
    def save(self,iteration):
        model_params=self.model.get_params()
        torch.save((model_params,self.optimizer.state_dict(), iteration), self.output_path + "/chkpnt" + str(iteration) + ".pth")
        return
    
    def restore(self,checkpoint):
        (model_params,op_state_dict, first_iter) = torch.load(checkpoint)
        self.model.load_params(model_params)
        self.optimizer.load_state_dict(op_state_dict)
        self.iter_start=first_iter
        return
    
    def __training_setup(self,gaussian_model:GaussianSplattingModel,args:OptimizationParams):
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
        
        self.tb_writer = SummaryWriter(self.output_path)
        return
    
    def __update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    
    def __img_to_tiles(self,img):
        N,H,W,C=img.shape
        H_tile=math.ceil(H/self.tile_size)
        W_tile=math.ceil(W/self.tile_size)
        H_pad=H_tile*self.tile_size-H
        W_pad=W_tile*self.tile_size-W
        pad_img=torch.nn.functional.pad(img,(0,0,0,W_pad,0,H_pad,0,0),'constant',0)
        out=pad_img.reshape(N,H_tile,self.tile_size,W_tile,self.tile_size,C).transpose(2,3).reshape(N,-1,self.tile_size,self.tile_size,C)
        return out
    
    def __flatten_tiles(self,tile_img:torch.Tensor):
        N=tile_img.shape[0]
        C=tile_img.shape[2]
        translated_tile_img=tile_img[:,1:].transpose(1,2).reshape(N,C,self.model.cached_tiles_size[1],self.model.cached_tiles_size[0],self.model.cached_tile_size,self.model.cached_tile_size).transpose(-2,-3)
        img=translated_tile_img.reshape((N,C,self.model.cached_tiles_size[1]*self.model.cached_tile_size,self.model.cached_tiles_size[0]*self.model.cached_tile_size))
        return img
    
    def __iter(self,epoch_i:int,batch_size:int,
                            view_matrix:torch.Tensor,view_project_matrix:torch.Tensor,camera_center:torch.Tensor,camera_focal:torch.Tensor,ground_truth:torch.Tensor):
        
        with torch.no_grad():
            total_views_num=view_matrix.shape[0]
            ndc_pos=self.model.world_to_ndc(self.model._xyz,view_project_matrix)
            translated_pos=self.model.world_to_view(self.model._xyz,view_matrix)
            visible_points_for_views,visible_points_num_for_views=self.model.culling_and_sort(ndc_pos,translated_pos)

            if batch_size > 1:            
                # cluster the views according to the visible_points_num
                visible_points_num_for_views,view_indices=torch.sort(visible_points_num_for_views)
                visible_points_for_views=visible_points_for_views[view_indices]
                view_matrix=view_matrix[view_indices]
                view_project_matrix=view_project_matrix[view_indices]
                camera_focal=camera_focal[view_indices]
                camera_center=camera_center[view_indices]
                ground_truth=ground_truth[view_indices]
        
        log_loss=0
        counter=0
        iter_range=list(range(0,total_views_num,batch_size))
        ssim_helper=loss_utils.LossSSIM().cuda()
        random.shuffle(iter_range)

        ### iter batch ###
        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],use_cuda=True) as prof:
        for i in iter_range:
            batch_tail=min(i+batch_size,total_views_num)

            ### gather batch data ###
            with torch.no_grad():
                visible_points_num_batch=visible_points_num_for_views[i:batch_tail]
                max_points_in_batch=visible_points_num_batch.max()
                visible_points_for_views_batch=visible_points_for_views[i:batch_tail,:max_points_in_batch]
                view_matrix_batch=view_matrix[i:batch_tail]
                view_project_matrix_batch=view_project_matrix[i:batch_tail]
                camera_focal_batch=camera_focal[i:batch_tail]
                camera_center_batch=camera_center[i:batch_tail]
                ground_truth_batch=ground_truth[i:batch_tail]

            visible_scales,visible_rotators,visible_positions,visible_opacities,visible_sh0=self.model.sample_by_visibility(visible_points_for_views_batch,visible_points_num_batch)

            ### (scale,rot)->3d covariance matrix->2d covariance matrix ###
            cov3d,transform_matrix=self.model.transform_to_cov3d_faster(visible_scales,visible_rotators)
            visible_cov2d=self.model.proj_cov3d_to_cov2d(cov3d,visible_positions,view_matrix_batch,camera_focal_batch)
            
            ### color ###
            SH_C0 = 0.28209479177387814
            visible_color=(visible_sh0*SH_C0+0.5).squeeze(2).clamp_min(0)
            
            ### mean of 2d-gaussian ###
            ndc_pos_batch=self.model.world_to_ndc(visible_positions,view_project_matrix_batch)
            
            #### binning ###
            tile_start_index,sorted_pointId,sorted_tileId,tiles_touched=self.model.binning(ndc_pos_batch,visible_cov2d,visible_points_num_batch)

            #### raster ###
            tile_img,tile_transmitance=self.model.raster(ndc_pos_batch,visible_cov2d,visible_color,visible_opacities,tile_start_index,sorted_pointId,sorted_tileId)
            img=self.__flatten_tiles(tile_img)[:,:,0:self.image_size[1],0:self.image_size[0]]
            
            #### loss ###
            l1_loss=loss_utils.l1_loss(img,ground_truth_batch)
            regularization_loss=(1-visible_opacities).mean()*0.01+visible_scales.var(2).mean()*100
            ssim_loss=ssim_helper.loss(img,ground_truth_batch)
            loss=(1.0-self.opt_params.lambda_dssim)*l1_loss+self.opt_params.lambda_dssim*(1-ssim_loss) +regularization_loss
            loss.backward()
            log_loss+=l1_loss.detach()
            counter+=1

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)

        #print(prof.key_averages().table(sort_by="self_cuda_time_total"))

        ### log ###
        log_loss/=counter
        self.tb_writer.add_scalar('loss',log_loss.cpu(),epoch_i)
        if epoch_i%10==0:
            with torch.no_grad():
                log_img_batch=img.cpu().numpy()
                log_transmitance=self.__flatten_tiles(tile_transmitance.unsqueeze(2))[:,0:self.image_size[1],0:self.image_size[0],:].cpu().numpy()
                log_groundtruth=ground_truth_batch.cpu().numpy()
            self.tb_writer.add_image('render/image',log_img_batch,epoch_i,dataformats="NCHW")
            self.tb_writer.add_image('render/transmitance',log_transmitance,epoch_i,dataformats="NCHW")
            self.tb_writer.add_image('render/gt',log_groundtruth,epoch_i,dataformats="NCHW")
        return

    @torch.no_grad
    def interface(self):
        self.model.update_tiles_coord(self.image_size,self.tile_size)
        view_matrix=torch.Tensor(self.view_manager.view_matrix_tensor).cuda()
        view_project_matrix=view_matrix@(torch.Tensor(self.view_manager.proj_matrix_tensor).cuda())
        camera_center=torch.Tensor(self.view_manager.camera_center_tensor).cuda()
        camera_focal=torch.Tensor(self.view_manager.camera_focal_tensor).cuda()
        total_views_num=view_matrix.shape[0]

        total_views_num=view_matrix.shape[0]
        ndc_pos=self.model.world_to_ndc(self.model._xyz,view_project_matrix)
        translated_pos=self.model.world_to_view(self.model._xyz,view_matrix)
        visible_points_for_views,visible_points_num_for_views=self.model.culling_and_sort(ndc_pos,translated_pos)

        img_list=[]

        for i in range(total_views_num):

            ### gather batch data ###
            with torch.no_grad():
                visible_points_num_batch=visible_points_num_for_views[i:i+1]
                max_points_in_batch=visible_points_num_batch.max()
                visible_points_for_views_batch=visible_points_for_views[i:i+1,:max_points_in_batch]
                view_matrix_batch=view_matrix[i:i+1]
                view_project_matrix_batch=view_project_matrix[i:i+1]
                camera_focal_batch=camera_focal[i:i+1]
                camera_center_batch=camera_center[i:i+1]

            visible_scales,visible_rotators,visible_positions,visible_opacities,visible_sh0=self.model.sample_by_visibility(visible_points_for_views_batch,visible_points_num_batch)

            ### (scale,rot)->3d covariance matrix->2d covariance matrix ###
            cov3d,transform_matrix=self.model.transform_to_cov3d_faster(visible_scales,visible_rotators)
            visible_cov2d=self.model.proj_cov3d_to_cov2d(cov3d,visible_positions,view_matrix_batch,camera_focal_batch)
            
            ### color ###
            SH_C0 = 0.28209479177387814
            visible_color=(visible_sh0*SH_C0+0.5).squeeze(2).clamp_min(0)
            
            ### mean of 2d-gaussian ###
            ndc_pos_batch=self.model.world_to_ndc(visible_positions,view_project_matrix_batch)
            
            #### binning ###
            tile_start_index,sorted_pointId,sorted_tileId,tiles_touched=self.model.binning(ndc_pos_batch,visible_cov2d,visible_points_num_batch)

            #### raster ###
            tile_img,tile_transmitance=self.model.raster(ndc_pos_batch,visible_cov2d,visible_color,visible_opacities,tile_start_index,sorted_pointId,sorted_tileId)
            img=self.__flatten_tiles(tile_img)[:,:,0:self.image_size[1],0:self.image_size[0]]
            img_list.append(img)
 
        return img_list

    @torch.no_grad
    def report_psnr(self,epoch_i):
        out_img_list=self.interface()
        img=torch.concat(out_img_list,dim=0)
        ground_truth=torch.Tensor(self.view_manager.view_gt_tensor).cuda()
        psnr=image_utils.psnr(img,ground_truth)

        print("\n[EPOCH {}] Evaluating: PSNR {}".format(epoch_i, psnr.mean()))
        return

    def start(self,epoch:int,load_checkpoint:str=None,checkpoint_epochs:typing.List=[],saving_epochs:typing.List=[]):
        if load_checkpoint is not None:
            self.restore(load_checkpoint)

        self.report_psnr(0)

        with torch.no_grad():
            self.model.update_tiles_coord(self.image_size,self.tile_size)
            view_matrix=torch.Tensor(self.view_manager.view_matrix_tensor).cuda()
            view_project_matrix=view_matrix@(torch.Tensor(self.view_manager.proj_matrix_tensor).cuda())
            camera_center=torch.Tensor(self.view_manager.camera_center_tensor).cuda()
            camera_focal=torch.Tensor(self.view_manager.camera_focal_tensor).cuda()
            ground_truth=torch.Tensor(self.view_manager.view_gt_tensor).cuda()
            total_views_num=view_matrix.shape[0]

        progress_bar = tqdm(range(0, epoch*self.view_manager.view_matrix_tensor.shape[0]), desc="Training progress")
        progress_bar.update(0)

        for epoch_i in range(self.iter_start+1,epoch+1):
            
            batch_size=2
            self.__iter(epoch_i,batch_size,view_matrix,view_project_matrix,camera_center,camera_focal,ground_truth)
            progress_bar.update(total_views_num)

            
            if epoch_i in checkpoint_epochs:
                print("\n[ITER {}] Saving Checkpoint".format(epoch_i))
                self.save(epoch_i)
            if epoch_i in saving_epochs:
                print("\n[ITER {}] Saving Gaussians".format(epoch_i))
                self.report_psnr(epoch_i)
                scene=GaussianScene()
                self.model.save_to_scene(scene)
                dir=os.path.join(self.output_path,"point_cloud/iteration_{}".format(epoch_i))
                scene.save_ply(os.path.join(dir,"point_cloud.ply"))
            
        return
    