from gaussian_splatting.model import GaussianSplattingModel
from training.arguments import OptimizationParams,ModelParams
from gaussian_splatting.gaussian_util import GaussianScene,get_expon_lr_func
from loader.InfoLoader import CameraInfo,ImageInfo,PinHoleCameraInfo
from gaussian_splatting.gaussian_util import View
from training import cache

import torch
import typing
from tqdm import tqdm
import numpy as np
import math

from matplotlib import pyplot as plt 
from torch.utils.tensorboard import SummaryWriter

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

    @torch.no_grad 
    def __wrap_tile_start_index_to_image(self,tile_start_index:torch.Tensor):
        tile_ele_num=tile_start_index[:,2:]-tile_start_index[:,1:-1]
        wrapped_img=tile_ele_num.reshape((-1,self.model.cached_tiles_size[1],self.model.cached_tiles_size[0]))
        return wrapped_img
    
    
    def __wrap_img_to_tile(self,img):
        N,H,W,C=img.shape
        H_tile=math.ceil(H/self.tile_size)
        W_tile=math.ceil(W/self.tile_size)
        H_pad=H_tile*self.tile_size-H
        W_pad=W_tile*self.tile_size-W
        pad_img=torch.nn.functional.pad(img,(0,0,0,W_pad,0,H_pad,0,0),'constant',0)
        out=pad_img.reshape(N,H_tile,self.tile_size,W_tile,self.tile_size,C).transpose(2,3).reshape(N,-1,self.tile_size,self.tile_size,C)
        return out
    
    def __wrap_tile_img_to_image(self,tile_img:torch.Tensor):
        N=tile_img.shape[0]
        translated_tile_img=tile_img[:,1:].reshape(N,self.model.cached_tiles_size[1],self.model.cached_tiles_size[0],self.model.cached_tile_size,self.model.cached_tile_size,-1).transpose(2,3)
        img=translated_tile_img.reshape((N,self.model.cached_tiles_size[1]*self.model.cached_tile_size,self.model.cached_tiles_size[0]*self.model.cached_tile_size,-1))
        return img
    
    def __iter_update_cache(self,epoch_i:int,batch_size:int,
                            view_matrix:torch.Tensor,view_project_matrix:torch.Tensor,camera_center:torch.Tensor,camera_focal:torch.Tensor,ground_truth:torch.Tensor):
        with torch.no_grad():
            total_views_num=view_matrix.shape[0]
            ndc_pos=self.model.world_to_ndc(self.model._xyz,view_project_matrix)
            translated_pos=self.model.world_to_view(self.model._xyz,view_matrix)
            visible_points_for_views,visible_points_num_for_views=self.model.culling_and_sort(ndc_pos,translated_pos)
            #cluster the views according to the visible_points_num
            visible_points_num_for_views,view_indices=torch.sort(visible_points_num_for_views)
            visible_points_for_views=visible_points_for_views[view_indices]
            view_matrix=view_matrix[view_indices]
            view_project_matrix=view_project_matrix[view_indices]
            camera_focal=camera_focal[view_indices]
            camera_center=camera_center[view_indices]
            ground_truth=ground_truth[view_indices]

        #iter batch
        log_loss=0
        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],use_cuda=True) as prof:
        for i in range(0,total_views_num,batch_size):
            batch_tail=min(i+batch_size,total_views_num)

            #gather batch data
            with torch.no_grad():
                visible_points_num_batch=visible_points_num_for_views[i:batch_tail]
                max_points_in_batch=visible_points_num_batch.max()
                visible_points_for_views_batch=visible_points_for_views[i:batch_tail,:max_points_in_batch]
                view_matrix_batch=view_matrix[i:batch_tail]
                view_project_matrix_batch=view_project_matrix[i:batch_tail]
                camera_focal_batch=camera_focal[i:batch_tail]
                camera_center_batch=camera_center[i:batch_tail]
                ground_truth_batch=ground_truth[i:batch_tail]
                #self.view_manager.update_cached_visibility_info(i,visible_points_for_views_batch,visible_points_num_batch)#cache

            scales,rotators,visible_positions,visible_opacities,visible_sh0=self.model.sample_by_visibility(visible_points_for_views_batch,visible_points_num_batch)

            #cov
            cov3d,transform_matrix=self.model.transform_to_cov3d_faster(scales,rotators)
            visible_cov2d=self.model.proj_cov3d_to_cov2d(cov3d,visible_positions,view_matrix_batch,camera_focal_batch)
            
            #color
            SH_C0 = 0.28209479177387814
            visible_color=(visible_sh0*SH_C0+0.5).squeeze(2).clamp_min(0)
            
            #ndc_pos
            ndc_pos_batch=self.model.world_to_ndc(visible_positions,view_project_matrix_batch)
            
            #binning
            tile_start_index,sorted_pointId,sorted_tileId,tiles_touched=self.model.tile_raster(ndc_pos_batch,visible_cov2d,visible_points_num_batch)
            #self.view_manager.update_cached_binning_info(i,tile_start_index,sorted_pointId,sorted_tileId)

            #raster
            tile_img,tile_transmitance=self.model.pixel_raster_in_tile(ndc_pos_batch,visible_cov2d,visible_color,visible_opacities,tile_start_index,sorted_pointId,sorted_tileId)
            img=self.__wrap_tile_img_to_image(tile_img)[:,0:self.image_size[1],0:self.image_size[0],:]
            
            loss=(img-ground_truth_batch).abs().mean()
            loss.backward()
            log_loss+=loss.detach()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
        #self.view_manager.update_cache_data_generation(view_indices)
        #print(prof.key_averages().table(sort_by="self_cuda_time_total"))

        #log
        log_loss/=total_views_num
        self.tb_writer.add_scalar('loss',log_loss.cpu(),epoch_i)
        with torch.no_grad():
            log_img_batch=img.cpu().numpy()
            log_transmitance=self.__wrap_tile_img_to_image(tile_transmitance)[:,0:self.image_size[1],0:self.image_size[0],:].cpu().numpy()
            log_groundtruth=ground_truth_batch.cpu().numpy()
        self.tb_writer.add_image('render/image',log_img_batch,epoch_i,dataformats="NHWC")
        self.tb_writer.add_image('render/transmitance',log_transmitance,epoch_i,dataformats="NHWC")
        self.tb_writer.add_image('render/gt',log_groundtruth,epoch_i,dataformats="NHWC")
        return
    
    def __iter(self,epoch_i:int,view_matrix:torch.Tensor,project_matrix:torch.Tensor,camera_center:torch.Tensor,camera_focal:torch.Tensor,ground_truth:torch.Tensor):

        with torch.no_grad():
            total_views_num=view_matrix.shape[0]
            view_indices=self.view_manager.cached_view_index.cuda()
            view_matrix=view_matrix[view_indices]
            project_matrix=project_matrix[view_indices]
            camera_focal=camera_focal[view_indices]
            camera_center=camera_center[view_indices]
            ground_truth=ground_truth[view_indices]

        log_loss=0
        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for batch_id,cache_data in self.view_manager.cache.items():
            with torch.no_grad():
                cache_data.check(self.view_manager.data_generation)
                visible_points_for_views_batch=cache_data.visible_info.visible_points.cuda()
                visible_points_num_batch=cache_data.visible_info.visible_points_num.cuda()
                tile_start_index=cache_data.binning_info.start_index.cuda()
                sorted_pointId=cache_data.binning_info.pointId.cuda()
                sorted_tileId=cache_data.binning_info.tileId.cuda()

                batch_size=visible_points_num_batch.shape[0]
                view_matrix_batch=view_matrix[batch_id:batch_id+batch_size]
                project_matrix_batch=project_matrix[batch_id:batch_id+batch_size]
                camera_focal_batch=camera_focal[batch_id:batch_id+batch_size]
                ground_truth_batch=ground_truth[batch_id:batch_id+batch_size]

            visible_cov3d,visible_positions,visible_opacities,visible_sh0=self.model.sample_by_visibility(visible_points_for_views_batch,visible_points_num_batch)
            ndc_pos_batch,_=self.model.worldpose_2_ndc(visible_positions,view_matrix_batch,project_matrix_batch)
            visible_cov2d=self.model.cov2d_after_culling(visible_cov3d,visible_positions,view_matrix_batch,camera_focal_batch)
            SH_C0 = 0.28209479177387814
            visible_color=(visible_sh0*SH_C0+0.5).squeeze(2).clamp_min(0)

            tile_img,tile_transmitance=self.model.pixel_raster_in_tile(ndc_pos_batch,visible_cov2d,visible_color,visible_opacities,tile_start_index,sorted_pointId,sorted_tileId)
            img=self.__wrap_tile_img_to_image(tile_img)[:,0:self.image_size[1],0:self.image_size[0],:]
            
            loss=(img-ground_truth_batch).abs().mean()
            loss.backward()
            log_loss+=loss.detach()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
        #prof.export_chrome_trace("trace_cache.json")
                
        #log
        log_loss/=total_views_num
        self.tb_writer.add_scalar('loss',log_loss.cpu(),epoch_i)
        with torch.no_grad():
            log_img_batch=img.cpu().numpy()
            log_transmitance=self.__wrap_tile_img_to_image(tile_transmitance)[:,0:self.image_size[1],0:self.image_size[0],:].cpu().numpy()
            log_groundtruth=ground_truth_batch.cpu().numpy()
        self.tb_writer.add_image('render/image',log_img_batch,epoch_i,dataformats="NHWC")
        self.tb_writer.add_image('render/transmitance',log_transmitance,epoch_i,dataformats="NHWC")
        self.tb_writer.add_image('render/gt',log_groundtruth,epoch_i,dataformats="NHWC")

        return
            
    def start(self,iteration:int,load_checkpoint:str=None,checkpoint_iterations:typing.List=[],saving_iterations:typing.List=[]):
        if load_checkpoint is not None:
            self.restore(load_checkpoint)

        with torch.no_grad():
            self.model.update_tiles_coord(self.image_size,self.tile_size)
            view_matrix=torch.Tensor(self.view_manager.view_matrix_tensor).cuda()
            view_project_matrix=view_matrix@(torch.Tensor(self.view_manager.proj_matrix_tensor).cuda())
            camera_center=torch.Tensor(self.view_manager.camera_center_tensor).cuda()
            camera_focal=torch.Tensor(self.view_manager.camera_focal_tensor).cuda()
            ground_truth=torch.Tensor(self.view_manager.view_gt_tensor).cuda()
            total_views_num=view_matrix.shape[0]

        progress_bar = tqdm(range(0, iteration*self.view_manager.view_matrix_tensor.shape[0]), desc="Training progress")
        progress_bar.update(0)

        for epoch_i in range(self.iter_start,iteration):
            # if epoch_i%5==0 or epoch_i<30:
            #     self.__iter_update_cache(epoch_i,1,view_matrix,project_matrix,camera_center,camera_focal,ground_truth)
            # else:
            #     self.__iter(epoch_i,view_matrix,project_matrix,camera_center,camera_focal,ground_truth)
            self.__iter_update_cache(epoch_i,1,view_matrix,view_project_matrix,camera_center,camera_focal,ground_truth)
            progress_bar.update(total_views_num)

            if epoch_i in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(epoch_i))
                self.save(epoch_i)
            if epoch_i in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(epoch_i))
                pass
        return
    