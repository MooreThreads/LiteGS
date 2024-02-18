from gaussian_splatting.model import GaussianSplattingModel
from training.arguments import OptimizationParams
from gaussian_splatting.gaussian_util import GaussianScene,get_expon_lr_func
from loader.InfoLoader import CameraInfo,ImageInfo,PinHoleCameraInfo
from gaussian_splatting.gaussian_util import View

import torch
import typing
from tqdm import tqdm
import numpy as np
import math

class ViewManager:
    '''
    cache view&proj matrix
    cache visibility of points in view
    cache sorted points in view
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
            camera_focal_list.append(np.expand_dims(cur_view.focal_x,0))
            gt_list.append(np.expand_dims(cur_view.image,0))
        self.view_matrix_tensor=np.concatenate(view_matrix_list)
        self.proj_matrix_tensor=np.concatenate(proj_matrix_list)
        self.camera_center_tensor=np.concatenate(camera_center_list)
        self.camera_focal_tensor=np.concatenate(camera_focal_list)
        self.view_gt_tensor=np.concatenate(gt_list)
        return
    

class GaussianTrain:
    def __init__(self,gaussian_model:GaussianSplattingModel,op:OptimizationParams,NerfNormRadius:int,image_list:typing.List[ImageInfo],camera_dict:typing.Dict[int,PinHoleCameraInfo]):
        self.spatial_lr_scale=NerfNormRadius
        self.image_list=image_list
        self.camera_dict=camera_dict
        self.view_manager=ViewManager(self.image_list,self.camera_dict)
        self.model=gaussian_model
        self.__training_setup(gaussian_model,op)

        self.image_size=self.image_list[0].image.size
        self.tile_size=32

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
        return
    
    def __update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    def start(self,iteration:int):
        self.model.cuda()
        progress_bar = tqdm(range(0, iteration*self.view_manager.view_matrix_tensor.shape[0]), desc="Training progress")


        self.model.update_tiles_coord(self.image_size,self.tile_size)
        for epoch_i in range(iteration):
            view_matrix=torch.Tensor(self.view_manager.view_matrix_tensor).cuda()
            project_matrix=torch.Tensor(self.view_manager.proj_matrix_tensor).cuda()
            camera_center=torch.Tensor(self.view_manager.camera_center_tensor).cuda()
            camera_focal=torch.Tensor(self.view_manager.camera_focal_tensor).cuda()
            ground_truth=torch.Tensor(self.view_manager.view_gt_tensor).cuda()
            total_views_num=view_matrix.shape[0]

            #if epoch_i%10==0:
            self.model.update_cov3d()
            ndc_pos=self.model.worldpose_2_ndc(self.model._xyz,view_matrix,project_matrix)
            visible_points_for_views,visible_points_num_for_views=self.model.culling_and_sort(ndc_pos)

            #cluster the views according to the visible_points_num
            with torch.no_grad():
                visible_points_num_for_views,view_indices=torch.sort(visible_points_num_for_views)
                visible_points_for_views=visible_points_for_views[view_indices]
                view_matrix=view_matrix[view_indices]
                project_matrix=project_matrix[view_indices]
                camera_focal=camera_focal[view_indices]
                camera_center=camera_center[view_indices]
                ndc_pos=ndc_pos[view_indices]
                ground_truth=ground_truth[view_indices]

            #batch
            batch_size=32
            for i in range(0,total_views_num,batch_size):
                batch_tail=min(i+batch_size,total_views_num)

                with torch.no_grad():
                    visible_points_num_batch=visible_points_num_for_views[i:batch_tail]
                    max_points_in_batch=visible_points_num_batch.max()
                    visible_points_for_views_batch=visible_points_for_views[i:batch_tail,:max_points_in_batch]
                    view_matrix_batch=view_matrix[i:batch_tail]
                    project_matrix_batch=project_matrix[i:batch_tail]
                    camera_focal_batch=camera_focal[i:batch_tail]
                    camera_center_batch=camera_center[i:batch_tail]
                    ground_truth_batch=ground_truth[i:batch_tail]

                    indices_temp=visible_points_for_views_batch.unsqueeze(2).repeat((1,1,4))
                    ndc_pos_batch=ndc_pos[i:batch_tail].gather(1,indices_temp)
                    

                visible_cov3d,visible_positions,visible_opacities,visible_sh0=self.model.sample_by_visibility(visible_points_for_views_batch,visible_points_num_batch)
                cov2d=self.model.cov2d_after_culling(visible_cov3d,visible_positions,
                                           view_matrix_batch,camera_focal_batch)
                
                tile_start_index,sorted_pointId,sorted_tileId=self.model.tile_raster(ndc_pos_batch,cov2d,visible_points_num_batch)
                self.model.pixel_raster_in_tile(ndc_pos_batch,cov2d,tile_start_index,sorted_pointId,sorted_tileId)




                #log
            progress_bar.update(total_views_num)
        return
    