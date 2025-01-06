from gaussian_splatting.model import GaussianSplattingModel
import torch
import math
import numpy as np
from util.statistic_helper import StatisticsHelperInst,StatisticsHelper
from util.cg_torch import quaternion_to_rotation_matrix
from training.arguments import OptimizationParams
from util.platform import platform_torch_compile

class DensityControllerBase:
    def __init__(self) -> None:
        return
    
    def IsDensify(self,epoch_i:int,iter_i:int)->bool:
        return False

    def _cat_tensors_to_optimizer(self, tensors_dict:dict,optimizer:torch.optim.Optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=-2).contiguous()
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=-2).contiguous()

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=-2).contiguous().requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=-2).contiguous().requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def _prune_chunk_optimizer(self,valid_mask:torch.Tensor,optimizer:torch.optim.Optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][valid_mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][valid_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _fix_model_parameters(self,gaussian_model:GaussianSplattingModel,optimizable_tensors:dict):
        gaussian_model._xyz = optimizable_tensors["xyz"]
        gaussian_model._features_dc = optimizable_tensors["f_dc"]
        gaussian_model._features_rest = optimizable_tensors["f_rest"]
        gaussian_model._opacity = optimizable_tensors["opacity"]
        gaussian_model._scaling = optimizable_tensors["scaling"]
        gaussian_model._rotation = optimizable_tensors["rotation"]
        return
    
    def _replace_tensor_to_optimizer(self, tensor:torch.Tensor, name:str,optimizer:torch.optim.Optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def _replace_params_to_optimizer(self, tensor_dict:dict[str,torch.nn.Parameter],optimizer:torch.optim.Optimizer):
        for group in optimizer.param_groups:
            if group["name"] in tensor_dict.keys():
                tensor=tensor_dict[group["name"]]
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del optimizer.state[group['params'][0]]
                    group["params"][0] = tensor
                    optimizer.state[group['params'][0]] = stored_state
        return

    def update_optimizer_and_model(self,gaussian_model:GaussianSplattingModel,optimizer:torch.optim.Optimizer,valid_chunks_mask:torch.Tensor=None,dict_clone:dict=None,dict_split:dict=None):
        gen_chunks_num=0
        if valid_chunks_mask is not None:
            optimizable_tensors=self._prune_chunk_optimizer(valid_chunks_mask,optimizer)
        if dict_clone is not None:
            optimizable_tensors=self._cat_tensors_to_optimizer(dict_clone,optimizer)
            gen_chunks_num+=dict_clone['xyz'].shape[-2]
        if dict_split is not None:
            optimizable_tensors=self._cat_tensors_to_optimizer(dict_split,optimizer)
            gen_chunks_num+=dict_split['xyz'].shape[-2]
        self._fix_model_parameters(gaussian_model,optimizable_tensors)
        gaussian_model.build_AABB_for_additional_chunks(gen_chunks_num,valid_chunks_mask)
        return

class DenityControlParams:
    def __init__(self, opt_params:OptimizationParams,sample_num:int):
        self.densification_interval = opt_params.densification_interval
        self.opacity_reset_interval = opt_params.opacity_reset_interval
        self.prune_interval=opt_params.prune_interval
        
        self.densify_from_iter = int(opt_params.densify_from_iter/sample_num)
        self.densify_until_iter = int(math.ceil(int(math.ceil(opt_params.densify_until_iter/sample_num))/opt_params.opacity_reset_interval)*opt_params.opacity_reset_interval+1)
        self.densify_grad_threshold = opt_params.densify_grad_threshold
        return

class DensityControllerOfficial(DensityControllerBase):
    @torch.no_grad()
    def __init__(self,grad_threshold, min_opacity, max_screen_size,percent_dense,view_matrix:torch.Tensor,opt_params:OptimizationParams)->None:
        self.grad_threshold=grad_threshold
        self.min_opacity=min_opacity
        self.max_screen_size=max_screen_size
        self.percent_dense=percent_dense
        self.opt_params=DenityControlParams(opt_params,view_matrix.shape[0])#convert iter -> epoch 

        #getNerfppNorm
        camera_pos=view_matrix.inverse()[:,3,:3]
        scene_center=camera_pos.mean(dim=0)
        dist = (scene_center - camera_pos).norm(dim=-1)
        diagonal = dist.max()
        translate=-scene_center
        radius=diagonal * 1.1

        self.screen_extent=radius.cpu()
        return
    
    @torch.no_grad()
    def prune(self,gaussian_model:GaussianSplattingModel)->torch.Tensor:
        fully_trans = (gaussian_model._opacity.sigmoid() < self.min_opacity).squeeze()
        invisible = (StatisticsHelperInst.visible_count==0)
        invisible.shape[0]
        prune_mask=fully_trans
        prune_mask[:invisible.shape[0]]|=invisible
        if self.max_screen_size:
            big_points_vs = StatisticsHelperInst.get_max('radii') > self.max_screen_size
            big_points_ws = gaussian_model._scaling.exp().max(dim=0).values > 0.1 * self.screen_extent
            prune_mask = prune_mask | big_points_ws
            prune_mask[:big_points_vs.shape[0]]|=big_points_vs
        return prune_mask

    @torch.no_grad()
    def densify_and_clone(self,gaussian_model:GaussianSplattingModel)->torch.Tensor:
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
        abnormal_mask = mean2d_grads >= self.grad_threshold
        tiny_pts_mask = gaussian_model._scaling.exp().max(dim=0).values <= self.percent_dense*self.screen_extent
        selected_pts_mask = abnormal_mask&tiny_pts_mask
        return selected_pts_mask
    
    @torch.no_grad()
    def densify_and_split(self,gaussian_model:GaussianSplattingModel,N=2)->torch.Tensor:
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
        abnormal_mask = mean2d_grads >= self.grad_threshold
        large_pts_mask = gaussian_model._scaling.exp().max(dim=0).values > self.percent_dense*self.screen_extent
        selected_pts_mask=abnormal_mask&large_pts_mask
        return selected_pts_mask
    
    @torch.no_grad()
    def prune_and_rebuild_chunk(self,gaussian_model:GaussianSplattingModel,optimizer:torch.optim.Optimizer):
        
        def __prune_torch_parameter(tensor:torch.nn.Parameter,prune_mask:torch.Tensor,padding_value:float):
            chunk_size=tensor.shape[-1]
            
            tensor_data=tensor.data.reshape(*tensor.shape[:-2],tensor.shape[-1]*tensor.shape[-2])
            pruned_data=tensor_data[...,~prune_mask]
            
            #padding
            padding_points_num=math.ceil(pruned_data.shape[-1]/chunk_size)*chunk_size-pruned_data.shape[-1]
            padding_data=torch.ones((*tensor.shape[:-2],padding_points_num),device=pruned_data.device)*padding_value
            pruned_data=torch.cat((pruned_data,padding_data),dim=-1)

            pruned_data_chunk=pruned_data.reshape(*pruned_data.shape[:-1],int(pruned_data.shape[-1]/chunk_size),chunk_size)
            tensor.data=pruned_data_chunk
            return
        
        prune_mask=self.prune(gaussian_model).reshape(-1)
        __prune_torch_parameter(gaussian_model._xyz,prune_mask,1.0)
        __prune_torch_parameter(gaussian_model._scaling,prune_mask,-5.0)
        __prune_torch_parameter(gaussian_model._rotation,prune_mask,0.0)
        __prune_torch_parameter(gaussian_model._features_dc,prune_mask,0.0)
        __prune_torch_parameter(gaussian_model._features_rest,prune_mask,0.0)
        __prune_torch_parameter(gaussian_model._opacity,prune_mask,-10)#sigmoid(opacity)<1/255. make padding points invalid
        #print("\nprune_num:{0} cur_points_num:{1}".format(prune_mask.sum().cpu(),gaussian_model._xyz.shape[-1]*gaussian_model._xyz.shape[-2]))
        gaussian_model.rebuild_chunk_morton(gaussian_model.chunk_size)

        params_dict={
                    "xyz":gaussian_model._xyz,
                    "f_dc":gaussian_model._features_dc,
                    "f_rest":gaussian_model._features_rest,
                    "opacity":gaussian_model._opacity,
                    "scaling":gaussian_model._scaling,
                    "rotation":gaussian_model._rotation
                    }
        self._replace_params_to_optimizer(params_dict,optimizer)
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def split_and_clone(self,gaussian_model:GaussianSplattingModel,optimizer:torch.optim.Optimizer):
        
        # valid_points_mask=None
        # if bPrune==True:
        #     prune_mask=self.prune(gaussian_model)
        #     valid_points_mask=~prune_mask

        clone_mask=self.densify_and_clone(gaussian_model)
        split_mask=self.densify_and_split(gaussian_model)

        def merge_chunk(tensor:torch.Tensor):
            return tensor.reshape(*tensor.shape[:-2],tensor.shape[-1]*tensor.shape[-2])
        features_dc_data=merge_chunk(gaussian_model._features_dc.data)
        features_rest_data=merge_chunk(gaussian_model._features_rest.data)
        opacity_data=merge_chunk(gaussian_model._opacity.data)
        rotation_data=merge_chunk(gaussian_model._rotation.data)
        scaling_data=merge_chunk(gaussian_model._scaling.data)
        xyz_data=merge_chunk(gaussian_model._xyz.data)
        clone_mask=clone_mask.reshape(-1)
        split_mask=split_mask.reshape(-1)

        #split
        stds=scaling_data[...,split_mask].exp()
        means=torch.zeros((3,stds.size(-1)),device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(1)
        rotation_matrix=quaternion_to_rotation_matrix(torch.nn.functional.normalize(rotation_data[...,split_mask],dim=0))
        shift=rotation_matrix.permute(2,0,1)@(samples.permute(2,0,1))
        shift=shift.permute(1,2,0).squeeze(1)
        
        split_xyz=xyz_data[...,split_mask]
        split_xyz[:3]+=shift
        clone_xyz=xyz_data[...,clone_mask]
        xyz_data[:3,split_mask]-=shift
        
        split_scaling = (scaling_data[...,split_mask].exp() / (0.8*2)).log()
        clone_scaling = scaling_data[...,clone_mask]
        scaling_data[...,split_mask]=split_scaling

        split_rotation=rotation_data[...,split_mask]
        clone_rotation=rotation_data[...,clone_mask]

        split_features_dc=features_dc_data[...,split_mask]
        clone_features_dc=features_dc_data[...,clone_mask]

        split_features_rest=features_rest_data[...,split_mask]
        clone_features_rest=features_rest_data[...,clone_mask]

        split_opacity=opacity_data[...,split_mask]
        clone_opacity=opacity_data[...,clone_mask]

        def devide_into_chunks(tensor:torch.Tensor,chunk_size:int):
            N=tensor.shape[-1]
            chunk_num=int(N/chunk_size)
            tensor=tensor[...,:chunk_num*chunk_size].reshape(*tensor.shape[:-1],chunk_num,chunk_size)
            return tensor

        dict_clone = {"xyz": devide_into_chunks(torch.cat((split_xyz,clone_xyz),dim=-1),gaussian_model.chunk_size),
        "opacity": devide_into_chunks(torch.cat((split_opacity,clone_opacity),dim=-1),gaussian_model.chunk_size),
        "scaling" : devide_into_chunks(torch.cat((split_scaling,clone_scaling),dim=-1),gaussian_model.chunk_size),
        "f_dc": devide_into_chunks(torch.cat((split_features_dc,clone_features_dc),dim=-1),gaussian_model.chunk_size),
        "f_rest": devide_into_chunks(torch.cat((split_features_rest,clone_features_rest),dim=-1),gaussian_model.chunk_size),
        "rotation" : devide_into_chunks(torch.cat((split_rotation,clone_rotation),dim=-1),gaussian_model.chunk_size)}
        
        self.update_optimizer_and_model(gaussian_model,optimizer,None,dict_clone,None)
        #print("\nclone_num:{0} split_num:{1} cur_points_num:{2}".format(clone_mask.sum().cpu(),split_mask.sum().cpu(),gaussian_model._xyz.shape[-1]*gaussian_model._xyz.shape[-2]))
        torch.cuda.empty_cache()
        return
    
    @torch.no_grad()
    def reset_opacity(self,gaussian_model:GaussianSplattingModel):
        def inverse_sigmoid(x):
            return torch.log(x/(1-x))
        actived_opacities=gaussian_model._opacity.sigmoid()
        decay_rate=0.5
        decay_mask=(actived_opacities>1/(255*decay_rate-1))
        decay_rate=decay_mask*decay_rate+(~decay_mask)*1.0
        gaussian_model._opacity.data=inverse_sigmoid(actived_opacities*decay_rate)#(actived_opacities.clamp_max(0.005))
        return

    @torch.no_grad()
    def step(self,gaussian_model:GaussianSplattingModel,optimizer:torch.optim.Optimizer,epoch_i:int):
        if self.IsDensify(epoch_i)==True:
            bResetOpacity=(epoch_i%self.opt_params.opacity_reset_interval==0)
            bPrune=(epoch_i%self.opt_params.prune_interval==0)
            self.split_and_clone(gaussian_model,optimizer)
            if bPrune:
                self.prune_and_rebuild_chunk(gaussian_model,optimizer)
            if bResetOpacity:
                self.reset_opacity(gaussian_model)
            StatisticsHelperInst.reset(gaussian_model._xyz.shape[-2],gaussian_model._xyz.shape[-1])
        
        if StatisticsHelperInst.bStart==False and self.IsDensify(epoch_i+1)==True:
            StatisticsHelperInst.start()
        return
    
    def IsDensify(self,epoch_i:int)->bool:
        bDensify=epoch_i >= self.opt_params.densify_from_iter and epoch_i<self.opt_params.densify_until_iter and epoch_i % self.opt_params.densification_interval == 0
        return bDensify
    