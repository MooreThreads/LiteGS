from gaussian_splatting.model import GaussianSplattingModel
import torch
import numpy as np
from util.statistic_helper import StatisticsHelperInst,StatisticsHelper
from util.cg_torch import quaternion_to_rotation_matrix

class DensityControllerBase:
    def __init__(self) -> None:
        return
    def _cat_tensors_to_optimizer(self, tensors_dict:dict,optimizer:torch.optim.Optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0).contiguous()
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0).contiguous()

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).contiguous().requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).contiguous().requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def _prune_optimizer(self,valid_mask:torch.Tensor,optimizer:torch.optim.Optimizer):
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

class DensityControllerOfficial(DensityControllerBase):
    @torch.no_grad
    def __init__(self,grad_threshold, min_opacity, max_screen_size,percent_dense,view_matrix:torch.Tensor)->None:
        self.grad_threshold=grad_threshold
        self.min_opacity=min_opacity
        self.max_screen_size=max_screen_size
        self.percent_dense=percent_dense

        #getNerfppNorm
        camera_pos=view_matrix.inverse()[:,3,:3]
        scene_center=camera_pos.mean(dim=0)
        dist = (scene_center - camera_pos).norm(dim=-1)
        diagonal = dist.max()
        translate=-scene_center
        radius=diagonal * 1.1

        self.screen_extent=radius.cpu()
        return
    
    @torch.no_grad
    def prune(self,gaussian_model:GaussianSplattingModel):
        prune_mask = (gaussian_model._opacity.sigmoid() < self.min_opacity).squeeze()
        if self.max_screen_size:
            big_points_vs = StatisticsHelperInst.get_max('radii')[:,0] > self.max_screen_size
            big_points_ws = gaussian_model._scaling.exp().max(dim=1).values > 0.1 * self.screen_extent
            prune_mask = prune_mask| big_points_vs | big_points_ws
        return prune_mask

    @torch.no_grad
    def densify_and_clone(self,gaussian_model:GaussianSplattingModel):
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')#?std(gaussian3d.mean.grad) instead of mean(gaussian2d.mean.grad)
        abnormal_mask = mean2d_grads.norm(dim=-1) >= self.grad_threshold
        tiny_pts_mask = gaussian_model._scaling.exp().max(dim=1).values <= self.percent_dense*self.screen_extent
        selected_pts_mask = abnormal_mask&tiny_pts_mask
        return selected_pts_mask
    
    @torch.no_grad
    def densify_and_split(self,gaussian_model:GaussianSplattingModel,N=2):
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
        abnormal_mask = mean2d_grads.norm(dim=-1) >= self.grad_threshold
        large_pts_mask = gaussian_model._scaling.exp().max(dim=1).values > self.percent_dense*self.screen_extent
        selected_pts_mask=abnormal_mask&large_pts_mask
        return selected_pts_mask

    def update_optimizer_and_model(self,gaussian_model:GaussianSplattingModel,optimizer:torch.optim.Optimizer,valid_points_mask:torch.Tensor,dict_clone:dict=None,dict_split:dict=None):
        optimizable_tensors=self._prune_optimizer(valid_points_mask,optimizer)
        optimizable_tensors=self._cat_tensors_to_optimizer(dict_clone,optimizer)
        optimizable_tensors=self._cat_tensors_to_optimizer(dict_split,optimizer)
        self._fix_model_parameters(gaussian_model,optimizable_tensors)
        return
    
    def densify_and_prune(self,gaussian_model:GaussianSplattingModel,optimizer:torch.optim.Optimizer):
        
        prune_mask=self.prune(gaussian_model)

        clone_mask=self.densify_and_clone(gaussian_model)
        
        dict_clone = {"xyz": gaussian_model._xyz[clone_mask],
        "opacity": gaussian_model._opacity[clone_mask],
        "scaling" : gaussian_model._scaling[clone_mask],
        "f_dc": gaussian_model._features_dc[clone_mask],
        "f_rest": gaussian_model._features_rest[clone_mask],
        "rotation" : gaussian_model._rotation[clone_mask]}

        split_mask=self.densify_and_split(gaussian_model)
        N=2
        stds=gaussian_model._scaling[split_mask].exp().repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rotation_matrix=quaternion_to_rotation_matrix(torch.nn.functional.normalize(gaussian_model._rotation[split_mask],dim=-1)).repeat(N,1,1)
        new_xyz = gaussian_model._xyz[split_mask].repeat(N, 1)
        new_xyz[...,:3]+=(rotation_matrix@samples.unsqueeze(-1)).squeeze(-1)
        new_scaling = (gaussian_model._scaling[split_mask].exp() / (0.8*N)).log().repeat(N,1)
        new_rotation = gaussian_model._rotation[split_mask].repeat(N,1)
        new_features_dc = gaussian_model._features_dc[split_mask].repeat(N,1,1)
        new_features_rest = gaussian_model._features_rest[split_mask].repeat(N,1,1)
        new_opacity = gaussian_model._opacity[split_mask].repeat(N,1)
        dict_split = {"xyz": new_xyz,
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "rotation" :new_rotation}
        
        valid_points_mask=(~prune_mask)&(~split_mask)
        self.update_optimizer_and_model(gaussian_model,optimizer,valid_points_mask,dict_clone,dict_split)
        print("\nclone_num:{0} split_num:{1} prune_num:{2} cur_points_num:{3}".format(clone_mask.sum().cpu(),split_mask.sum().cpu(),prune_mask.sum().cpu(),gaussian_model._xyz.shape[0]))
        torch.cuda.empty_cache()
        return
    
    @torch.no_grad
    def reset_opacity(self,gaussian_model:GaussianSplattingModel,optimizer:torch.optim.Optimizer):
        def inverse_sigmoid(x):
            return torch.log(x/(1-x))
        decay_opacities=gaussian_model._opacity.sigmoid().clamp_max(0.01)
        opacities_new = inverse_sigmoid(decay_opacities)
        optimizable_tensors = self._replace_tensor_to_optimizer(opacities_new, "opacity",optimizer)
        self._opacity = optimizable_tensors["opacity"]
        torch.cuda.empty_cache()
        return

