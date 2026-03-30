import torch
import torch.nn as nn
from typing import Optional

from .. import arguments
from .. import scene
from .. import io_manager
from .. import render
from ..training import optimizer as opt_module
from ..training import densify
from ..training import optimizer_adapter
from .. import utils
from ..utils.statistic_helper import StatisticsHelperInst


class GaussianSplattingModel(nn.Module):
    """
    Manages Gaussian splatting parameters, clustering, optimizer and scheduler.
    Inherits nn.Module to provide forward() for culling + activation.
    """
    @classmethod
    def from_arrays(
        cls,
        init_xyz: torch.Tensor,init_color: torch.Tensor,
        mp: arguments.ModelParams,
        norm_radius: float,
        op: arguments.OptimizationParams,
        dp: arguments.DensifyParams):

        assert init_xyz.shape[0] == init_color.shape[0]
        xyz, scale, rot, sh_0, sh_rest, opacity =  scene.create_gaussians(init_xyz, init_color, mp.sh_degree)
        active_sh_degree=0
        model=cls(
            xyz, scale, rot, sh_0, sh_rest, opacity,
            active_sh_degree,
            mp,
            norm_radius,op,dp
        )
        return model
    
    @classmethod
    def from_ply(
        cls,
        ply_path:str,
        mp: arguments.ModelParams,
        norm_radius: None|float,#for training
        op: None|arguments.OptimizationParams,#for training
        dp: None|arguments.DensifyParams#for training
    ):
        xyz,scale,rot,sh_0,sh_rest,opacity=io_manager.load_ply(ply_path,mp.sh_degree)
        xyz=torch.Tensor(xyz).cuda()
        scale=torch.Tensor(scale).cuda()
        rot=torch.Tensor(rot).cuda()
        sh_0=torch.Tensor(sh_0).cuda()
        sh_rest=torch.Tensor(sh_rest).cuda()
        opacity=torch.Tensor(opacity).cuda()

        active_sh_degree=mp.sh_degree
        model=cls(
            xyz, scale, rot, sh_0, sh_rest, opacity,
            active_sh_degree,
            mp,
            norm_radius,op,dp
        )

        return model

    def __init__(
        self,
        xyz, scale, rot, sh_0, sh_rest, opacity,
        active_sh_degree:int,
        mp: arguments.ModelParams,
        norm_radius: None|float,#for training
        op: None|arguments.OptimizationParams,#for training
        dp: None|arguments.DensifyParams#for training
    ):
        super().__init__()
        self.cluster_size = mp.cluster_size
        self.sh_degree = mp.sh_degree


        cur_points_num=xyz.shape[-1]
        if self.cluster_size > 0:
            xyz, scale, rot, sh_0, sh_rest, opacity = scene.cluster.cluster_points(self.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity)
            cur_points_num=xyz.shape[-1]*xyz.shape[-2]

        # Register as parameters
        self.xyz = nn.Parameter(xyz)
        self.scale = nn.Parameter(scale)
        self.rot = nn.Parameter(rot)
        self.sh_0 = nn.Parameter(sh_0)
        self.sh_rest = nn.Parameter(sh_rest)
        self.opacity = nn.Parameter(opacity)

        # Create optimizer, scheduler and density controller
        self.optimizer=None
        self.sh_optimizer=None
        self.is_sparse_grad=True
        if op is not None:
            self.optimizer, self.scheduler,self.sh_optimizer = opt_module.get_optimizer(
                self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity,
                norm_radius, op, self.cluster_size>0
            )
            self.is_sparse_grad = op.sparse_grad

        
        if dp is not None:
            self.density_controller = densify.DensityControllerTamingGS(
                norm_radius, dp, self.cluster_size > 0, cur_points_num
            )
        else:
            self.density_controller=None

        self.active_sh_degree = active_sh_degree
        self.spatial_rearrange()

        return

    def get_gaussian_params(self) -> densify.GaussianParams:
        return densify.GaussianParams(
            self.xyz, self.scale, self.rot,
            self.sh_0, self.sh_rest, self.opacity
        )

    def set_gaussian_params(self, params: densify.GaussianParams) -> None:
        def _as_parameter(tensor: torch.Tensor | nn.Parameter) -> nn.Parameter:
            if isinstance(tensor, nn.Parameter):
                return tensor
            return nn.Parameter(tensor.requires_grad_(True))

        self.xyz = _as_parameter(params.xyz)
        self.scale = _as_parameter(params.scale)
        self.rot = _as_parameter(params.rot)
        self.sh_0 = _as_parameter(params.sh_0)
        self.sh_rest = _as_parameter(params.sh_rest)
        self.opacity = _as_parameter(params.opacity)
        return

    @torch.no_grad()
    def apply_densify_edits(self, edits: densify.DensifyEdits) -> None:
        params = self.get_gaussian_params()
        if self.cluster_size > 0:
            xyz, scale, rot, sh_0, sh_rest, opacity = scene.cluster.uncluster(
                params.xyz, params.scale, params.rot, params.sh_0, params.sh_rest, params.opacity
            )
            params = densify.GaussianParams(xyz, scale, rot, sh_0, sh_rest, opacity)

        if edits.append_params is not None:
            params = densify.DensityControllerBase._concat_params(params, edits.append_params)
        if edits.keep_mask is not None:
            params = densify.DensityControllerBase._slice_params(params, edits.keep_mask)
        if edits.opacity_override is not None:
            params = densify.DensityControllerBase._replace_opacity(params, edits.opacity_override)

        if self.cluster_size > 0:
            xyz, scale, rot, sh_0, sh_rest, opacity = scene.cluster.cluster_points(
                self.cluster_size,
                params.xyz, params.scale, params.rot, params.sh_0, params.sh_rest, params.opacity
            )
            params = densify.GaussianParams(xyz, scale, rot, sh_0, sh_rest, opacity)

        self.set_gaussian_params(params)
        self.update_cluster_aabb()
        return

    @torch.no_grad()
    def sync_optimizers_after_densify(self, edits: densify.DensifyEdits) -> None:
        optimizer_list = [self.optimizer, self.sh_optimizer]
        if edits.append_params is not None:
            optimizer_adapter.append_(optimizer_list, edits.append_params, self.cluster_size > 0)
        if edits.keep_mask is not None:
            optimizer_adapter.prune_(optimizer_list, edits.keep_mask, self.cluster_size > 0)
        if edits.opacity_override is not None:
            optimizer_adapter.replace_(optimizer_list, "opacity", edits.opacity_override, self.cluster_size > 0)
        if edits.clear_optimizer_state:
            for optimizer in optimizer_list:
                if optimizer is not None:
                    optimizer.state.clear()

        params_dict = optimizer_adapter.collect_named_parameters(optimizer_list)
        self.set_gaussian_params(densify.GaussianParams(
            params_dict["xyz"],
            params_dict["scale"],
            params_dict["rot"],
            params_dict["sh_0"],
            params_dict["sh_rest"],
            params_dict["opacity"],
        ))
        self.update_cluster_aabb()
        return

    def state_dict(self,destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to include optimizer, scheduler, and other needed states.
        This allows saving/loading like a regular nn.Module.
        """
        # Call parent's state_dict for parameters (xyz, scale, rot, sh_0, sh_rest, opacity)
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Add scalar value
        state[prefix + 'active_sh_degree'] = torch.tensor(self.active_sh_degree)

        # Add optimizer state
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            state[prefix + 'optimizer_state_dict'] = self.optimizer.state_dict()
        if hasattr(self, 'sh_optimizer') and self.sh_optimizer is not None:
            state[prefix + 'sh_optimizer_state_dict'] = self.sh_optimizer.state_dict()

        # Add scheduler state
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            state[prefix + 'scheduler_state_dict'] = self.scheduler.state_dict()

        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Override load_state_dict to handle optimizer, scheduler, and other needed states.
        This allows loading like a regular nn.Module.
        """
        # Extract scalar value before loading
        active_sh_degree = state_dict.pop('active_sh_degree').item() if 'active_sh_degree' in state_dict else 0

        # Extract optimizer and scheduler states
        optimizer_state = state_dict.pop('optimizer_state_dict', None)
        sh_optimizer_state = state_dict.pop('sh_optimizer_state_dict', None)
        scheduler_state = state_dict.pop('scheduler_state_dict', None)

        # Load parameters using parent's load_state_dict
        super().load_state_dict(state_dict, strict=strict)

        # Restore scalar value
        self.active_sh_degree = active_sh_degree

        # Restore optimizer state
        if optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if sh_optimizer_state is not None and self.sh_optimizer is not None:
            self.sh_optimizer.load_state_dict(sh_optimizer_state)

        # Restore scheduler state
        if scheduler_state is not None and hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.update_cluster_aabb()

        return torch.nn.modules.module._IncompatibleKeys([], [])

    def get_cluster_aabb(self):
        """Return cluster AABB (origin, extend)."""
        return self.cluster_origin, self.cluster_extend

    def update_cluster_aabb(self) -> None:
        """Update cluster AABB based on current scale and rotation."""
        if self.cluster_size > 0:
            cluster_origin, cluster_extend = scene.cluster.get_cluster_AABB(
                self.xyz, self.scale.exp(), nn.functional.normalize(self.rot, dim=0)
            )
            self.cluster_origin = cluster_origin
            self.cluster_extend = cluster_extend
        return

    @torch.no_grad()
    def spatial_rearrange(self) -> None:
        if self.optimizer is None:
            (
                self.xyz.data, self.scale.data, self.rot.data, 
                self.sh_0.data, self.sh_rest.data, 
                self.opacity.data
            ) = scene.spatial_refine(
                self.cluster_size > 0, 
                None,
                self.xyz, self.scale, self.rot, 
                self.sh_0, self.sh_rest, 
                self.opacity
            )
        else:
            (
                self.xyz, self.scale, self.rot, 
                self.sh_0, self.sh_rest, 
                self.opacity
            ) = scene.spatial_refine(self.cluster_size > 0, [self.optimizer,self.sh_optimizer], self.xyz)
        self.update_cluster_aabb()
        return

    def densify_step(self, epoch: int):
        """
        Perform density control step (densify, prune, opacity reset).
        Returns updated parameters.
        """
        if self.density_controller is None:
            return

        edits = self.density_controller.step(self.get_gaussian_params(), epoch)
        if edits.changed:
            if self.optimizer is None and self.sh_optimizer is None:
                self.apply_densify_edits(edits)
            else:
                self.sync_optimizers_after_densify(edits)

            if edits.stats_need_reset:
                StatisticsHelperInst.reset(
                    self.xyz.shape[-2],
                    self.xyz.shape[-1],
                    self.density_controller.is_densify_actived
                )
                torch.cuda.empty_cache()

        if epoch % self.density_controller.densify_params.interval == 0:
            self.spatial_rearrange()

        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree = min(int(epoch / 5), self.sh_degree)

        return

    def forward(
        self,
        view_matrix: torch.Tensor,
        frustumplane: torch.Tensor,
        idx_tensor: torch.Tensor,
        feedback_visible_chunks_num:torch.Tensor
    ):
        """
        Perform culling + activation (render_preprocess).

        Returns:
            visible_chunkid, visible_chunks_num,  xyz,  scale,  rot, color,  opacity
        """
        
        visible_chunkid=None
        visible_chunks_num=None
        if self.training==False:
            feedback_visible_chunks_num=None
        
        if self.cluster_size>0:
            visibility,visible_chunks_num,visible_chunkid=utils.wrapper.litegs_fused.frustum_culling_aabb(
                self.cluster_origin,self.cluster_extend,frustumplane,
                feedback_visible_chunks_num,idx_tensor
            )
            if StatisticsHelperInst.bStart:
                StatisticsHelperInst.set_compact_mask(visible_chunkid,visible_chunks_num)

            #  xyz, scale, rot,color, opacity=utils.wrapper.CullCompactActivateWithSparseGrad.apply(
            #     pp.sparse_grad,actived_sh_degree,
            #     visible_chunkid,visible_chunks_num,
            #     view_matrix,
            #     xyz,scale,rot,sh_0,sh_rest,opacity
            # )

            # Step 1: Compact + Activate (without SH)
            xyz, scale, rot, opacity=utils.wrapper.CompactActivateNoSH.apply(
                self.is_sparse_grad,
                visible_chunkid,visible_chunks_num,
                self.xyz,self.scale,self.rot,self.opacity
            )

            # Step 2: Compact + SH (using activated position for view direction)
            if hasattr(self.sh_optimizer,'forward'):
                color=self.sh_optimizer.forward(self.active_sh_degree,visible_chunkid,visible_chunks_num,view_matrix,self.xyz)
            else:
                color=utils.wrapper.CompactSH.apply(
                    self.is_sparse_grad,
                    self.active_sh_degree,
                    visible_chunkid,visible_chunks_num,
                    view_matrix,
                    self.xyz,self.sh_0,self.sh_rest
                )

            xyz, scale, rot,color, opacity=scene.cluster.uncluster( xyz, scale, rot,color, opacity)  
        else:
            pad_one=torch.ones((1,self.xyz.shape[-1]),dtype=self.xyz.dtype,device=self.xyz.device)
            xyz=torch.concat((self.xyz,pad_one),dim=0)
            scale=self.scale.exp()
            rot=torch.nn.functional.normalize(self.rot,dim=0)
            opacity=self.opacity.sigmoid()
            with torch.no_grad():
                camera_center=(-view_matrix[...,3:4,:3]@(view_matrix[...,:3,:3].transpose(-1,-2))).squeeze(1)
                dirs= xyz[:3]-camera_center.unsqueeze(-1)
                dirs=torch.nn.functional.normalize(dirs,dim=-2)
            color=utils.wrapper.SphericalHarmonicToRGB.call_fused(self.active_sh_degree,self.sh_0,self.sh_rest,dirs)


        if visible_chunks_num is not None:
            valid_length = visible_chunks_num * self.cluster_size

        return (
            visible_chunkid, visible_chunks_num,valid_length,
            xyz, scale, rot, color, opacity
        )

    def save_ply(self, save_path: str):
        """Save Gaussian parameters to PLY file."""
        if self.cluster_size > 0:
            tensors = scene.cluster.uncluster(
                self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity
            )
        else:
            tensors = (self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity)

        param_nyp = []
        for tensor in tensors:
            param_nyp.append(tensor.detach().cpu().numpy())

        io_manager.save_ply(save_path, *param_nyp)
        return
