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
from ..data import FramesBuffer


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
                norm_radius, dp, self.cluster_size, cur_points_num
            )
        else:
            self.density_controller=None

        self.active_sh_degree = active_sh_degree
        self.spatial_rearrange()

        return

    @torch.no_grad()
    def apply_densify_edits(self, edits: densify.DensifyEdits) -> None:
        if self.cluster_size > 0:
            xyz, scale, rot, sh_0, sh_rest, opacity = scene.cluster.uncluster(
                self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity
            )
        else:
            xyz = self.xyz
            scale = self.scale
            rot = self.rot
            sh_0 = self.sh_0
            sh_rest = self.sh_rest
            opacity = self.opacity

        if edits.append_params is not None:
            xyz = torch.cat((xyz, edits.append_params.xyz), dim=-1)
            scale = torch.cat((scale, edits.append_params.scale), dim=-1)
            rot = torch.cat((rot, edits.append_params.rot), dim=-1)
            sh_0 = torch.cat((sh_0, edits.append_params.sh_0), dim=-1)
            sh_rest = torch.cat((sh_rest, edits.append_params.sh_rest), dim=-1)
            opacity = torch.cat((opacity, edits.append_params.opacity), dim=-1)
        
        if edits.opacity_override is not None:
            opacity = edits.opacity_override

        if edits.prune_index is not None:
            keep_mask=torch.ones(xyz.shape[-1],dtype=torch.bool,device=xyz.device)
            keep_mask[edits.prune_index]=False
            xyz = xyz[..., keep_mask]
            scale = scale[..., keep_mask]
            rot = rot[..., keep_mask]
            sh_0 = sh_0[..., keep_mask]
            sh_rest = sh_rest[..., keep_mask]
            opacity = opacity[..., keep_mask]


        if self.cluster_size > 0:
            xyz, scale, rot, sh_0, sh_rest, opacity = scene.cluster.cluster_points(
                self.cluster_size,
                xyz, scale, rot, sh_0, sh_rest, opacity
            )

        #p.s. Create new nn.Parameter instead of assigning to .data. We change the shape of nn.Parameter so we need a new nn.Parameter to create new autograd graph!
        exchange_dict={}
        new_param = nn.Parameter(xyz)
        exchange_dict[self.xyz]=new_param
        self.xyz = new_param
        new_param = nn.Parameter(scale)
        exchange_dict[self.scale]=new_param
        self.scale = new_param
        new_param = nn.Parameter(rot)
        exchange_dict[self.rot]=new_param
        self.rot = new_param
        new_param = nn.Parameter(sh_0)
        exchange_dict[self.sh_0]=new_param
        self.sh_0 = new_param
        new_param = nn.Parameter(sh_rest)
        exchange_dict[self.sh_rest]=new_param
        self.sh_rest = new_param
        new_param = nn.Parameter(opacity)
        exchange_dict[self.opacity]=new_param
        self.opacity = new_param
        self.update_cluster_aabb()
        return exchange_dict


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
        assert(self.cluster_size > 0)
        cluster_origin, cluster_extend = scene.cluster.get_cluster_AABB(
            self.xyz, self.scale.exp(), nn.functional.normalize(self.rot, dim=0)
        )
        self.cluster_origin = cluster_origin
        self.cluster_extend = cluster_extend
        return

    @torch.no_grad()
    def spatial_rearrange(self) -> None:

        assert(self.xyz.grad is None)

        indices = scene.spatial_refine(self.cluster_size > 0, self.xyz)

        if self.cluster_size > 0:
            xyz, scale, rot, sh_0, sh_rest, opacity = scene.cluster.uncluster(
                self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity
            )
            xyz = xyz[..., indices]
            scale = scale[..., indices]
            rot = rot[..., indices]
            sh_0 = sh_0[..., indices]
            sh_rest = sh_rest[..., indices]
            opacity = opacity[..., indices]
            xyz, scale, rot, sh_0, sh_rest, opacity = scene.cluster.cluster_points(
                self.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity
            )
        else:
            xyz = self.xyz.data[..., indices].contiguous()
            scale = self.scale.data[..., indices].contiguous()
            rot = self.rot.data[..., indices].contiguous()
            sh_0 = self.sh_0.data[..., indices].contiguous()
            sh_rest = self.sh_rest.data[..., indices].contiguous()
            opacity = self.opacity.data[..., indices].contiguous()

        self.xyz.data = xyz
        self.scale.data = scale
        self.rot.data = rot
        self.sh_0.data = sh_0
        self.sh_rest.data = sh_rest
        self.opacity.data = opacity

        if self.optimizer is not None or self.sh_optimizer is not None:
            optimizer_adapter.reorder_states_by_index(
                [self.optimizer, self.sh_optimizer],
                indices,
                self.cluster_size > 0
            )
        self.update_cluster_aabb()
        return

    def densify_step(self, epoch: int):
        """
        Perform density control step (densify, prune, opacity reset).
        Returns updated parameters.
        """
        if self.density_controller is None:
            return
        
        if self.cluster_size > 0:
            xyz, scale, rot, sh_0, sh_rest, opacity=scene.cluster.uncluster(self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity)  
        else:
            xyz, scale, rot, sh_0, sh_rest, opacity=(self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity)  

        edits = self.density_controller.step(densify.GaussianParams(xyz, scale, rot, sh_0, sh_rest, opacity),epoch)
        if edits.changed:
            exchange_dict = self.apply_densify_edits(edits)
            optimizer_adapter.sync_state_after_densify([self.optimizer, self.sh_optimizer],edits,exchange_dict,self.cluster_size > 0)
            
        StatisticsHelperInst.reset(self.xyz.shape[-2], self.xyz.shape[-1],self.density_controller.is_densify_actived)

        if epoch % self.density_controller.densify_params.interval == 0:
            self.spatial_rearrange()

        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree = min(int(epoch / 5), self.sh_degree)

        torch.cuda.empty_cache()
        return

    def forward(
        self,
        view_matrix: torch.Tensor,
        frustumplane: torch.Tensor,
        idx_tensor: torch.Tensor,
        training_frame_buffer:FramesBuffer|None
    ):
        """
        Perform culling + activation (render_preprocess).

        Returns:
            visible_chunkid, visible_chunks_num,  xyz,  scale,  rot, color,  opacity
        """
        
        visible_chunkid=None
        visible_chunks_num=None
        feedback_visible_chunks_num=None
        if self.training:
            feedback_visible_chunks_num=training_frame_buffer.feedback_visible_chunks_num
            
        
        if self.cluster_size>0:
            visibility,visible_chunks_num,visible_chunkid=utils.ops.frustum_culling_aabb(
                self.cluster_origin,self.cluster_extend,frustumplane,
                feedback_visible_chunks_num,idx_tensor
            )
            if StatisticsHelperInst.bStart:
                StatisticsHelperInst.set_compact_mask(visible_chunkid,visible_chunks_num)

            # Step 1: Compact + Activate (without SH)
            xyz, scale, rot, opacity=utils.ops.compact_activate_nosh(
                self.is_sparse_grad,
                visible_chunkid,visible_chunks_num,
                self.xyz,self.scale,self.rot,self.opacity,
                backend=utils.ops.Backend.CUDA
            )

            # Step 2: Compact + SH (using activated position for view direction)
            if hasattr(self.sh_optimizer,'forward'):
                color=self.sh_optimizer.forward(self.active_sh_degree,visible_chunkid,visible_chunks_num,view_matrix,self.xyz)
            else:
                color=utils.ops.compact_sh(
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
            color=utils.ops.spherical_harmonic_to_rgb(self.active_sh_degree,self.sh_0,self.sh_rest,dirs)


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
