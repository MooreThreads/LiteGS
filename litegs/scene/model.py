import torch
import torch.nn as nn
from typing import Optional

from .. import arguments
from .. import scene
from .. import io_manager
from .. import render
from ..training import optimizer as opt_module
from ..training import densify


class GaussianSplattingModel(nn.Module):
    """
    Manages Gaussian splatting parameters, clustering, optimizer and scheduler.
    Inherits nn.Module to provide forward() for culling + activation.
    """

    def __init__(
        self,
        init_xyz: torch.Tensor,
        init_color: torch.Tensor,
        sh_degree: int,
        norm_radius: float,
        op: arguments.OptimizationParams,
        mp: arguments.ModelParams,
        dp: arguments.DensifyParams,
        init_points_num: int
    ):
        super().__init__()
        self.mp = mp
        self.op = op
        self.init_points_num = init_points_num


        # Initialize from scratch
        xyz, scale, rot, sh_0, sh_rest, opacity = \
            scene.create_gaussians(init_xyz, init_color, sh_degree)

        # Amply clustering if needed
        if mp.cluster_size > 0:
            xyz, scale, rot, sh_0, sh_rest, opacity = \
                scene.cluster.cluster_points(mp.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity)

        # Register as parameters
        self.xyz = nn.Parameter(xyz)
        self.scale = nn.Parameter(scale)
        self.rot = nn.Parameter(rot)
        self.sh_0 = nn.Parameter(sh_0)
        self.sh_rest = nn.Parameter(sh_rest)
        self.opacity = nn.Parameter(opacity)

        # Create optimizer, scheduler and density controller
        self.optimizer, self.scheduler = opt_module.get_optimizer(
            self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity,
            norm_radius, op, mp
        )
        self.density_controller = densify.DensityControllerTamingGS(
            norm_radius, dp, mp.cluster_size > 0, init_points_num
        )

        self.active_sh_degree = 0
        self.update_cluster_aabb()

        return

    def get_params(self):
        """Return all Gaussian parameters."""
        return self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity

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
        scheduler_state = state_dict.pop('scheduler_state_dict', None)

        # Load parameters using parent's load_state_dict
        super().load_state_dict(state_dict, strict=strict)

        # Restore scalar value
        self.active_sh_degree = active_sh_degree

        # Restore optimizer state
        if optimizer_state is not None and hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)

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
        if self.mp.cluster_size > 0:
            cluster_origin, cluster_extend = scene.cluster.get_cluster_AABB(
                self.xyz, self.scale.exp(), nn.functional.normalize(self.rot, dim=0)
            )
            self.cluster_origin = cluster_origin
            self.cluster_extend = cluster_extend
        return

    def spatial_rearrange(self) -> None:
        (self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity) = \
            scene.spatial_refine(self.mp.cluster_size > 0, self.optimizer, self.xyz)
        return

    def step(self, epoch: int):
        """
        Perform density control step (densify, prune, opacity reset).
        Returns updated parameters.
        """
        
        self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity = \
            self.density_controller.step(self.optimizer, epoch)
        
        if epoch % self.density_controller.densify_params.densification_interval == 0:
            self.spatial_rearrange()
            self.update_cluster_aabb()

        if self.active_sh_degree < self.mp.sh_degree:
            self.active_sh_degree = min(int(epoch / 5), self.mp.sh_degree)

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
            visible_chunkid, visible_chunks_num, culled_xyz, culled_scale, culled_rot, color, culled_opacity
        """
        xyz, scale, rot, sh_0, sh_rest, opacity = self.get_params()

        (
            visible_chunkid, visible_chunks_num,
            culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity
        ) = render.render_preprocess(
            self.cluster_origin, self.cluster_extend, frustumplane, view_matrix,
            xyz, scale, rot, sh_0, sh_rest, opacity,
            feedback_visible_chunks_num, idx_tensor,
            self.mp.cluster_size>0,self.op.sparse_grad, self.active_sh_degree
        )

        if visible_chunks_num is not None:
            valid_length = visible_chunks_num * self.mp.cluster_size

        return (
            visible_chunkid, visible_chunks_num,valid_length,
            culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity
        )

    def save_ply(self, save_path: str):
        """Save Gaussian parameters to PLY file."""
        if self.mp.cluster_size > 0:
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
