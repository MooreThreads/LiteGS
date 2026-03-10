import torch
import torch.nn as nn
from typing import Optional

from .. import arguments
from .. import scene
from .. import io_manager
from ..training import optimizer as opt_module
from ..training import densify


class GaussianSplattingModel:
    """
    Manages Gaussian splatting parameters, clustering, optimizer and scheduler.
    """

    def __init__(
        self,
        init_xyz: torch.Tensor,
        init_color: torch.Tensor,
        sh_degree: int,
        norm_radius: float,
        op: arguments.OptimizationParams,
        pp: arguments.PipelineParams,
        dp: arguments.DensifyParams,
        init_points_num: int,
        start_checkpoint: Optional[str] = None
    ):
        self.pp = pp
        self.op = op
        self.init_points_num = init_points_num

        if start_checkpoint is None:
            # Initialize from scratch
            self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity = \
                scene.create_gaussians(init_xyz, init_color, sh_degree)

            # Apply clustering if needed
            if pp.cluster_size > 0:
                self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity = \
                    scene.cluster.cluster_points(pp.cluster_size, self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity)

            # Convert to nn.Parameter
            self.xyz = nn.Parameter(self.xyz)
            self.scale = nn.Parameter(self.scale)
            self.rot = nn.Parameter(self.rot)
            self.sh_0 = nn.Parameter(self.sh_0)
            self.sh_rest = nn.Parameter(self.sh_rest)
            self.opacity = nn.Parameter(self.opacity)

            # Create optimizer and scheduler
            self.optimizer, self.scheduler = opt_module.get_optimizer(
                self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity,
                norm_radius, op, pp
            )

            self.start_epoch = 0
        else:
            # Load from checkpoint
            (self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest,
             self.opacity, self.start_epoch, self.optimizer, self.scheduler) = \
                io_manager.load_checkpoint(start_checkpoint)

        # Initialize cluster AABB
        self.cluster_origin = None
        self.cluster_extend = None
        if pp.cluster_size > 0:
            self.update_cluster_aabb()

        # Create density controller
        self.density_controller = densify.DensityControllerTamingGS(
            norm_radius, dp, pp.cluster_size > 0, init_points_num
        )

        # Active SH degree (progressive SH)
        self.active_sh_degree = 0

    def get_params(self):
        """Return all Gaussian parameters."""
        return self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity

    def get_cluster_aabb(self):
        """Return cluster AABB (origin, extend)."""
        return self.cluster_origin, self.cluster_extend

    def update_cluster_aabb(self)->None:
        """Update cluster AABB based on current scale and rotation."""
        if self.pp.cluster_size > 0:
            self.cluster_origin, self.cluster_extend = scene.cluster.get_cluster_AABB(
                self.xyz, self.scale.exp(), nn.functional.normalize(self.rot, dim=0)
            )
        return

    def spatial_rearrange(self)->None:
        (self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity)=scene.spatial_refine(self.pp.cluster_size>0,self.optimizer,self.xyz)
        return

    def step(self, epoch: int):
        """
        Perform density control step (densify, prune, opacity reset).
        Returns updated parameters.
        """
        self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity = \
            self.density_controller.step(self.optimizer, epoch)
        return self.get_params()

    def save_ply(self, save_path: str):
        """Save Gaussian parameters to PLY file."""
        if self.pp.cluster_size > 0:
            tensors = scene.cluster.uncluster(
                self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity
            )
        else:
            tensors = (self.xyz, self.scale, self.rot, self.sh_0, self.sh_rest, self.opacity)

        param_nyp = []
        for tensor in tensors:
            param_nyp.append(tensor.detach().cpu().numpy())

        io_manager.save_ply(save_path, *param_nyp)

    @staticmethod
    def from_checkpoint(checkpoint_path: str, sh_degree: int, norm_radius: float,
                        op: arguments.OptimizationParams, pp: arguments.PipelineParams,
                        dp: arguments.DensifyParams, init_points_num: int):
        """Load model from checkpoint."""
        model = GaussianSplattingModel.__new__(GaussianSplattingModel)
        model.pp = pp
        model.op = op
        model.init_points_num = init_points_num

        (model.xyz, model.scale, model.rot, model.sh_0, model.sh_rest,
         model.opacity, model.start_epoch, model.optimizer, model.scheduler) = \
            io_manager.load_checkpoint(checkpoint_path)

        model.cluster_origin = None
        model.cluster_extend = None
        if pp.cluster_size > 0:
            model.update_cluster_aabb()

        model.density_controller = densify.DensityControllerTamingGS(
            norm_radius, dp, pp.cluster_size > 0, init_points_num
        )

        # Active SH degree (progressive SH)
        model.active_sh_degree = 0

        return model
