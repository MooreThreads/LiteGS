import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

from .. import utils

if TYPE_CHECKING:
    from .. import data


class LearnableViewProj(nn.Module):
    """
    Manages learnable camera intrinsic and extrinsic parameters.
    Inherits nn.Module for state_dict save/load support.
    """

    def __init__(self, training_set: 'data.CameraFrameDataset', extr_only: bool = True):
        super().__init__()
        self.extr_only = extr_only

        # Initialize extrinsic parameters (per frame)
        noise_extr = torch.cat([frame.extr_params[None, :] for frame in training_set.frames])
        self.extrinsics = nn.Embedding(
            noise_extr.shape[0], noise_extr.shape[1],
            _weight=noise_extr.clone(), sparse=True
        )

        # Initialize intrinsic parameters (shared, using first camera)
        camera = list(training_set.cameras.values())[0]
        noise_intr = torch.tensor(camera.intr_params, dtype=torch.float32, device='cuda').unsqueeze(0)
        self.intrinsics = nn.Parameter(noise_intr)

        # Create optimizers (not part of nn.Module, so handled manually)
        self.view_optimizer = torch.optim.SparseAdam(self.extrinsics.parameters(), lr=1e-4)
        self.proj_optimizer = torch.optim.Adam([self.intrinsics], lr=1e-5)

    def forward(self, idx: torch.Tensor, img_h: int, img_w: int):
        """
        Get view matrix, projection matrix, and frustum plane for given frame indices.

        Args:
            idx: Frame indices tensor
            img_h: Image height
            img_w: Image width

        Returns:
            view_matrix, proj_matrix, viewproj_matrix, frustumplane
        """
        extr = self.extrinsics(idx)
        intr = self.intrinsics
        view_matrix, proj_matrix, viewproj_matrix, frustumplane=utils.wrapper.CreateViewProj.apply(extr, intr, img_h, img_w, 0.01, 5000)
        return view_matrix, proj_matrix, viewproj_matrix, frustumplane

    def step(self):
        """Perform optimization step for view and projection parameters."""
        self.view_optimizer.step()
        self.view_optimizer.zero_grad()
        if not self.extr_only:
            self.proj_optimizer.step()
            self.proj_optimizer.zero_grad()

    def state_dict(self,destination=None , prefix='', keep_vars=False):
        """
        Override state_dict to include optimizer states.
        """
        # Call parent's state_dict for parameters (extrinsics, intrinsics)
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Add optimizer states
        state[prefix + 'view_optimizer_state_dict'] = self.view_optimizer.state_dict()
        state[prefix + 'proj_optimizer_state_dict'] = self.proj_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Override load_state_dict to handle optimizer states.
        """
        # Extract optimizer states
        view_optimizer_state = state_dict.pop('view_optimizer_state_dict', None)
        proj_optimizer_state = state_dict.pop('proj_optimizer_state_dict', None)

        # Load parameters using parent's load_state_dict
        super().load_state_dict(state_dict, strict=strict)

        # Restore optimizer states
        if view_optimizer_state is not None:
            self.view_optimizer.load_state_dict(view_optimizer_state)
        if proj_optimizer_state is not None:
            self.proj_optimizer.load_state_dict(proj_optimizer_state)

        return torch.nn.modules.module._IncompatibleKeys([], [])
