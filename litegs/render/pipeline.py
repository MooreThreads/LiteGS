import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING, Tuple
from torchmetrics.image import psnr

from .. import arguments
from .. import render
from ..scene.viewproj import LearnableViewProj
from .. import data

if TYPE_CHECKING:
    from ..scene.model import GaussianSplattingModel


class RenderPipeline(nn.Module):
    """
    Manages rendering pipeline: only rasterization (render).
    Culling + activation is now handled by GaussianSplattingModel.forward().
    Inherits nn.Module to support state_dict save/load.
    """

    def __init__(
        self,
        mp: arguments.ModelParams,
        gs_model:'GaussianSplattingModel',
        training_set: 'data.CameraFrameDataset'
    ):
        super().__init__()
        self.mp = mp
        self.frames_buffer = data.FramesBuffer(training_set)
        self.model = gs_model
        self.start_epoch = 0

        # Learnable view-projection as sub-module
        self.learnable_viewproj: Optional[LearnableViewProj] = None
        if mp.learnable_viewproj:
            self.learnable_viewproj = LearnableViewProj(training_set)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to include start_epoch and model states.
        """
        # Call parent's state_dict for sub-modules (learnable_viewproj)
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Add start_epoch
        state[prefix + 'start_epoch'] = torch.tensor(self.start_epoch)

        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Override load_state_dict to handle start_epoch and model states.
        """
        # Extract start_epoch
        start_epoch = state_dict.pop('start_epoch').item() if 'start_epoch' in state_dict else 0

        # Load sub-module states
        super().load_state_dict(state_dict, strict=strict)

        # Restore start_epoch
        self.start_epoch = start_epoch

        return torch.nn.modules.module._IncompatibleKeys([], [])

    def forward(
        self,
        view_matrix:torch.Tensor,proj_matrix:torch.Tensor,frustumplane:torch.Tensor,
        idx_tensor:torch.Tensor,
        output_shape: tuple[int, int],bTraining:bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Gaussian rasterization (only render, no culling/activation).

        Returns:
            img, transmitance, depth, normal, primitive_visible
        """
        feedback_visible_chunks_num=None
        feedback_binning_allocate_size=None
        if bTraining:
            feedback_visible_chunks_num=self.frames_buffer.feedback_visible_chunks_num
            feedback_binning_allocate_size=self.frames_buffer.feedback_binning_allocate_size

        if self.learnable_viewproj is not None:
            view_matrix, proj_matrix,viewproj_matrix,frustumplane=self.learnable_viewproj(idx_tensor.cuda(),output_shape[0],output_shape[1])

        (
            visible_chunkid, visible_chunks_num,valid_length,
            xyz, scale, rot, color, opacity
        ) = self.model(view_matrix, frustumplane, idx_tensor, feedback_visible_chunks_num)

        img,transmitance,depth,normal,primitive_visible=render.render(
            view_matrix, proj_matrix,
            xyz, scale, rot, color, opacity,
            valid_length, feedback_binning_allocate_size, idx_tensor,
            output_shape, self.mp
        )

        if bTraining:
            self.model.optimizer.update_sparse_visibility(visible_chunkid,visible_chunks_num,primitive_visible,valid_length)

        return img,transmitance,depth,normal
