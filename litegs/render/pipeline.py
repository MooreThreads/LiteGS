import torch
from typing import Optional, TYPE_CHECKING, Tuple
from torchmetrics.image import psnr

from .. import arguments
from .. import render
from ..scene.viewproj import LearnableViewProj
from .. import data

if TYPE_CHECKING:
    from ..scene.model import GaussianSplattingModel


class RenderPipeline:
    """
    Manages rendering pipeline: only rasterization (render).
    Culling + activation is now handled by GaussianSplattingModel.forward().
    """

    def __init__(
        self,
        mp: arguments.ModelParams,
        training_set: 'data.CameraFrameDataset'
    ):
        self.mp = mp
        self.frames_buffer = data.FramesBuffer(training_set)
        self.learnable_viewproj: Optional[LearnableViewProj] = None
        if mp.learnable_viewproj:
            self.learnable_viewproj = LearnableViewProj(training_set)

    def forward(
        self,
        view_matrix:torch.Tensor,proj_matrix:torch.Tensor,frustumplane:torch.Tensor,
        model:'GaussianSplattingModel',
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
            view_matrix, proj_matrix,viewproj_matrix,frustumplane=self.learnable_viewproj()

        (
            visible_chunkid, visible_chunks_num,valid_length,
            xyz, scale, rot, color, opacity
        ) = model(view_matrix, frustumplane, idx_tensor, feedback_visible_chunks_num)
        
        img,transmitance,depth,normal,primitive_visible=render.render(
            view_matrix, proj_matrix,
            xyz, scale, rot, color, opacity,
            valid_length, feedback_binning_allocate_size, idx_tensor,
            output_shape, self.mp
        )

        if bTraining:
            model.optimizer.update_sparse_visibility(visible_chunkid,visible_chunks_num,primitive_visible,valid_length)

        return img,transmitance,depth,normal
