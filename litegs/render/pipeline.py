import torch
from typing import Optional, TYPE_CHECKING, Tuple
from torchmetrics.image import psnr

from .. import arguments
from .. import render
from .. import data
from .. import scene
from ..scene.viewproj import LearnableViewProj

if TYPE_CHECKING:
    from ..scene.model import GaussianSplattingModel


class RenderPipeline:
    """
    Manages rendering pipeline: culling, rasterization, and optional learnable view-projection.
    """

    def __init__(
        self,
        pp: arguments.PipelineParams,
        training_set: 'data.CameraFrameDataset'
    ):
        self.pp = pp
        self.frames_buffer = data.FramesBuffer(training_set)
        self.learnable_viewproj: Optional[LearnableViewProj] = None

    def set_learnable_viewproj(self, training_set: 'data.CameraFrameDataset'):
        """Enable learnable view-projection parameters."""
        self.learnable_viewproj = LearnableViewProj(training_set)

    def render_preprocess(
        self,
        cluster_origin: Optional[torch.Tensor],
        cluster_extend: Optional[torch.Tensor],
        frustumplane: torch.Tensor,
        view_matrix: torch.Tensor,
        model: 'GaussianSplattingModel',
        feedback_buffer: Optional[torch.Tensor],
        idx_tensor: Optional[torch.Tensor],
        actived_sh_degree: int
    ) -> Tuple:
        """
        Perform cluster culling and gaussian activation.

        Returns:
            visible_chunkid, visible_chunks_num, culled_xyz, culled_scale, culled_rot, color, culled_opacity
        """
        xyz, scale, rot, sh_0, sh_rest, opacity = model.get_params()

        return render.render_preprocess(
            cluster_origin, cluster_extend, frustumplane, view_matrix,
            xyz, scale, rot, sh_0, sh_rest, opacity,
            feedback_buffer, idx_tensor,
            self.pp, actived_sh_degree
        )

    def render(
        self,
        view_matrix: torch.Tensor,
        proj_matrix: torch.Tensor,
        culled_xyz: torch.Tensor,
        culled_scale: torch.Tensor,
        culled_rot: torch.Tensor,
        culled_color: torch.Tensor,
        culled_opacity: torch.Tensor,
        valid_length: Optional[torch.Tensor],
        feedback_binning_allocate_size: Optional[torch.Tensor],
        idx_tensor: Optional[torch.Tensor],
        actived_sh_degree: int,
        output_shape: tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Gaussian rasterization.

        Returns:
            img, transmitance, depth, normal, primitive_visible
        """
        return render.render(
            view_matrix, proj_matrix,
            culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity,
            valid_length, feedback_binning_allocate_size, idx_tensor,
            actived_sh_degree, output_shape, self.pp
        )

    def forward(
        self,
        model: 'GaussianSplattingModel',
        view_matrix: torch.Tensor,
        proj_matrix: torch.Tensor,
        frustumplane: torch.Tensor,
        gt_image: torch.Tensor,
        idx_tensor: torch.Tensor,
        actived_sh_degree: int,
        frames_buffer: 'data.FramesBuffer'
    ) -> Tuple:
        """
        Full forward pass: preprocess -> render.

        Returns:
            img, transmitance, depth, normal, primitive_visible,
            visible_chunkid, visible_chunks_num
        """
        # Handle learnable view-projection
        if self.learnable_viewproj is not None:
            idx_tensor = idx_tensor.cuda()
            view_matrix, proj_matrix, _, frustumplane = self.learnable_viewproj.get_viewproj(
                idx_tensor, gt_image.shape[2], gt_image.shape[3]
            )

        # Cluster culling
        cluster_origin, cluster_extend = model.get_cluster_aabb()
        valid_length = None

        (
            visible_chunkid, visible_chunks_num,
            culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity
        ) = self.render_preprocess(
            cluster_origin, cluster_extend, frustumplane, view_matrix,
            model, frames_buffer.feedback_visible_chunks_num, idx_tensor,
            actived_sh_degree
        )

        if visible_chunks_num is not None:
            valid_length = visible_chunks_num * self.pp.cluster_size

        # Render
        img, transmitance, depth, normal, primitive_visible = self.render(
            view_matrix, proj_matrix,
            culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity,
            valid_length, frames_buffer.feedback_binning_allocate_size, idx_tensor,
            actived_sh_degree, gt_image.shape[2:]
        )

        return (img, transmitance, depth, normal, primitive_visible,
                visible_chunkid, visible_chunks_num)

    def eval_forward(
        self,
        model: 'GaussianSplattingModel',
        view_matrix: torch.Tensor,
        proj_matrix: torch.Tensor,
        frustumplane: torch.Tensor,
        idx: torch.Tensor,
        actived_sh_degree: int,
        cluster_origin: Optional[torch.Tensor] = None,
        cluster_extend: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluation forward pass (no grad).

        Returns:
            img, transmitance
        """
        if self.learnable_viewproj is not None:
            extr = self.learnable_viewproj.extrinsics(idx)
            intr = self.learnable_viewproj.intrinsics
            view_matrix, proj_matrix, _, frustumplane = self.learnable_viewproj.get_viewproj(
                idx, view_matrix.shape[2], view_matrix.shape[3]
            )

        xyz, scale, rot, sh_0, sh_rest, opacity = model.get_params()

        # Use old-style render_preprocess for evaluation
        visible_chunkid, visible_chunks_num, culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity = \
            render.render_preprocess(
                cluster_origin, cluster_extend, frustumplane, view_matrix,
                xyz, scale, rot, sh_0, sh_rest, opacity,
                None, idx, self.pp, actived_sh_degree
            )

        img, transmitance, depth, normal, primitive_visible = render.render(
            view_matrix, proj_matrix,
            culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity,
            None, None, idx, actived_sh_degree, (view_matrix.shape[2], view_matrix.shape[3]), self.pp
        )

        return img, transmitance

    def compute_psnr(
        self,
        model: 'GaussianSplattingModel',
        loader,
        actived_sh_degree: int,
        cluster_origin: Optional[torch.Tensor] = None,
        cluster_extend: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute average PSNR over a data loader.

        Returns:
            Average PSNR value
        """
        psnr_metric = psnr.PeakSignalNoiseRatio(data_range=(0.0, 1.0)).cuda()
        psnr_list = []

        with torch.no_grad():
            for view_matrix, proj_matrix, frustumplane, gt_image, idx in loader:
                view_matrix = view_matrix.cuda()
                proj_matrix = proj_matrix.cuda()
                frustumplane = frustumplane.cuda()
                gt_image = gt_image.cuda() / 255.0
                idx = idx.cuda()

                img, _ = self.eval_forward(
                    model, view_matrix, proj_matrix, frustumplane, idx,
                    actived_sh_degree, cluster_origin, cluster_extend
                )

                psnr_list.append(psnr_metric(img, gt_image).unsqueeze(0))

        return torch.concat(psnr_list, dim=0).mean().item()
