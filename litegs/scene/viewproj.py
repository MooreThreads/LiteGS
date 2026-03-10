import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

from .. import utils

if TYPE_CHECKING:
    from .. import data


class LearnableViewProj:
    """
    Manages learnable camera intrinsic and extrinsic parameters.
    """

    def __init__(self, training_set: 'data.CameraFrameDataset'):
        self.training_set = training_set

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

        # Create optimizers
        self.view_optimizer = torch.optim.SparseAdam(self.extrinsics.parameters(), lr=1e-4)
        self.proj_optimizer = torch.optim.Adam([self.intrinsics], lr=1e-5)

    def get_viewproj(self, idx: torch.Tensor, img_h: int, img_w: int):
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
        return utils.wrapper.CreateViewProj.apply(extr, intr, img_h, img_w, 0.01, 5000)

    def step(self):
        """Perform optimization step for view and projection parameters."""
        self.view_optimizer.step()
        self.view_optimizer.zero_grad()
        # proj_optimizer currently disabled in original code
        # self.proj_optimizer.step()
        # self.proj_optimizer.zero_grad()

    def save(self, path: str):
        """Save learnable parameters."""
        torch.save(
            list(self.extrinsics.parameters()) + [self.intrinsics],
            path
        )

    def load(self, path: str):
        """Load learnable parameters."""
        loaded = torch.load(path)
        for i, param in enumerate(self.extrinsics.parameters()):
            param.data = loaded[i]
        self.intrinsics.data = loaded[-1]
