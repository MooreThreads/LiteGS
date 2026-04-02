import math

import torch

from ..fused_backend import fused
from .backend import Backend, normalize_backend


@torch.no_grad()
def binning_cuda(
    ndc: torch.Tensor,
    view_depth: torch.Tensor,
    inv_cov2d: torch.Tensor,
    opacity: torch.Tensor,
    valid_length: torch.Tensor | None,
    feedback_binning_allocate_size: torch.Tensor | None,
    idx_tensor: torch.Tensor | None,
    img_pixel_shape: tuple[int, int],
    tile_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_tile_shape = (
        int(math.ceil(img_pixel_shape[0] / float(tile_size[0]))),
        int(math.ceil(img_pixel_shape[1] / float(tile_size[1]))),
    )
    tiles_num = img_tile_shape[0] * img_tile_shape[1]

    _, _, allocate_size = fused.get_allocate_size(
        ndc,
        view_depth,
        inv_cov2d,
        opacity,
        img_pixel_shape[0],
        img_pixel_shape[1],
        tile_size[0],
        tile_size[1],
        valid_length,
    )
    visible_mask = allocate_size != 0

    _, depth_sorted_index = view_depth.sort(dim=-1, descending=False)
    for batch_index in range(ndc.shape[0]):
        allocate_size[batch_index] = allocate_size[batch_index, depth_sorted_index[batch_index]]
    depth_sorted_allocate_size = allocate_size
    prefix_sum = depth_sorted_allocate_size.cumsum(1, dtype=torch.int32)

    sorted_tile_id, sorted_point_id = fused.create_table(
        ndc,
        inv_cov2d,
        opacity,
        prefix_sum,
        depth_sorted_index,
        feedback_binning_allocate_size,
        idx_tensor,
        img_pixel_shape[0],
        img_pixel_shape[1],
        tile_size[0],
        tile_size[1],
    )
    tile_start_index = fused.tileRange(sorted_tile_id, int(tiles_num))
    return tile_start_index, sorted_point_id, visible_mask.sum(0)


def binning(
    ndc: torch.Tensor,
    view_depth: torch.Tensor,
    inv_cov2d: torch.Tensor,
    opacity: torch.Tensor,
    valid_length: torch.Tensor | None,
    feedback_binning_allocate_size: torch.Tensor | None,
    idx_tensor: torch.Tensor | None,
    img_pixel_shape: tuple[int, int],
    tile_size: tuple[int, int],
    *,
    backend=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bin projected Gaussians into screen-space tiles.

    Args:
        ndc: Homogeneous NDC positions with shape ``[N, 4, P]``.
            The layout is ``[view, component, primitive]``.
        view_depth: View-space depth values with shape ``[N, P]``.
        inv_cov2d: Inverse 2D covariance matrices with shape ``[N, 2, 2, P]``.
            The layout is ``[view, row, col, primitive]``.
        opacity: Per-view opacity values with shape ``[N, P]``.
        valid_length: Optional active primitive count tensor used by the CUDA backend.
        feedback_binning_allocate_size: Optional feedback buffer for allocation reuse.
        idx_tensor: Optional frame index tensor used together with the feedback buffer.
        img_pixel_shape: Image shape as ``(height, width)`` in pixels.
        tile_size: Tile shape as ``(tile_height, tile_width)`` in pixels.
        backend: Backend selection. ``SCRIPT`` is not implemented for this operator.

    Returns:
        tile_start_index: has shape ``[N, tiles_num + 2]`` and dtype ``int32``.
            Tile index ``0`` is a reserved invalid label and is intentionally left empty.
            For a valid tile id ``tile_i``, the tile covers the half-open range
            ``[tile_start_index[:, tile_i], tile_start_index[:, tile_i + 1])`` inside
            ``sorted_point_id``. The extra two entries come from the reserved invalid
            tile ``0`` plus the terminal end offset. If either endpoint of that range is
            ``-1``, the tile is empty.
        sorted_point_id: stores the primitive ids referenced by the tile table.
        primitive_visible: stores per-primitive visibility counts aggregated across views.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        raise NotImplementedError("binning does not provide a script backend")
    return binning_cuda(
        ndc,
        view_depth,
        inv_cov2d,
        opacity,
        valid_length,
        feedback_binning_allocate_size,
        idx_tensor,
        img_pixel_shape,
        tile_size,
    )
