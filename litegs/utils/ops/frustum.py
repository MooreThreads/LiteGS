import torch

from ..fused_backend import fused
from .backend import Backend, normalize_backend


@torch.no_grad()
def frustum_culling_aabb_script(
    aabb_origin: torch.Tensor,
    aabb_ext: torch.Tensor,
    frustumplane: torch.Tensor,
    feedback_buffer: torch.Tensor | None = None,
    data_idx: torch.Tensor | None = None,
):
    """
    Cull clustered AABBs against the view frustum on the Python side.

    Args:
        aabb_origin: Cluster AABB origins with shape ``[3, M]``.
        aabb_ext: Cluster AABB half extents with shape ``[3, M]``.
        frustumplane: Frustum plane equations with shape ``[V, 6, 4]``.
        feedback_buffer: Optional host feedback buffer used to predict the allocated compact buffer size.
        data_idx: Optional frame indices used together with ``feedback_buffer``.

    Returns:
        A tuple ``(visibility, visible_chunk_num, visible_chunkid)``.
        ``visibility`` is a global visibility mask of shape ``[M]`` reduced across all views.
        ``visible_chunk_num`` has shape ``[1]`` and stores the valid compacted prefix length.
        ``visible_chunkid`` has shape ``[A]`` where ``A`` is either the exact visible count or a predicted
        allocation size when feedback is provided. Only the prefix ``:visible_chunk_num`` is defined.
    """
    dist_origin = (frustumplane[..., :3, None] * aabb_origin).sum(dim=-2).permute(2, 0, 1) + frustumplane[..., 3]
    dist_ext = (frustumplane[..., :3, None] * aabb_ext).abs().sum(dim=-2).permute(2, 0, 1)
    pushed_origin_dist = dist_origin + dist_ext
    per_view_visibility = (pushed_origin_dist >= 0).all(dim=-1)
    visibility = per_view_visibility.any(dim=-1).contiguous()

    visible_count = int(visibility.sum().item())
    visible_chunk_num = torch.tensor([visible_count], device=frustumplane.device, dtype=torch.int32)
    visible_chunkid = visibility.nonzero(as_tuple=False)[:, 0].to(dtype=torch.int64)

    return visibility, visible_chunk_num, visible_chunkid


@torch.no_grad()
def frustum_culling_aabb_cuda(
    aabb_origin: torch.Tensor,
    aabb_ext: torch.Tensor,
    frustumplane: torch.Tensor,
    feedback_buffer: torch.Tensor | None = None,
    data_idx: torch.Tensor | None = None,
):
    visibility, visible_chunk_num, visible_chunkid = fused.frustum_culling_aabb(
        aabb_origin,
        aabb_ext,
        frustumplane,
        feedback_buffer,
        data_idx,
    )
    return visibility, visible_chunk_num, visible_chunkid


def frustum_culling_aabb(
    aabb_origin: torch.Tensor,
    aabb_ext: torch.Tensor,
    frustumplane: torch.Tensor,
    feedback_buffer: torch.Tensor | None = None,
    data_idx: torch.Tensor | None = None,
    *,
    backend=None,
):
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return frustum_culling_aabb_script(aabb_origin, aabb_ext, frustumplane, feedback_buffer, data_idx)
    return frustum_culling_aabb_cuda(aabb_origin, aabb_ext, frustumplane, feedback_buffer, data_idx)
