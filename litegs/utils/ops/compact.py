import torch

from ..CompactedTensor import CompactedTensor
from ..fused_backend import fused
from .backend import Backend, normalize_backend


def _valid_chunk_count(visible_chunkid: torch.Tensor, visible_chunk_num: torch.Tensor | None) -> int:
    if visible_chunk_num is None:
        return int(visible_chunkid.shape[0])
    if visible_chunk_num.numel() == 0:
        return int(visible_chunkid.shape[0])
    return min(int(visible_chunk_num.detach().cpu().reshape(-1)[0].item()), int(visible_chunkid.shape[0]))


class _CompactActivateNoSHCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b_sparse_grad, visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity):
        ctx.chunk_num = xyz.shape[-2]
        ctx.chunk_size = xyz.shape[-1]
        ctx.b_sparse_grad = b_sparse_grad
        activated_position, activated_scale, activated_rotation, activated_opacity = fused.cull_compact_activate_nosh(
            visible_chunkid,
            visible_chunk_num,
            xyz,
            scale,
            rot,
            opacity,
        )
        ctx.save_for_backward(visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity)
        return activated_position, activated_scale, activated_rotation, activated_opacity

    @staticmethod
    def backward(ctx, activated_position_grad, activated_scale_grad, activated_rotation_grad, activated_opacity_grad):
        chunk_num = ctx.chunk_num
        chunk_size = ctx.chunk_size
        b_sparse_grad = ctx.b_sparse_grad
        visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity = ctx.saved_tensors
        compacted_grads = fused.activate_backward_nosh(
            visible_chunkid,
            visible_chunk_num,
            xyz,
            scale,
            rot,
            opacity,
            activated_position_grad.contiguous(),
            activated_scale_grad.contiguous(),
            activated_rotation_grad.contiguous(),
            activated_opacity_grad.contiguous(),
        )

        if b_sparse_grad:
            allocate_chunk_num = visible_chunkid.shape[0]
            grads = []
            for grad in compacted_grads:
                size = (*grad.shape[:-2], chunk_num, chunk_size)
                grads.append(CompactedTensor(size, visible_chunkid, grad.reshape(-1, allocate_chunk_num, chunk_size)))
        else:
            grads = []
            for compacted_grad in compacted_grads:
                size = (*compacted_grad.shape[:-2], chunk_num, chunk_size)
                grad = torch.zeros(size, device=compacted_grad.device, dtype=compacted_grad.dtype)
                grad[..., visible_chunkid, :] = compacted_grad
                grads.append(grad)

        return None, None, None, *grads


def _compact_activate_nosh_script_forward(
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor | None,
    xyz: torch.Tensor,
    scale: torch.Tensor,
    rot: torch.Tensor,
    opacity: torch.Tensor,
):
    valid_chunk_num = _valid_chunk_count(visible_chunkid, visible_chunk_num)
    allocate_chunk_num = int(visible_chunkid.shape[0])
    active_chunkid = visible_chunkid[:valid_chunk_num]

    activated_position = torch.zeros((4, allocate_chunk_num, xyz.shape[-1]), device=xyz.device, dtype=xyz.dtype)
    activated_scale = torch.zeros((scale.shape[0], allocate_chunk_num, scale.shape[-1]), device=scale.device, dtype=scale.dtype)
    activated_rotation = torch.zeros((rot.shape[0], allocate_chunk_num, rot.shape[-1]), device=rot.device, dtype=rot.dtype)
    activated_opacity = torch.zeros((opacity.shape[0], allocate_chunk_num, opacity.shape[-1]), device=opacity.device, dtype=opacity.dtype)

    if valid_chunk_num > 0:
        activated_position[:3, :valid_chunk_num, :] = xyz[..., active_chunkid, :]
        activated_position[3, :valid_chunk_num, :] = 1.0
        activated_scale[:, :valid_chunk_num, :] = scale[..., active_chunkid, :].exp()
        activated_rotation[:, :valid_chunk_num, :] = torch.nn.functional.normalize(rot[..., active_chunkid, :], dim=0)
        activated_opacity[:, :valid_chunk_num, :] = opacity[..., active_chunkid, :].sigmoid()

    return activated_position, activated_scale, activated_rotation, activated_opacity


def compact_activate_nosh_script(
    b_sparse_grad: bool,
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor,
    xyz: torch.Tensor,
    scale: torch.Tensor,
    rot: torch.Tensor,
    opacity: torch.Tensor,
):
    if b_sparse_grad:
        raise NotImplementedError("compact_activate_nosh_script does not implement sparse gradients")
    return _compact_activate_nosh_script_forward(visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity)


def compact_activate_nosh_cuda(
    b_sparse_grad: bool,
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor,
    xyz: torch.Tensor,
    scale: torch.Tensor,
    rot: torch.Tensor,
    opacity: torch.Tensor,
):
    return _CompactActivateNoSHCuda.apply(b_sparse_grad, visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity)


def compact_activate_nosh(
    b_sparse_grad: bool,
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor,
    xyz: torch.Tensor,
    scale: torch.Tensor,
    rot: torch.Tensor,
    opacity: torch.Tensor,
    *,
    backend=None,
):
    """
    Compact visible clustered Gaussians and apply activation without spherical harmonics.

    Args:
        b_sparse_grad: Whether backward should return ``CompactedTensor`` gradients.
        visible_chunkid: Allocated visible chunk id buffer with shape ``[A]``.
        visible_chunk_num: Device tensor storing the valid prefix length inside ``visible_chunkid``.
        xyz: Clustered positions with shape ``[3, C, K]``.
        scale: Clustered log-scale values with shape ``[3, C, K]``.
        rot: Clustered quaternion values with shape ``[4, C, K]``.
        opacity: Clustered opacity logits with shape ``[1, C, K]``.
        backend: Backend selection. ``None`` means using the current default backend.

    Returns:
        A tuple ``(position, scale, rotation, opacity)``.
        ``position`` has shape ``[4, A, K]`` and includes the homogeneous row of ones.
        The remaining outputs preserve the clustered layout with shapes ``[3, A, K]``, ``[4, A, K]``,
        and ``[1, A, K]``.
        Entries beyond ``visible_chunk_num`` are zero-filled.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return compact_activate_nosh_script(b_sparse_grad, visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity)
    return compact_activate_nosh_cuda(b_sparse_grad, visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity)
