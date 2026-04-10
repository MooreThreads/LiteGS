import torch

from ..CompactedTensor import CompactedTensor
from ..fused_backend import fused
from .. import spherical_harmonics
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


class _CompactSHCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b_sparse_grad, sh_degree, visible_chunkid, visible_chunk_num, view_matrix, position, sh_base, sh_rest):
        ctx.sh_degree = sh_degree
        ctx.chunk_num = sh_base.shape[-2]
        ctx.chunk_size = sh_base.shape[-1]
        ctx.b_sparse_grad = b_sparse_grad
        color = fused.compact_sh_forward(
            sh_degree,
            visible_chunkid,
            visible_chunk_num,
            view_matrix,
            position,
            sh_base,
            sh_rest,
        )
        ctx.save_for_backward(visible_chunkid, visible_chunk_num, view_matrix, position, sh_base, sh_rest)
        return color

    @staticmethod
    def backward(ctx, color_grad):
        sh_degree = ctx.sh_degree
        chunk_num = ctx.chunk_num
        chunk_size = ctx.chunk_size
        b_sparse_grad = ctx.b_sparse_grad
        visible_chunkid, visible_chunk_num, view_matrix, position, sh_base, sh_rest = ctx.saved_tensors
        sh_base_grad, sh_rest_grad = fused.compact_sh_backward(
            sh_degree,
            visible_chunkid,
            visible_chunk_num,
            view_matrix,
            position,
            sh_base,
            sh_rest,
            color_grad.contiguous(),
        )

        if b_sparse_grad:
            allocate_chunk_num = visible_chunkid.shape[0]
            sh_base_grad = CompactedTensor(
                (*sh_base_grad.shape[:-2], chunk_num, chunk_size),
                visible_chunkid,
                sh_base_grad.reshape(-1, allocate_chunk_num, chunk_size),
            )
            sh_rest_grad = CompactedTensor(
                (*sh_rest_grad.shape[:-2], chunk_num, chunk_size),
                visible_chunkid,
                sh_rest_grad.reshape(-1, allocate_chunk_num, chunk_size),
            )
        else:
            dense_sh_base_grad = torch.zeros(
                (*sh_base_grad.shape[:-2], chunk_num, chunk_size),
                device=sh_base_grad.device,
                dtype=sh_base_grad.dtype,
            )
            dense_sh_base_grad[..., visible_chunkid, :] = sh_base_grad
            sh_base_grad = dense_sh_base_grad

            dense_sh_rest_grad = torch.zeros(
                (*sh_rest_grad.shape[:-2], chunk_num, chunk_size),
                device=sh_rest_grad.device,
                dtype=sh_rest_grad.dtype,
            )
            dense_sh_rest_grad[..., visible_chunkid, :] = sh_rest_grad
            sh_rest_grad = dense_sh_rest_grad

        return None, None, None, None, None, None, sh_base_grad, sh_rest_grad


def _compact_activate_nosh_script_forward(
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor | None,
    xyz: torch.Tensor,
    scale: torch.Tensor,
    rot: torch.Tensor,
    opacity: torch.Tensor,
):
    valid_chunk_num = _valid_chunk_count(visible_chunkid, visible_chunk_num)
    active_chunkid = visible_chunkid[:valid_chunk_num]
    activated_position = torch.cat(
        (
            xyz[..., active_chunkid, :],
            torch.ones((1, valid_chunk_num, xyz.shape[-1]), device=xyz.device, dtype=xyz.dtype),
        ),
        dim=0,
    )
    activated_scale = scale[..., active_chunkid, :].exp()
    activated_rotation = torch.nn.functional.normalize(rot[..., active_chunkid, :], dim=0)
    activated_opacity = opacity[..., active_chunkid, :].sigmoid()
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
        CUDA returns tensors with shape ``[4, A, K]`` / ``[3, A, K]`` / ``[4, A, K]`` / ``[1, A, K]``
        where ``A`` is the allocated compact buffer length.
        SCRIPT returns the exact valid prefix with shape ``[4, V, K]`` / ``[3, V, K]`` / ``[4, V, K]`` /
        ``[1, V, K]`` where ``V = visible_chunk_num``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return compact_activate_nosh_script(b_sparse_grad, visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity)
    return compact_activate_nosh_cuda(b_sparse_grad, visible_chunkid, visible_chunk_num, xyz, scale, rot, opacity)


def _compact_sh_script_forward(
    sh_degree: int,
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor | None,
    view_matrix: torch.Tensor,
    position: torch.Tensor,
    sh_base: torch.Tensor,
    sh_rest: torch.Tensor,
):
    valid_chunk_num = _valid_chunk_count(visible_chunkid, visible_chunk_num)
    active_chunkid = visible_chunkid[:valid_chunk_num]
    with torch.no_grad():
        camera_center = (-view_matrix[..., 3:4, :3] @ view_matrix[..., :3, :3].transpose(-1, -2)).squeeze(1)
        dirs = position.detach()[..., active_chunkid, :].unsqueeze(0) - camera_center.unsqueeze(-1).unsqueeze(-1)
        dirs = torch.nn.functional.normalize(dirs, dim=1)
    sh = torch.cat((sh_base[..., active_chunkid, :], sh_rest[..., active_chunkid, :]), dim=0)
    color = spherical_harmonics.sh_to_rgb(sh_degree, sh, dirs)
    return color


def compact_sh_script(
    b_sparse_grad: bool,
    sh_degree: int,
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor,
    view_matrix: torch.Tensor,
    position: torch.Tensor,
    sh_base: torch.Tensor,
    sh_rest: torch.Tensor,
):
    if b_sparse_grad:
        raise NotImplementedError("compact_sh_script does not implement sparse gradients")
    return _compact_sh_script_forward(sh_degree, visible_chunkid, visible_chunk_num, view_matrix, position, sh_base, sh_rest)


def compact_sh_cuda(
    b_sparse_grad: bool,
    sh_degree: int,
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor,
    view_matrix: torch.Tensor,
    position: torch.Tensor,
    sh_base: torch.Tensor,
    sh_rest: torch.Tensor,
):
    return _CompactSHCuda.apply(b_sparse_grad, sh_degree, visible_chunkid, visible_chunk_num, view_matrix, position, sh_base, sh_rest)


def compact_sh(
    b_sparse_grad: bool,
    sh_degree: int,
    visible_chunkid: torch.Tensor,
    visible_chunk_num: torch.Tensor,
    view_matrix: torch.Tensor,
    position: torch.Tensor,
    sh_base: torch.Tensor,
    sh_rest: torch.Tensor,
    *,
    backend=None,
):
    """
    Compact visible clustered spherical harmonics and evaluate RGB colors.

    Args:
        b_sparse_grad: Whether CUDA backward should return ``CompactedTensor`` gradients.
        sh_degree: Active spherical harmonic degree.
        visible_chunkid: Allocated visible chunk id buffer with shape ``[A]``.
        visible_chunk_num: Device tensor storing the valid prefix length inside ``visible_chunkid``.
        view_matrix: View matrices with shape ``[V, 4, 4]``.
        position: Clustered positions with shape ``[3, C, K]``.
        sh_base: Base SH coefficients with shape ``[1, 3, C, K]``.
        sh_rest: Remaining SH coefficients with shape ``[(degree + 1)^2 - 1, 3, C, K]``.
        backend: Backend selection. ``None`` means using the current default backend.

    Returns:
        RGB colors with shape ``[V, 3, A, K]`` in compacted chunk order.
        CUDA may over-allocate to ``A`` for device-only valid counts.
        SCRIPT returns the exact valid compacted shape ``[V, 3, visible_chunk_num, K]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return compact_sh_script(b_sparse_grad, sh_degree, visible_chunkid, visible_chunk_num, view_matrix, position, sh_base, sh_rest)
    return compact_sh_cuda(b_sparse_grad, sh_degree, visible_chunkid, visible_chunk_num, view_matrix, position, sh_base, sh_rest)
