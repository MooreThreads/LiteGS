import torch

from ..fused_backend import fused
from .backend import Backend, normalize_backend


class _EighAndInverse2x2MatrixCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cov2d: torch.Tensor, valid_length: torch.Tensor | None = None):
        saved_valid_length = valid_length
        if saved_valid_length is None:
            saved_valid_length = torch.tensor([], device=cov2d.device, dtype=torch.int32)
        runtime_valid_length = None if saved_valid_length.numel() == 0 else saved_valid_length
        eigen_val, eigen_vec, cov2d_inv = fused.eigh_and_inv_2x2matrix_forward(cov2d, runtime_valid_length)
        ctx.save_for_backward(cov2d_inv, saved_valid_length)
        return eigen_val, eigen_vec, cov2d_inv

    @staticmethod
    def backward(ctx, eigen_val_grad: torch.Tensor, eigen_vec_grad: torch.Tensor, cov2d_inv_grad: torch.Tensor):
        cov2d_inv_grad = cov2d_inv_grad.contiguous()
        cov2d_inv, saved_valid_length = ctx.saved_tensors
        runtime_valid_length = None if saved_valid_length.numel() == 0 else saved_valid_length
        cov2d_grad = fused.inv_2x2matrix_backward(cov2d_inv, cov2d_inv_grad, runtime_valid_length)
        cov2d_grad.nan_to_num_(0)
        return cov2d_grad, None


def eigh_and_inverse_2x2_matrix_script(
    cov2d: torch.Tensor,
    valid_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        eigen_val, eigen_vec = torch.linalg.eigh(cov2d.permute(0, 3, 1, 2).reshape(-1, 2, 2))
        eigen_val = eigen_val.reshape(cov2d.shape[0], cov2d.shape[3], 2).permute(0, 2, 1)
        eigen_vec = eigen_vec.reshape(cov2d.shape[0], cov2d.shape[3], 2, 2).permute(0, 2, 3, 1)

    cov2d_inv = torch.linalg.inv(cov2d.permute(0, 3, 1, 2).reshape(-1, 2, 2))
    cov2d_inv = cov2d_inv.reshape(cov2d.shape[0], cov2d.shape[3], 2, 2).permute(0, 2, 3, 1)
    return eigen_val, eigen_vec, cov2d_inv


def eigh_and_inverse_2x2_matrix_cuda(
    cov2d: torch.Tensor,
    valid_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _EighAndInverse2x2MatrixCuda.apply(cov2d, valid_length)


def eigh_and_inverse_2x2_matrix(
    cov2d: torch.Tensor,
    valid_length: torch.Tensor | None = None,
    *,
    backend=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute eigen decomposition and inverse for batched 2x2 covariance matrices.

    Args:
        cov2d: Symmetric 2x2 matrices with shape ``[N, 2, 2, P]``.
            The layout is ``[view, row, col, primitive]``.
        valid_length: Optional active primitive count tensor. This parameter is
            only used by the CUDA backend and is ignored by the script backend.
        backend: Backend selection. ``None`` means using the current default backend from
            ``ops.backend``.

    Returns:
        A tuple ``(eigen_val, eigen_vec, cov2d_inv)``.
        ``eigen_val`` has shape ``[N, 2, P]`` with layout ``[view, eigen_index, primitive]``.
        ``eigen_vec`` has shape ``[N, 2, 2, P]`` with layout ``[view, row, col, primitive]``.
        ``cov2d_inv`` has shape ``[N, 2, 2, P]`` with layout ``[view, row, col, primitive]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return eigh_and_inverse_2x2_matrix_script(cov2d, valid_length)
    return eigh_and_inverse_2x2_matrix_cuda(cov2d, valid_length)
