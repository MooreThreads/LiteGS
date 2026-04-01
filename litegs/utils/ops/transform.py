import torch

from ..fused_backend import load_fused_backend
from .backend import Backend, normalize_backend


class _CreateTransformMatrixCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale: torch.Tensor, rot: torch.Tensor, valid_length: torch.Tensor | None = None):
        litegs_fused = load_fused_backend()
        saved_valid_length = valid_length
        if saved_valid_length is None:
            saved_valid_length = torch.tensor([], device=scale.device, dtype=torch.int32)
        ctx.save_for_backward(scale, rot, saved_valid_length)
        runtime_valid_length = None if saved_valid_length.numel() == 0 else saved_valid_length
        return litegs_fused.createTransformMatrix_forward(rot, scale, runtime_valid_length)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        litegs_fused = load_fused_backend()
        scale, rot, saved_valid_length = ctx.saved_tensors
        runtime_valid_length = None if saved_valid_length.numel() == 0 else saved_valid_length
        grad_rot, grad_scale = litegs_fused.createTransformMatrix_backward(grad_output, rot, scale, runtime_valid_length)
        return grad_scale, grad_rot, None


def create_transform_matrix_script(
    scale: torch.Tensor,
    rot: torch.Tensor,
    valid_length: torch.Tensor | None = None,
) -> torch.Tensor:
    rotation_matrix = torch.zeros((3, 3, rot.shape[-1]), device=rot.device, dtype=rot.dtype)

    r = rot[0]
    x = rot[1]
    y = rot[2]
    z = rot[3]

    rotation_matrix[0, 0] = 1 - 2 * (y * y + z * z)
    rotation_matrix[0, 1] = 2 * (x * y + r * z)
    rotation_matrix[0, 2] = 2 * (x * z - r * y)

    rotation_matrix[1, 0] = 2 * (x * y - r * z)
    rotation_matrix[1, 1] = 1 - 2 * (x * x + z * z)
    rotation_matrix[1, 2] = 2 * (y * z + r * x)

    rotation_matrix[2, 0] = 2 * (x * z + r * y)
    rotation_matrix[2, 1] = 2 * (y * z - r * x)
    rotation_matrix[2, 2] = 1 - 2 * (x * x + y * y)

    return rotation_matrix * scale.unsqueeze(1)


def create_transform_matrix_cuda(
    scale: torch.Tensor,
    rot: torch.Tensor,
    valid_length: torch.Tensor | None = None,
) -> torch.Tensor:
    return _CreateTransformMatrixCuda.apply(scale, rot, valid_length)


def create_transform_matrix(
    scale: torch.Tensor,
    rot: torch.Tensor,
    valid_length: torch.Tensor | None = None,
    *,
    backend=None,
) -> torch.Tensor:
    """
    Build per-primitive transform matrices using the selected backend.

    Args:
        scale: Gaussian scale parameters with shape ``[3, P]``.
            The first dimension stores scale along x/y/z, and the last dimension
            indexes primitives.
        rot: Quaternion rotation parameters with shape ``[4, P]``.
            The first dimension is ordered as ``[r, x, y, z]``, and the last
            dimension indexes primitives.
        valid_length: Optional active primitive count tensor. This parameter is
            only used by the CUDA backend and is ignored by the script backend.
        backend: Backend selection. ``None`` means using the current default
            backend from ``ops.backend``.

    Returns:
        Transform matrices with shape ``[3, 3, P]``.
        The layout is ``[row, col, primitive]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return create_transform_matrix_script(scale, rot, valid_length)
    return create_transform_matrix_cuda(scale, rot, valid_length)


def create_rayspace_transform_matrix_script(
    view_pos: torch.Tensor,
    proj_matrix: torch.Tensor,
    output_shape: tuple[int, int],
    valid_length: torch.Tensor | None = None,
) -> torch.Tensor:
    t = view_pos.clone()
    limit_x = t[:, 2] / proj_matrix[:, 0, 0].unsqueeze(-1) * 1.3
    limit_y = t[:, 2] / proj_matrix[:, 1, 1].unsqueeze(-1) * 1.3
    t[:, 0] = torch.minimum(torch.maximum(t[:, 0], -limit_x), limit_x)
    t[:, 1] = torch.minimum(torch.maximum(t[:, 1], -limit_y), limit_y)
    t[:, 2].clamp_(1e-2)

    jacobian = torch.zeros((t.shape[0], 3, 3, t.shape[-1]), device=t.device, dtype=t.dtype)
    tz_square = t[:, 2] * t[:, 2]
    focal_length_x = output_shape[1] * proj_matrix[:, 0, 0] * 0.5
    focal_length_y = output_shape[0] * proj_matrix[:, 1, 1] * 0.5
    jacobian[:, 0, 0] = focal_length_x.unsqueeze(-1) / t[:, 2]
    jacobian[:, 1, 1] = focal_length_y.unsqueeze(-1) / t[:, 2]
    jacobian[:, 2, 0] = -(focal_length_x.unsqueeze(-1) * t[:, 0]) / tz_square
    jacobian[:, 2, 1] = -(focal_length_y.unsqueeze(-1) * t[:, 1]) / tz_square
    return jacobian


def create_rayspace_transform_matrix_cuda(
    view_pos: torch.Tensor,
    proj_matrix: torch.Tensor,
    output_shape: tuple[int, int],
    valid_length: torch.Tensor | None = None,
) -> torch.Tensor:
    litegs_fused = load_fused_backend()
    return litegs_fused.jacobianRayspace(view_pos, proj_matrix, output_shape[0], output_shape[1], valid_length)


def create_rayspace_transform_matrix(
    view_pos: torch.Tensor,
    proj_matrix: torch.Tensor,
    output_shape: tuple[int, int],
    valid_length: torch.Tensor | None = None,
    *,
    backend=None,
) -> torch.Tensor:
    """
    Build per-view ray-space Jacobians using the selected backend.

    Args:
        view_pos: View-space homogeneous positions with shape ``[N, 4, P]``.
            ``N`` is the number of views, the middle dimension stores
            ``[x, y, z, w]``, and the last dimension indexes primitives.
        proj_matrix: Projection matrices with shape ``[N, 4, 4]``.
        output_shape: Image shape as ``(height, width)`` in pixels.
        valid_length: Optional active primitive count tensor. This parameter is
            only used by the CUDA backend and is ignored by the script backend.
        backend: Backend selection. ``None`` means using the current default
            backend from ``ops.backend``.

    Returns:
        Ray-space Jacobians with shape ``[N, 3, 3, P]``.
        The layout is ``[view, row, col, primitive]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return create_rayspace_transform_matrix_script(view_pos, proj_matrix, output_shape, valid_length)
    return create_rayspace_transform_matrix_cuda(view_pos, proj_matrix, output_shape, valid_length)
