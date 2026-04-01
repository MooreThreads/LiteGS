import torch

from ..fused_backend import fused
from .backend import Backend, normalize_backend


class _CreateTransformMatrixCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale: torch.Tensor, rot: torch.Tensor, valid_length: torch.Tensor | None = None):
        saved_valid_length = valid_length
        if saved_valid_length is None:
            saved_valid_length = torch.tensor([], device=scale.device, dtype=torch.int32)
        ctx.save_for_backward(scale, rot, saved_valid_length)
        runtime_valid_length = None if saved_valid_length.numel() == 0 else saved_valid_length
        return fused.createTransformMatrix_forward(rot, scale, runtime_valid_length)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scale, rot, saved_valid_length = ctx.saved_tensors
        runtime_valid_length = None if saved_valid_length.numel() == 0 else saved_valid_length
        grad_rot, grad_scale = fused.createTransformMatrix_backward(grad_output, rot, scale, runtime_valid_length)
        return grad_scale, grad_rot, None


class _MvpTransformCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position: torch.Tensor, view_matrix: torch.Tensor, proj_matrix: torch.Tensor, valid_length: torch.Tensor | None = None):
        saved_valid_length = valid_length
        if saved_valid_length is None:
            saved_valid_length = torch.tensor([], device=position.device, dtype=torch.int32)
        view_pos, ndc_pos = fused.mvp_transform_forward(
            position,
            view_matrix,
            proj_matrix,
            None if saved_valid_length.numel() == 0 else saved_valid_length,
        )
        ctx.save_for_backward(view_pos, view_matrix, proj_matrix, saved_valid_length)
        return view_pos, ndc_pos

    @staticmethod
    def backward(ctx, grad_view_pos: torch.Tensor, grad_ndc_pos: torch.Tensor):
        grad_view_pos = grad_view_pos.contiguous()
        grad_ndc_pos = grad_ndc_pos.contiguous()
        view_pos, view_matrix, proj_matrix, saved_valid_length = ctx.saved_tensors
        grad_position = fused.mvp_transform_backward(
            grad_ndc_pos,
            grad_view_pos,
            view_matrix,
            proj_matrix,
            view_pos,
            None if saved_valid_length.numel() == 0 else saved_valid_length,
        )
        return grad_position, None, None, None


class _CreateCov2dDirectlyCuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        J: torch.Tensor,
        view_matrix: torch.Tensor,
        transform_matrix: torch.Tensor,
        valid_length: torch.Tensor | None = None,
    ):
        saved_valid_length = valid_length
        if saved_valid_length is None:
            saved_valid_length = torch.tensor([], device=transform_matrix.device, dtype=torch.int32)
        runtime_valid_length = None if saved_valid_length.numel() == 0 else saved_valid_length
        ctx.save_for_backward(J, view_matrix, transform_matrix, saved_valid_length)
        return fused.createCov2dDirectly_forward(J, view_matrix, transform_matrix, runtime_valid_length)

    @staticmethod
    def backward(ctx, grad_cov2d: torch.Tensor):
        grad_cov2d = grad_cov2d.contiguous()
        J, view_matrix, transform_matrix, saved_valid_length = ctx.saved_tensors
        runtime_valid_length = None if saved_valid_length.numel() == 0 else saved_valid_length
        transform_matrix_grad = fused.createCov2dDirectly_backward(
            grad_cov2d,
            J,
            view_matrix,
            transform_matrix,
            runtime_valid_length,
        )
        return None, None, transform_matrix_grad, None


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
    return fused.jacobianRayspace(view_pos, proj_matrix, output_shape[0], output_shape[1], valid_length)


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


def mvp_transform_script(
    position: torch.Tensor,
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
    valid_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    view_pos = torch.einsum("jn,bji->bin", position, view_matrix)
    hom_pos = torch.einsum("bjn,bji->bin", view_pos, proj_matrix)
    inv_w = torch.where(hom_pos[:, 3].abs() > 1e-12, 1.0 / hom_pos[:, 3], torch.zeros_like(hom_pos[:, 3]))
    ndc_pos = torch.empty_like(hom_pos)
    ndc_pos[:, 0] = hom_pos[:, 0] * inv_w
    ndc_pos[:, 1] = hom_pos[:, 1] * inv_w
    ndc_pos[:, 2] = hom_pos[:, 2] * inv_w
    ndc_pos[:, 3] = 1.0
    return view_pos, ndc_pos


def mvp_transform_cuda(
    position: torch.Tensor,
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
    valid_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _MvpTransformCuda.apply(position, view_matrix, proj_matrix, valid_length)


def mvp_transform(
    position: torch.Tensor,
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
    valid_length: torch.Tensor | None = None,
    *,
    backend=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform world-space homogeneous positions to view space and NDC.

    Args:
        position: World-space homogeneous positions with shape ``[4, P]``.
            The first dimension stores ``[x, y, z, w]``, and the last dimension
            indexes primitives.
        view_matrix: View matrices with shape ``[B, 4, 4]``.
        proj_matrix: Projection matrices with shape ``[B, 4, 4]``.
        valid_length: Optional active primitive count tensor. This parameter is
            only used by the CUDA backend and is ignored by the script backend.
        backend: Backend selection. ``None`` means using the current default
            backend from ``ops.backend``.

    Returns:
        A tuple ``(view_pos, ndc_pos)``.
        Both tensors have shape ``[B, 4, P]`` with layout ``[batch, component, primitive]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return mvp_transform_script(position, view_matrix, proj_matrix, valid_length)
    return mvp_transform_cuda(position, view_matrix, proj_matrix, valid_length)


def create_cov2d_directly_script(
    J: torch.Tensor,
    view_matrix: torch.Tensor,
    transform_matrix: torch.Tensor,
    valid_length: torch.Tensor | None = None,
) -> torch.Tensor:
    view3 = view_matrix[:, :3, :3].detach()
    rayspace = J[:, :, :2, :].permute(0, 3, 1, 2).detach()
    world_transform = transform_matrix.permute(2, 0, 1)
    temp0 = torch.einsum("pij,bjk->bpik", world_transform, view3)
    temp1 = torch.matmul(temp0, rayspace)
    cov2d = torch.matmul(temp1.transpose(-1, -2), temp1)
    cov2d[:, :, 0, 0] += 0.3
    cov2d[:, :, 1, 1] += 0.3
    return cov2d.permute(0, 2, 3, 1).contiguous()


def create_cov2d_directly_cuda(
    J: torch.Tensor,
    view_matrix: torch.Tensor,
    transform_matrix: torch.Tensor,
    valid_length: torch.Tensor | None = None,
) -> torch.Tensor:
    return _CreateCov2dDirectlyCuda.apply(J, view_matrix, transform_matrix, valid_length)


def create_cov2d_directly(
    J: torch.Tensor,
    view_matrix: torch.Tensor,
    transform_matrix: torch.Tensor,
    valid_length: torch.Tensor | None = None,
    *,
    backend=None,
) -> torch.Tensor:
    """
    Build 2D covariance matrices directly from ray-space Jacobians and world transforms.

    Args:
        J: Ray-space Jacobians with shape ``[N, 3, 3, P]``.
            Only the first two columns are used when projecting to screen space.
        view_matrix: View matrices with shape ``[N, 4, 4]``.
            Only the top-left ``3x3`` rotation block is used.
        transform_matrix: World transform matrices with shape ``[3, 3, P]``.
            The layout is ``[row, col, primitive]``.
        valid_length: Optional active primitive count tensor. This parameter is
            only used by the CUDA backend and is ignored by the script backend.
        backend: Backend selection. ``None`` means using the current default
            backend from ``ops.backend``.

    Returns:
        Screen-space covariance matrices with shape ``[N, 2, 2, P]``.
        The layout is ``[view, row, col, primitive]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return create_cov2d_directly_script(J, view_matrix, transform_matrix, valid_length)
    return create_cov2d_directly_cuda(J, view_matrix, transform_matrix, valid_length)
