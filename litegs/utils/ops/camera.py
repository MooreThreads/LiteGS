import torch

from ..fused_backend import fused
from .backend import Backend, normalize_backend


@torch.no_grad()
def _viewproj_to_frustumplane(viewproj_matrix: torch.Tensor) -> torch.Tensor:
    frustumplane = torch.zeros((viewproj_matrix.shape[0], 6, 4), device=viewproj_matrix.device, dtype=viewproj_matrix.dtype)
    frustumplane[:, 0, 0] = viewproj_matrix[:, 0, 3] + viewproj_matrix[:, 0, 0]
    frustumplane[:, 0, 1] = viewproj_matrix[:, 1, 3] + viewproj_matrix[:, 1, 0]
    frustumplane[:, 0, 2] = viewproj_matrix[:, 2, 3] + viewproj_matrix[:, 2, 0]
    frustumplane[:, 0, 3] = viewproj_matrix[:, 3, 3] + viewproj_matrix[:, 3, 0]
    frustumplane[:, 1, 0] = viewproj_matrix[:, 0, 3] - viewproj_matrix[:, 0, 0]
    frustumplane[:, 1, 1] = viewproj_matrix[:, 1, 3] - viewproj_matrix[:, 1, 0]
    frustumplane[:, 1, 2] = viewproj_matrix[:, 2, 3] - viewproj_matrix[:, 2, 0]
    frustumplane[:, 1, 3] = viewproj_matrix[:, 3, 3] - viewproj_matrix[:, 3, 0]
    frustumplane[:, 2, 0] = viewproj_matrix[:, 0, 3] + viewproj_matrix[:, 0, 1]
    frustumplane[:, 2, 1] = viewproj_matrix[:, 1, 3] + viewproj_matrix[:, 1, 1]
    frustumplane[:, 2, 2] = viewproj_matrix[:, 2, 3] + viewproj_matrix[:, 2, 1]
    frustumplane[:, 2, 3] = viewproj_matrix[:, 3, 3] + viewproj_matrix[:, 3, 1]
    frustumplane[:, 3, 0] = viewproj_matrix[:, 0, 3] - viewproj_matrix[:, 0, 1]
    frustumplane[:, 3, 1] = viewproj_matrix[:, 1, 3] - viewproj_matrix[:, 1, 1]
    frustumplane[:, 3, 2] = viewproj_matrix[:, 2, 3] - viewproj_matrix[:, 2, 1]
    frustumplane[:, 3, 3] = viewproj_matrix[:, 3, 3] - viewproj_matrix[:, 3, 1]
    frustumplane[:, 4, 0] = viewproj_matrix[:, 0, 2]
    frustumplane[:, 4, 1] = viewproj_matrix[:, 1, 2]
    frustumplane[:, 4, 2] = viewproj_matrix[:, 2, 2]
    frustumplane[:, 4, 3] = viewproj_matrix[:, 3, 2]
    frustumplane[:, 5, 0] = viewproj_matrix[:, 0, 3] - viewproj_matrix[:, 0, 2]
    frustumplane[:, 5, 1] = viewproj_matrix[:, 1, 3] - viewproj_matrix[:, 1, 2]
    frustumplane[:, 5, 2] = viewproj_matrix[:, 2, 3] - viewproj_matrix[:, 2, 2]
    frustumplane[:, 5, 3] = viewproj_matrix[:, 3, 3] - viewproj_matrix[:, 3, 2]
    return frustumplane


class _CreateViewProjCuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        view_params: torch.Tensor,
        proj_params: torch.Tensor,
        img_h: int,
        img_w: int,
        z_near: float,
        z_far: float,
    ):
        view_matrix, proj_matrix, viewproj_matrix, frustumplane = fused.create_viewproj_forward(
            view_params,
            proj_params,
            img_h,
            img_w,
            z_near,
            z_far,
        )
        ctx.save_for_backward(view_params, proj_params)
        ctx.img_h = img_h
        ctx.img_w = img_w
        ctx.z_near = z_near
        ctx.z_far = z_far
        return view_matrix, proj_matrix, viewproj_matrix, frustumplane

    @staticmethod
    def backward(ctx, view_matrix_grad, proj_matrix_grad, viewproj_matrix_grad, frustumplane_grad):
        view_params, proj_params = ctx.saved_tensors
        view_params_grad, proj_params_grad = fused.create_viewproj_backward(
            view_matrix_grad.contiguous(),
            proj_matrix_grad.contiguous(),
            viewproj_matrix_grad.contiguous(),
            view_params,
            proj_params,
            ctx.img_h,
            ctx.img_w,
            ctx.z_near,
            ctx.z_far,
        )
        return view_params_grad, proj_params_grad, None, None, None, None


def create_viewproj_script(
    view_params: torch.Tensor,
    proj_params: torch.Tensor,
    img_h: int,
    img_w: int,
    z_near: float,
    z_far: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    quat = torch.nn.functional.normalize(view_params[:, :4], dim=-1)
    r = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    view_matrix = torch.zeros((view_params.shape[0], 4, 4), device=view_params.device, dtype=view_params.dtype)
    view_matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    view_matrix[:, 0, 1] = 2 * (x * y + r * z)
    view_matrix[:, 0, 2] = 2 * (x * z - r * y)
    view_matrix[:, 1, 0] = 2 * (x * y - r * z)
    view_matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    view_matrix[:, 1, 2] = 2 * (y * z + r * x)
    view_matrix[:, 2, 0] = 2 * (x * z + r * y)
    view_matrix[:, 2, 1] = 2 * (y * z - r * x)
    view_matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    view_matrix[:, 3, :3] = view_params[:, 4:7]
    view_matrix[:, 3, 3] = 1.0

    proj_00 = proj_params.reshape(-1)[0]
    proj_11 = proj_00 * img_w / img_h
    proj_matrix = torch.zeros((view_params.shape[0], 4, 4), device=view_params.device, dtype=view_params.dtype)
    proj_matrix[:, 0, 0] = proj_00
    proj_matrix[:, 1, 1] = proj_11
    proj_matrix[:, 2, 2] = z_far / (z_far - z_near)
    proj_matrix[:, 2, 3] = 1.0
    proj_matrix[:, 3, 2] = -z_far * z_near / (z_far - z_near)

    viewproj_matrix = torch.matmul(view_matrix, proj_matrix)
    frustumplane = _viewproj_to_frustumplane(viewproj_matrix)
    return view_matrix, proj_matrix, viewproj_matrix, frustumplane


def create_viewproj_cuda(
    view_params: torch.Tensor,
    proj_params: torch.Tensor,
    img_h: int,
    img_w: int,
    z_near: float,
    z_far: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _CreateViewProjCuda.apply(view_params, proj_params, img_h, img_w, z_near, z_far)


def create_viewproj(
    view_params: torch.Tensor,
    proj_params: torch.Tensor,
    img_h: int,
    img_w: int,
    z_near: float,
    z_far: float,
    *,
    backend=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build view, projection, view-projection, and frustum-plane tensors from compact camera parameters.

    Args:
        view_params: Camera extrinsic parameters with shape ``[N, 7]``.
            The layout is ``[view, (r, x, y, z, tx, ty, tz)]`` where the quaternion is
            normalized internally and translation uses row-vector convention.
        proj_params: Shared projection parameter tensor containing ``recp_tan_half_fov_x``.
            The current implementation expects a single scalar value, typically stored as
            shape ``[1]``.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        z_near: Near plane distance.
        z_far: Far plane distance.
        backend: Backend selection. ``None`` means using the current default backend from
            ``ops.backend``.

    Returns:
        A tuple ``(view_matrix, proj_matrix, viewproj_matrix, frustumplane)``.
        ``view_matrix`` has shape ``[N, 4, 4]``.
        ``proj_matrix`` has shape ``[N, 4, 4]``.
        ``viewproj_matrix`` has shape ``[N, 4, 4]``.
        ``frustumplane`` has shape ``[N, 6, 4]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return create_viewproj_script(view_params, proj_params, img_h, img_w, z_near, z_far)
    return create_viewproj_cuda(view_params, proj_params, img_h, img_w, z_near, z_far)
