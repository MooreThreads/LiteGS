import torch

from ..fused_backend import fused
from .backend import Backend, normalize_backend


class _GaussiansRasterCuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sorted_point_id: torch.Tensor,
        tile_start_index: torch.Tensor,
        ndc: torch.Tensor,
        cov2d_inv: torch.Tensor,
        color: torch.Tensor,
        opacities: torch.Tensor,
        tiles: torch.Tensor | None,
        img_h: int,
        img_w: int,
        tile_h: int,
        tile_w: int,
        enable_transmitance: bool = False,
        enable_depth: bool = False,
    ):
        from ..statistic_helper import StatisticsHelperInst

        img, transmitance, depth, last_contributor, packed_params, fragment_count, fragment_weight = fused.rasterize_forward(
            sorted_point_id,
            tile_start_index,
            ndc,
            cov2d_inv,
            color,
            opacities,
            tiles,
            img_h,
            img_w,
            tile_h,
            tile_w,
            StatisticsHelperInst.bStart,
            enable_transmitance,
            enable_depth,
        )

        ctx.save_for_backward(
            sorted_point_id,
            tile_start_index,
            transmitance,
            last_contributor,
            packed_params,
            tiles,
            fragment_count,
            fragment_weight,
        )
        ctx.arg_tile_size = (tile_h, tile_w)
        ctx.img_hw = (img_h, img_w)

        if not enable_depth:
            depth = None
        if not enable_transmitance:
            transmitance = None
        normal = None
        return img, transmitance, depth, normal, last_contributor

    @staticmethod
    def backward(ctx, grad_rgb_image, grad_transmitance_image, grad_depth_image, grad_normal_image, grad_last_contributor):
        from ..statistic_helper import StatisticsHelperInst

        sorted_point_id, tile_start_index, transmitance, last_contributor, packed_params, tiles, fragment_count, fragment_weight = ctx.saved_tensors
        img_h, img_w = ctx.img_hw
        tile_h, tile_w = ctx.arg_tile_size

        grad_rgb_image_max = grad_rgb_image.abs().max()
        grad_rgb_image = grad_rgb_image / grad_rgb_image_max
        grad_ndc, grad_cov2d_inv, grad_color, grad_opacities, _, grad_o_square = fused.rasterize_backward(
            sorted_point_id,
            tile_start_index,
            packed_params,
            tiles,
            transmitance,
            last_contributor,
            grad_rgb_image,
            grad_transmitance_image,
            grad_depth_image,
            grad_rgb_image_max,
            img_h,
            img_w,
            tile_h,
            tile_w,
            StatisticsHelperInst.bStart,
        )
        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.update_mean_std("fragment_weight", fragment_weight, fragment_weight * fragment_weight, fragment_count, None)
            StatisticsHelperInst.update_mean_std(
                "fragment_err",
                grad_opacities.unsqueeze(0),
                grad_o_square * grad_rgb_image_max * grad_rgb_image_max,
                fragment_count,
                None,
            )

        return (
            None,
            None,
            grad_ndc,
            grad_cov2d_inv,
            grad_color,
            grad_opacities,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_gaussians_cuda(
    sorted_point_id: torch.Tensor,
    tile_start_index: torch.Tensor,
    ndc: torch.Tensor,
    cov2d_inv: torch.Tensor,
    color: torch.Tensor,
    opacities: torch.Tensor,
    tiles: torch.Tensor | None,
    img_h: int,
    img_w: int,
    tile_h: int,
    tile_w: int,
    enable_transmitance: bool = False,
    enable_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    return _GaussiansRasterCuda.apply(
        sorted_point_id,
        tile_start_index,
        ndc,
        cov2d_inv,
        color,
        opacities,
        tiles,
        img_h,
        img_w,
        tile_h,
        tile_w,
        enable_transmitance,
        enable_depth,
    )


def rasterize_gaussians(
    sorted_point_id: torch.Tensor,
    tile_start_index: torch.Tensor,
    ndc: torch.Tensor,
    cov2d_inv: torch.Tensor,
    color: torch.Tensor,
    opacities: torch.Tensor,
    tiles: torch.Tensor | None,
    img_h: int,
    img_w: int,
    tile_h: int,
    tile_w: int,
    enable_transmitance: bool = False,
    enable_depth: bool = False,
    *,
    backend=None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """
    Rasterize sorted Gaussians into screen-space images.

    Args:
        sorted_point_id: Sorted primitive ids with shape ``[N, F]`` and dtype ``int32``.
        tile_start_index: Tile range table with shape ``[N, tiles_num + 2]`` and dtype ``int32``.
        ndc: Homogeneous NDC positions with shape ``[N, 4, P]``.
        cov2d_inv: Inverse 2D covariance matrices with shape ``[N, 2, 2, P]``.
        color: Per-view RGB values with shape ``[N, 3, P]``.
        opacities: Per-view opacity values with shape ``[N, P]``.
        tiles: Optional cached tile list tensor used by the fused backend.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        tile_h: Tile height in pixels.
        tile_w: Tile width in pixels.
        enable_transmitance: Whether to return the transmitance image.
        enable_depth: Whether to return the depth image.
        backend: Backend selection. ``SCRIPT`` is not implemented for this operator.

    Returns:
        A tuple ``(img, transmitance, depth, normal, last_contributor)``.
        ``img`` has shape ``[N, 3, H, W]``.
        ``transmitance`` is either ``None`` or shape ``[N, 1, H, W]``.
        ``depth`` is either ``None`` or shape ``[N, 1, H, W]``.
        ``normal`` is currently always ``None``.
        ``last_contributor`` has shape ``[N, 1, H, W]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        raise NotImplementedError("rasterize_gaussians does not provide a script backend")
    return rasterize_gaussians_cuda(
        sorted_point_id,
        tile_start_index,
        ndc,
        cov2d_inv,
        color,
        opacities,
        tiles,
        img_h,
        img_w,
        tile_h,
        tile_w,
        enable_transmitance,
        enable_depth,
    )
