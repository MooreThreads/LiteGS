import torch

from ..fused_backend import fused
from ..spherical_harmonics import sh_to_rgb
from .backend import Backend, normalize_backend


class _SphericalHarmonicToRgbCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, degree: int, sh_base: torch.Tensor, sh_rest: torch.Tensor, dirs: torch.Tensor):
        ctx.degree = degree
        ctx.sh_rest_dim = sh_rest.shape[0]
        ctx.save_for_backward(dirs, sh_base, sh_rest)
        return fused.sh2rgb_forward(degree, sh_base, sh_rest, dirs)

    @staticmethod
    def backward(ctx, grad_rgb: torch.Tensor):
        grad_rgb = grad_rgb.contiguous()
        dirs, sh_base, sh_rest = ctx.saved_tensors
        sh_base_grad, sh_rest_grad, dir_grad = fused.sh2rgb_backward(
            ctx.degree,
            grad_rgb,
            ctx.sh_rest_dim,
            dirs,
            sh_base,
            sh_rest,
        )
        return None, sh_base_grad, sh_rest_grad, dir_grad


def spherical_harmonic_to_rgb_script(
    degree: int,
    sh_base: torch.Tensor,
    sh_rest: torch.Tensor,
    dirs: torch.Tensor,
) -> torch.Tensor:
    return sh_to_rgb(degree, torch.cat((sh_base, sh_rest), dim=0), dirs)


def spherical_harmonic_to_rgb_cuda(
    degree: int,
    sh_base: torch.Tensor,
    sh_rest: torch.Tensor,
    dirs: torch.Tensor,
) -> torch.Tensor:
    return _SphericalHarmonicToRgbCuda.apply(degree, sh_base, sh_rest, dirs)


def spherical_harmonic_to_rgb(
    degree: int,
    sh_base: torch.Tensor,
    sh_rest: torch.Tensor,
    dirs: torch.Tensor,
    *,
    backend=None,
) -> torch.Tensor:
    """
    Evaluate spherical harmonics and convert them to RGB using the selected backend.

    Args:
        degree: Spherical harmonic degree. The current implementation supports degrees 0 to 3.
        sh_base: DC spherical harmonic coefficients with shape ``[1, 3, P]``.
            The layout is ``[coefficient, channel, primitive]``.
        sh_rest: Remaining spherical harmonic coefficients with shape ``[(degree + 1) ** 2 - 1, 3, P]``.
            The layout is ``[coefficient, channel, primitive]``.
        dirs: Unit view directions with shape ``[N, 3, P]``.
            The layout is ``[view, axis, primitive]``.
        backend: Backend selection. ``None`` means using the current default backend from
            ``ops.backend``.

    Returns:
        RGB values with shape ``[N, 3, P]``.
        The layout is ``[view, channel, primitive]``.
    """
    selected_backend = normalize_backend(backend)
    if selected_backend == Backend.SCRIPT:
        return spherical_harmonic_to_rgb_script(degree, sh_base, sh_rest, dirs)
    return spherical_harmonic_to_rgb_cuda(degree, sh_base, sh_rest, dirs)
