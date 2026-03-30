import torch
from dataclasses import dataclass

from ..arguments import DensifyParams
from ..utils.statistic_helper import StatisticsHelperInst
from ..scene import cluster
from ..utils import wrapper


@dataclass
class GaussianParams:
    xyz: torch.Tensor
    scale: torch.Tensor
    rot: torch.Tensor
    sh_0: torch.Tensor
    sh_rest: torch.Tensor
    opacity: torch.Tensor


@dataclass
class DensifyEdits:
    keep_mask: torch.Tensor | None = None
    append_params: GaussianParams | None = None
    opacity_override: torch.Tensor | None = None
    clear_optimizer_state: bool = False
    changed: bool = False
    stats_need_reset: bool = False


class DensityControllerBase:
    def __init__(self, densify_params: DensifyParams, bCluster: bool) -> None:
        self.densify_params = densify_params
        self.bCluster = bCluster

    @torch.no_grad()
    def step(self, params: GaussianParams, epoch: int) -> DensifyEdits:
        return DensifyEdits()

    @staticmethod
    def _concat_params(lhs: GaussianParams, rhs: GaussianParams | None) -> GaussianParams:
        if rhs is None:
            return lhs
        return GaussianParams(
            xyz=torch.cat((lhs.xyz, rhs.xyz), dim=-1),
            scale=torch.cat((lhs.scale, rhs.scale), dim=-1),
            rot=torch.cat((lhs.rot, rhs.rot), dim=-1),
            sh_0=torch.cat((lhs.sh_0, rhs.sh_0), dim=-1),
            sh_rest=torch.cat((lhs.sh_rest, rhs.sh_rest), dim=-1),
            opacity=torch.cat((lhs.opacity, rhs.opacity), dim=-1),
        )

    @staticmethod
    def _slice_params(params: GaussianParams, keep_mask: torch.Tensor | None) -> GaussianParams:
        if keep_mask is None:
            return params
        return GaussianParams(
            xyz=params.xyz[..., keep_mask],
            scale=params.scale[..., keep_mask],
            rot=params.rot[..., keep_mask],
            sh_0=params.sh_0[..., keep_mask],
            sh_rest=params.sh_rest[..., keep_mask],
            opacity=params.opacity[..., keep_mask],
        )

    @staticmethod
    def _replace_opacity(params: GaussianParams, opacity: torch.Tensor | None) -> GaussianParams:
        if opacity is None:
            return params
        return GaussianParams(
            xyz=params.xyz,
            scale=params.scale,
            rot=params.rot,
            sh_0=params.sh_0,
            sh_rest=params.sh_rest,
            opacity=opacity,
        )

    @staticmethod
    def _append_count(params: GaussianParams | None) -> int:
        if params is None:
            return 0
        return int(params.xyz.shape[-1])

    @staticmethod
    def _normalize_cluster_keep_mask(keep_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
        delete_mask = ~keep_mask
        delete_count = int(delete_mask.sum().item())
        aligned_delete_count = (delete_count // chunk_size) * chunk_size
        if aligned_delete_count == delete_count:
            return keep_mask

        normalized_keep_mask = torch.ones_like(keep_mask, dtype=torch.bool)
        if aligned_delete_count > 0:
            delete_indices = delete_mask.nonzero()[:aligned_delete_count, 0]
            normalized_keep_mask[delete_indices] = False
        return normalized_keep_mask

    @staticmethod
    def _truncate_append_params(append_params: GaussianParams | None, chunk_size: int) -> GaussianParams | None:
        if append_params is None:
            return None

        append_count = append_params.xyz.shape[-1]
        aligned_count = (append_count // chunk_size) * chunk_size
        if aligned_count == 0:
            return None
        if aligned_count == append_count:
            return append_params

        keep_mask = torch.zeros(
            append_count,
            dtype=torch.bool,
            device=append_params.xyz.device
        )
        keep_mask[:aligned_count] = True
        return DensityControllerBase._slice_params(append_params, keep_mask)

    def _to_unclustered_params(self, params: GaussianParams) -> tuple[GaussianParams, int | None]:
        if not self.bCluster:
            return params, None

        chunk_size = params.xyz.shape[-1]
        xyz, scale, rot, sh_0, sh_rest, opacity = cluster.uncluster(
            params.xyz, params.scale, params.rot, params.sh_0, params.sh_rest, params.opacity
        )
        return GaussianParams(xyz, scale, rot, sh_0, sh_rest, opacity), chunk_size

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1 - x))


class DensityControllerOfficial(DensityControllerBase):
    @torch.no_grad()
    def __init__(self, screen_extent: float, densify_params: DensifyParams, bCluster: bool, init_points_num: int) -> None:
        self.grad_threshold = densify_params.densify_grad_threshold
        self.min_opacity = densify_params.opacity_threshold
        self.percent_dense = densify_params.percent_dense
        self.screen_extent = screen_extent
        self.max_screen_size = densify_params.screen_size_threshold
        self.init_points_num = init_points_num
        super().__init__(densify_params, bCluster)

    @torch.no_grad()
    def get_prune_mask(self, actived_opacity: torch.Tensor, actived_scale: torch.Tensor) -> torch.Tensor:
        transparent = (actived_opacity < self.min_opacity).squeeze()
        invisible = StatisticsHelperInst.get_global_culling()
        prune_mask = transparent
        prune_mask[:invisible.shape[0]] |= invisible
        return prune_mask

    @torch.no_grad()
    def get_keep_mask(self, actived_opacity: torch.Tensor, actived_scale: torch.Tensor) -> torch.Tensor:
        return ~self.get_prune_mask(actived_opacity, actived_scale)

    @torch.no_grad()
    def get_clone_mask(self, actived_scale: torch.Tensor) -> torch.Tensor:
        mean2d_grads = StatisticsHelperInst.get_mean('mean2d_grad').squeeze()
        abnormal_mask = mean2d_grads >= self.grad_threshold
        tiny_pts_mask = actived_scale.max(dim=0).values <= self.percent_dense * self.screen_extent
        return abnormal_mask & tiny_pts_mask

    @torch.no_grad()
    def get_split_mask(self, actived_scale: torch.Tensor, N: int = 2) -> torch.Tensor:
        mean2d_grads = StatisticsHelperInst.get_mean('mean2d_grad').squeeze()
        abnormal_mask = mean2d_grads >= self.grad_threshold
        large_pts_mask = actived_scale.max(dim=0).values > self.percent_dense * self.screen_extent
        return abnormal_mask & large_pts_mask

    @torch.no_grad()
    def split_and_clone(self, params: GaussianParams, epoch: int) -> GaussianParams | None:
        clone_mask = self.get_clone_mask(params.scale.exp())
        split_mask = self.get_split_mask(params.scale.exp())

        stds = params.scale[..., split_mask].exp()
        means = torch.zeros((3, stds.size(-1)), device=params.xyz.device)
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix = wrapper.CreateTransformMatrix.call_fused(
            torch.ones_like(params.scale[..., split_mask].exp()),
            torch.nn.functional.normalize(params.rot[..., split_mask], dim=0)
        )
        transform_matrix = transform_matrix[:3, :3]
        shift = (samples.permute(2, 0, 1)) @ transform_matrix.permute(2, 0, 1)
        shift = shift.permute(1, 2, 0).squeeze(0)

        split_xyz = params.xyz[..., split_mask] + shift
        clone_xyz = params.xyz[..., clone_mask]
        append_xyz = torch.cat((split_xyz, clone_xyz), dim=-1)
        if append_xyz.shape[-1] == 0:
            return None

        split_scale = (params.scale[..., split_mask].exp() / (0.8 * 2)).log()
        clone_scale = params.scale[..., clone_mask]
        append_scale = torch.cat((split_scale, clone_scale), dim=-1)

        split_rot = params.rot[..., split_mask]
        clone_rot = params.rot[..., clone_mask]
        append_rot = torch.cat((split_rot, clone_rot), dim=-1)

        split_sh_0 = params.sh_0[..., split_mask]
        clone_sh_0 = params.sh_0[..., clone_mask]
        append_sh_0 = torch.cat((split_sh_0, clone_sh_0), dim=-1)

        split_sh_rest = params.sh_rest[..., split_mask]
        clone_sh_rest = params.sh_rest[..., clone_mask]
        append_sh_rest = torch.cat((split_sh_rest, clone_sh_rest), dim=-1)

        split_opacity = params.opacity[..., split_mask]
        clone_opacity = params.opacity[..., clone_mask]
        append_opacity = torch.cat((split_opacity, clone_opacity), dim=-1)

        return GaussianParams(
            append_xyz,
            append_scale,
            append_rot,
            append_sh_0,
            append_sh_rest,
            append_opacity,
        )

    @torch.no_grad()
    def reset_opacity(self, params: GaussianParams, epoch: int) -> torch.Tensor | None:
        actived_opacities = params.opacity.sigmoid()
        if self.densify_params.opacity_reset_mode == 'decay':
            decay_rate = 0.5
            return self._inverse_sigmoid((actived_opacities * decay_rate).clamp_min(1.0 / 128))
        if self.densify_params.opacity_reset_mode == 'reset':
            return self._inverse_sigmoid(actived_opacities.clamp_max(0.005))
        return None

    @torch.no_grad()
    def is_densify_actived(self, epoch: int):
        return epoch < self.densify_params.end and epoch >= self.densify_params.start and (
            epoch % self.densify_params.interval == 0
        )

    @torch.no_grad()
    def step(self, params: GaussianParams, epoch: int) -> DensifyEdits:
        edits = DensifyEdits()
        if not (epoch < self.densify_params.end and epoch >= self.densify_params.start):
            return edits

        working_params, chunk_size = self._to_unclustered_params(params)

        if epoch % self.densify_params.interval == 0:
            append_params = self.split_and_clone(working_params, epoch)
            if chunk_size is not None:
                append_params = self._truncate_append_params(append_params, chunk_size)

            merged_params = self._concat_params(working_params, append_params)
            keep_mask = self.get_keep_mask(merged_params.opacity.sigmoid(), merged_params.scale.exp())
            if chunk_size is not None:
                keep_mask = self._normalize_cluster_keep_mask(keep_mask, chunk_size)

            edits.append_params = append_params
            edits.keep_mask = keep_mask
            working_params = self._slice_params(merged_params, keep_mask)
            edits.changed = edits.changed or self._append_count(append_params) > 0 or not keep_mask.all().item()

        if epoch % self.densify_params.opacity_reset_interval == 0:
            edits.opacity_override = self.reset_opacity(working_params, epoch)
            edits.clear_optimizer_state = self.densify_params.opacity_reset_mode == 'decay'
            edits.changed = edits.changed or edits.opacity_override is not None

        if edits.changed:
            edits.stats_need_reset = True

        return edits


class DensityControllerTamingGS(DensityControllerOfficial):
    @torch.no_grad()
    def __init__(self, screen_extent: int, densify_params: DensifyParams, bCluster: bool, init_points_num: int) -> None:
        assert densify_params.target_primitives != 0.0
        self.target_points_num = densify_params.target_primitives
        super().__init__(screen_extent, densify_params, bCluster, init_points_num)

    @torch.no_grad()
    def get_prune_mask(self, actived_opacity: torch.Tensor, actived_scale: torch.Tensor) -> torch.Tensor:
        if self.densify_params.prune_mode == 'weight':
            prune_mask = torch.zeros(actived_opacity.shape[1], device=actived_opacity.device).bool()
            frag_weight, frag_count = StatisticsHelperInst.get_mean('fragment_weight')
            weight_sum = (frag_weight * frag_count).nan_to_num(0).squeeze()
            invisible = weight_sum == 0
            prune_mask[:invisible.shape[0]] |= invisible
            return prune_mask
        if self.densify_params.prune_mode == 'threshold':
            return super().get_prune_mask(actived_opacity, actived_scale)
        return torch.zeros(actived_opacity.shape[1], device=actived_opacity.device).bool()

    def get_score(self, xyz, scale, rot, sh_0, sh_rest, opacity) -> torch.Tensor:
        var, frag_count = StatisticsHelperInst.get_var('fragment_err')
        score = var * frag_count * (opacity.sigmoid() * opacity.sigmoid())
        score = score.squeeze().nan_to_num(0)
        score.clamp_min_(0)
        return score

    @torch.no_grad()
    def split_and_clone(self, params: GaussianParams, epoch: int) -> GaussianParams | None:
        prune_num = int(self.get_prune_mask(params.opacity.sigmoid(), params.scale.exp()).sum().item())

        cur_target_count = (
            (self.target_points_num - self.init_points_num)
            / (self.densify_params.end - self.densify_params.start)
            * (epoch - self.densify_params.start)
            + self.init_points_num
        )
        budget = min(max(int(cur_target_count - params.xyz.shape[-1]), 1) + prune_num, params.xyz.shape[-1])

        score = self.get_score(params.xyz, params.scale, params.rot, params.sh_0, params.sh_rest, params.opacity)
        densify_index = torch.multinomial(score, budget, replacement=False)
        clone_index = densify_index[
            (params.scale[:, densify_index].exp().max(dim=0).values <= self.percent_dense * self.screen_extent)
        ]
        split_index = densify_index[
            (params.scale[:, densify_index].exp().max(dim=0).values > self.percent_dense * self.screen_extent)
        ]

        stds = params.scale[..., split_index].exp()
        means = torch.zeros((3, stds.size(-1)), device=params.xyz.device)
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix = wrapper.CreateTransformMatrix.call_fused(
            torch.ones_like(params.scale[..., split_index]),
            torch.nn.functional.normalize(params.rot[..., split_index], dim=0)
        )
        transform_matrix = transform_matrix[:3, :3]
        shift = (samples.permute(2, 0, 1)) @ transform_matrix.permute(2, 0, 1)
        shift = shift.permute(1, 2, 0).squeeze(0)

        split_xyz = params.xyz[..., split_index] + shift
        clone_xyz = params.xyz[..., clone_index]
        append_xyz = torch.cat((split_xyz, clone_xyz), dim=-1)
        if append_xyz.shape[-1] == 0:
            return None

        split_scale = (params.scale[..., split_index].exp() / (0.8 * 2)).log()
        clone_scale = params.scale[..., clone_index]
        append_scale = torch.cat((split_scale, clone_scale), dim=-1)

        split_rot = params.rot[..., split_index]
        clone_rot = params.rot[..., clone_index]
        append_rot = torch.cat((split_rot, clone_rot), dim=-1)

        split_sh_0 = params.sh_0[..., split_index]
        clone_sh_0 = params.sh_0[..., clone_index]
        append_sh_0 = torch.cat((split_sh_0, clone_sh_0), dim=-1)

        split_sh_rest = params.sh_rest[..., split_index]
        clone_sh_rest = params.sh_rest[..., clone_index]
        append_sh_rest = torch.cat((split_sh_rest, clone_sh_rest), dim=-1)

        split_opacity = params.opacity[..., split_index]
        clone_opacity = params.opacity[..., clone_index]
        append_opacity = torch.cat((split_opacity, clone_opacity), dim=-1)

        return GaussianParams(
            append_xyz,
            append_scale,
            append_rot,
            append_sh_0,
            append_sh_rest,
            append_opacity,
        )
