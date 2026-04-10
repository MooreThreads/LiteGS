import torch
from dataclasses import dataclass

from ..arguments import DensifyParams
from ..utils.statistic_helper import StatisticsHelperInst
from ..scene import cluster
from .. import utils


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
    prune_index: torch.Tensor | None = None
    append_params: GaussianParams | None = None
    opacity_override: torch.Tensor | None = None
    clear_optimizer_state: bool = False
    changed: bool = False


class DensityControllerBase:
    def __init__(self, densify_params: DensifyParams, chunk_size: int) -> None:
        self.densify_params = densify_params
        self.chunk_size = chunk_size

    @torch.no_grad()
    def step(self, params: GaussianParams, epoch: int) -> DensifyEdits:
        return DensifyEdits()

    @staticmethod
    def _fix_cluster_edits(edits: DensifyEdits | None, chunk_size: int) -> None:
        if edits is None:
            return None

        if edits.append_params is not None:
            append_count = edits.append_params.xyz.shape[-1]
            aligned_count = (append_count // chunk_size) * chunk_size
            
            edits.append_params.xyz=edits.append_params.xyz[...,:aligned_count]
            edits.append_params.scale=edits.append_params.scale[...,:aligned_count]
            edits.append_params.rot=edits.append_params.rot[...,:aligned_count]
            edits.append_params.sh_0=edits.append_params.sh_0[...,:aligned_count]
            edits.append_params.sh_rest=edits.append_params.sh_rest[...,:aligned_count]
            edits.append_params.opacity=edits.append_params.opacity[...,:aligned_count]

        if edits.prune_index is not None:
            prune_num=edits.prune_index.shape[0]
            prune_num = (prune_num // chunk_size) * chunk_size
            edits.prune_index=edits.prune_index[:prune_num]

        return



class DensityControllerOfficial(DensityControllerBase):
    @torch.no_grad()
    def __init__(self, screen_extent: float, densify_params: DensifyParams, chunk_size: int, init_points_num: int) -> None:
        self.grad_threshold = densify_params.densify_grad_threshold
        self.min_opacity = densify_params.opacity_threshold
        self.percent_dense = densify_params.percent_dense
        self.screen_extent = screen_extent
        self.max_screen_size = densify_params.screen_size_threshold
        self.init_points_num = init_points_num
        super().__init__(densify_params, chunk_size)

    @torch.no_grad()
    def get_prune_index(self, actived_opacity: torch.Tensor, actived_scale: torch.Tensor) -> torch.Tensor:
        transparent = (actived_opacity < self.min_opacity).squeeze()
        invisible = StatisticsHelperInst.get_global_culling()
        return invisible.nonzero()[:,0]

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
        transform_matrix = utils.ops.create_transform_matrix(
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
    def reset_opacity(self, opacity_params: torch.Tensor, epoch: int, append_opacity_params:torch.Tensor=None) -> torch.Tensor | None:
        def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x / (1 - x))
        if append_opacity_params is not None:
            opacity_params=torch.cat([opacity_params,append_opacity_params],dim=-1)
        actived_opacities = opacity_params.sigmoid()
        if self.densify_params.opacity_reset_mode == 'decay':
            decay_rate = 0.5
            return _inverse_sigmoid((actived_opacities * decay_rate).clamp_min(1.0 / 128))
        if self.densify_params.opacity_reset_mode == 'reset':
            return _inverse_sigmoid(actived_opacities.clamp_max(0.005))
        return None

    @torch.no_grad()
    def is_densify_actived(self, epoch: int):
        return epoch < self.densify_params.end and epoch >= self.densify_params.start and (epoch % self.densify_params.interval == 0)

    @torch.no_grad()
    def step(self, params: GaussianParams, epoch: int) -> DensifyEdits:
        edits = DensifyEdits()
        if not (epoch < self.densify_params.end and epoch >= self.densify_params.start):
            return edits

        if epoch % self.densify_params.interval == 0:
            append_params = self.split_and_clone(params, epoch)
            prune_index = self.get_prune_index(params.opacity.sigmoid(), params.scale.exp())
            edits.append_params = append_params
            edits.prune_index = prune_index
            edits.changed = edits.changed or True
        
        if self.chunk_size>0 and edits.changed:
            self._fix_cluster_edits(edits,self.chunk_size)

        if epoch % self.densify_params.opacity_reset_interval == 0:
            append_opacity_params=None
            if edits.append_params is not None:
                append_opacity_params=edits.append_params.opacity
            edits.opacity_override = self.reset_opacity(params.opacity, epoch, append_opacity_params)
            edits.clear_optimizer_state = self.densify_params.opacity_reset_mode == 'decay'
            edits.changed = edits.changed or True

        return edits


class DensityControllerTamingGS(DensityControllerOfficial):
    @torch.no_grad()
    def __init__(self, screen_extent: int, densify_params: DensifyParams, chunk_size: int, init_points_num: int) -> None:
        assert densify_params.target_primitives != 0.0
        self.target_points_num = densify_params.target_primitives
        super().__init__(screen_extent, densify_params, chunk_size, init_points_num)

    @torch.no_grad()
    def get_prune_index(self, actived_opacity: torch.Tensor, actived_scale: torch.Tensor) -> torch.Tensor:
        if self.densify_params.prune_mode == 'weight':
            prune_mask = torch.zeros(actived_opacity.shape[1], device=actived_opacity.device).bool()
            frag_weight, frag_count = StatisticsHelperInst.get_mean('fragment_weight')
            weight_sum = (frag_weight * frag_count).nan_to_num(0).squeeze()
            return (weight_sum == 0).nonzero()[:,0]
        if self.densify_params.prune_mode == 'threshold':
            return super().get_prune_index(actived_opacity, actived_scale)
        assert(False)
        return None

    def get_score(self, xyz, scale, rot, sh_0, sh_rest, opacity) -> torch.Tensor:
        var, frag_count = StatisticsHelperInst.get_var('fragment_err')
        score = var * frag_count * (opacity.sigmoid() * opacity.sigmoid())
        score = score.squeeze().nan_to_num(0)
        score.clamp_min_(0)
        return score

    @torch.no_grad()
    def split_and_clone(self, params: GaussianParams, epoch: int) -> GaussianParams | None:
        prune_num = self.get_prune_index(params.opacity.sigmoid(), params.scale.exp()).shape[0]

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
        transform_matrix = utils.ops.create_transform_matrix(
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
