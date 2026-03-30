import torch
import torch.nn as nn

from .densify import GaussianParams
from ..scene import cluster


def collect_named_parameters(optimizer_list: list[torch.optim.Optimizer | None]) -> dict[str, nn.Parameter]:
    param_dict: dict[str, nn.Parameter] = {}
    for optimizer in optimizer_list:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            if not group.get("params"):
                continue
            param_dict[group["name"]] = group["params"][0]
    return param_dict


def prune_(
    optimizer_list: list[torch.optim.Optimizer | None],
    keep_mask: torch.Tensor,
    bcluster: bool
) -> dict[str, nn.Parameter]:
    if keep_mask is None:
        return collect_named_parameters(optimizer_list)

    normalized_keep_mask = keep_mask
    if bcluster:
        ref_param = next(iter(collect_named_parameters(optimizer_list).values()))
        normalized_keep_mask = _normalize_cluster_keep_mask(keep_mask, ref_param.shape[-1])

    for optimizer in optimizer_list:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            old_param = group["params"][0]
            stored_state = optimizer.state.pop(old_param, None)
            new_tensor = _prune_tensor(old_param.detach(), normalized_keep_mask, bcluster)
            new_param = nn.Parameter(new_tensor.requires_grad_(True))
            group["params"][0] = new_param
            if stored_state is not None:
                optimizer.state[new_param] = _remap_state(stored_state, old_param, new_tensor, normalized_keep_mask, None, bcluster)

    return collect_named_parameters(optimizer_list)


def append_(
    optimizer_list: list[torch.optim.Optimizer | None],
    append_params: GaussianParams,
    bcluster: bool
) -> dict[str, nn.Parameter]:
    if append_params is None:
        return collect_named_parameters(optimizer_list)

    if append_params.xyz.shape[-1] == 0:
        return collect_named_parameters(optimizer_list)

    normalized_append = append_params
    if bcluster:
        ref_param = next(iter(collect_named_parameters(optimizer_list).values()))
        normalized_append = _truncate_append_params(append_params, ref_param.shape[-1])
        if normalized_append is None:
            return collect_named_parameters(optimizer_list)

    append_dict = {
        "xyz": normalized_append.xyz,
        "scale": normalized_append.scale,
        "rot": normalized_append.rot,
        "sh_0": normalized_append.sh_0,
        "sh_rest": normalized_append.sh_rest,
        "opacity": normalized_append.opacity,
    }

    for optimizer in optimizer_list:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            old_param = group["params"][0]
            stored_state = optimizer.state.pop(old_param, None)
            append_tensor = append_dict[group["name"]]
            new_tensor = _append_tensor(old_param.detach(), append_tensor, bcluster)
            new_param = nn.Parameter(new_tensor.requires_grad_(True))
            group["params"][0] = new_param
            if stored_state is not None:
                optimizer.state[new_param] = _remap_state(stored_state, old_param, new_tensor, None, append_tensor, bcluster)

    return collect_named_parameters(optimizer_list)


def replace_(
    optimizer_list: list[torch.optim.Optimizer | None],
    name: str,
    tensor: torch.Tensor,
    bcluster: bool
) -> dict[str, nn.Parameter]:
    for optimizer in optimizer_list:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            if group["name"] != name:
                continue

            old_param = group["params"][0]
            stored_state = optimizer.state.pop(old_param, None)
            new_tensor = _replace_tensor(old_param.detach(), tensor, bcluster)
            new_param = nn.Parameter(new_tensor.requires_grad_(True))
            group["params"][0] = new_param
            if stored_state is not None:
                optimizer.state[new_param] = _replace_state(stored_state, old_param, new_tensor)

    return collect_named_parameters(optimizer_list)


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


def _truncate_append_params(append_params: GaussianParams, chunk_size: int) -> GaussianParams | None:
    append_count = append_params.xyz.shape[-1]
    aligned_count = (append_count // chunk_size) * chunk_size
    if aligned_count == 0:
        return None
    if aligned_count == append_count:
        return append_params

    keep_mask = torch.zeros(append_count, dtype=torch.bool, device=append_params.xyz.device)
    keep_mask[:aligned_count] = True
    return GaussianParams(
        xyz=append_params.xyz[..., keep_mask],
        scale=append_params.scale[..., keep_mask],
        rot=append_params.rot[..., keep_mask],
        sh_0=append_params.sh_0[..., keep_mask],
        sh_rest=append_params.sh_rest[..., keep_mask],
        opacity=append_params.opacity[..., keep_mask],
    )


def _prune_tensor(tensor: torch.Tensor, keep_mask: torch.Tensor, bcluster: bool) -> torch.Tensor:
    if bcluster:
        chunk_size = tensor.shape[-1]
        unclustered_tensor, = cluster.uncluster(tensor)
        pruned_tensor = unclustered_tensor[..., keep_mask]
        clustered_tensor, = cluster.cluster_points(chunk_size, pruned_tensor)
        return clustered_tensor.contiguous()
    return tensor[..., keep_mask].contiguous()


def _append_tensor(tensor: torch.Tensor, append_tensor: torch.Tensor, bcluster: bool) -> torch.Tensor:
    if bcluster:
        chunk_size = tensor.shape[-1]
        unclustered_tensor, = cluster.uncluster(tensor)
        appended_tensor = torch.cat((unclustered_tensor, append_tensor), dim=-1)
        clustered_tensor, = cluster.cluster_points(chunk_size, appended_tensor)
        return clustered_tensor.contiguous()
    return torch.cat((tensor, append_tensor), dim=-1).contiguous()


def _replace_tensor(tensor: torch.Tensor, new_tensor: torch.Tensor, bcluster: bool) -> torch.Tensor:
    if bcluster:
        chunk_size = tensor.shape[-1]
        clustered_tensor, = cluster.cluster_points(chunk_size, new_tensor)
        return clustered_tensor.contiguous()
    return new_tensor.contiguous()


def _remap_state(
    stored_state: dict,
    old_param: nn.Parameter,
    new_tensor: torch.Tensor,
    keep_mask: torch.Tensor | None,
    append_tensor: torch.Tensor | None,
    bcluster: bool
) -> dict:
    new_state = {}
    for key, value in stored_state.items():
        if not isinstance(value, torch.Tensor) or value.shape != old_param.shape:
            new_state[key] = value
            continue

        remapped_value = value
        if keep_mask is not None:
            remapped_value = _prune_tensor(remapped_value, keep_mask, bcluster)
        if append_tensor is not None:
            zeros = torch.zeros_like(append_tensor, dtype=remapped_value.dtype, device=remapped_value.device)
            remapped_value = _append_tensor(remapped_value, zeros, bcluster)
        if remapped_value.shape != new_tensor.shape:
            remapped_value = torch.zeros_like(new_tensor, dtype=remapped_value.dtype, device=remapped_value.device)
        new_state[key] = remapped_value
    return new_state


def _replace_state(stored_state: dict, old_param: nn.Parameter, new_tensor: torch.Tensor) -> dict:
    new_state = {}
    for key, value in stored_state.items():
        if not isinstance(value, torch.Tensor) or value.shape != old_param.shape:
            new_state[key] = value
            continue
        new_state[key] = torch.zeros_like(new_tensor, dtype=value.dtype, device=value.device)
    return new_state
