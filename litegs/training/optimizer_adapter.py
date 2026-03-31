import torch

from .densify import DensifyEdits
from ..scene import cluster


def sync_state_after_densify(
    optimizer_list: list[torch.optim.Optimizer | None],
    edits: DensifyEdits,
    exchange_dict: dict[torch.Tensor,torch.Tensor],
    bcluster: bool
) -> None:
    if not edits.changed:
        return

    for optimizer in optimizer_list:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            param = group["params"][0]
            stored_state = optimizer.state.get(param, None)
            if param in exchange_dict.keys():
                
                if stored_state is None:
                    continue

                name = group["name"]
                append_tensor = None if edits.append_params is None else getattr(edits.append_params, name)
                target_tensor = param.detach()

                for key, value in stored_state.items():
                    if not isinstance(value, torch.Tensor) or key=='step':
                        continue

                    remapped_value = value
                    if edits.prune_index is not None:
                        remapped_value = _prune_state_tensor(remapped_value, edits.prune_index, bcluster)
                    if append_tensor is not None:
                        zeros = torch.zeros_like(append_tensor, dtype=remapped_value.dtype, device=remapped_value.device)
                        remapped_value = _append_state_tensor(remapped_value, zeros, bcluster)
                    value.data = remapped_value
                
                group["params"][0] = exchange_dict[param]
                optimizer.state[exchange_dict[param]] = stored_state
                if exchange_dict[param] is not param:
                    optimizer.state.pop(param, None)

    if edits.clear_optimizer_state:
        for optimizer in optimizer_list:
            if optimizer is not None:
                optimizer.state.clear()

    return


def _prune_state_tensor(state_tensor: torch.Tensor, prune_index: torch.Tensor, bcluster: bool) -> torch.Tensor:

    if bcluster:
        chunk_size = state_tensor.shape[-1]
        unclustered_tensor, = cluster.uncluster(state_tensor)
        keep_mask=torch.ones(unclustered_tensor.shape[-1],dtype=torch.bool,device=state_tensor.device)
        keep_mask[prune_index]=False
        pruned_tensor = unclustered_tensor[..., keep_mask]
        clustered_tensor, = cluster.cluster_points(chunk_size, pruned_tensor)
        return clustered_tensor.contiguous()
    
    keep_mask=torch.ones(state_tensor.shape[-1],dtype=torch.bool,device=state_tensor.device)
    keep_mask[prune_index]=False
    return state_tensor[..., keep_mask].contiguous()


def _append_state_tensor(state_tensor: torch.Tensor, append_tensor: torch.Tensor, bcluster: bool) -> torch.Tensor:
    if bcluster:
        chunk_size = state_tensor.shape[-1]
        unclustered_tensor, = cluster.uncluster(state_tensor)
        appended_tensor = torch.cat((unclustered_tensor, append_tensor), dim=-1)
        clustered_tensor, = cluster.cluster_points(chunk_size, appended_tensor)
        return clustered_tensor.contiguous()
    return torch.cat((state_tensor, append_tensor), dim=-1).contiguous()


def reorder_states_by_index(
    optimizer_list: list[torch.optim.Optimizer | None],
    indices: torch.Tensor,
    bcluster: bool
) -> None:
    for optimizer in optimizer_list:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            param = group["params"][0]
            stored_state = optimizer.state.get(param, None)
            if stored_state is None:
                continue
            for key, value in stored_state.items():
                if isinstance(value, torch.Tensor) and value.shape == param.shape:
                    value.data = _reorder_tensor(value.data, indices, bcluster)
    return


def _reorder_tensor(tensor: torch.Tensor, indices: torch.Tensor, bcluster: bool) -> torch.Tensor:
    if bcluster:
        chunk_size = tensor.shape[-1]
        unclustered_tensor, = cluster.uncluster(tensor)
        reordered_tensor = unclustered_tensor[..., indices]
        clustered_tensor, = cluster.cluster_points(chunk_size, reordered_tensor)
        return clustered_tensor.contiguous()
    return tensor[..., indices].contiguous()
