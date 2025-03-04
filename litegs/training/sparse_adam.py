import torch
from gaussian_splatting.wrapper import sparse_adam_update

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visible_chunk):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"].view(-1,param.shape[-2],param.shape[-1])
            exp_avg_sq = stored_state["exp_avg_sq"].view(-1,param.shape[-2],param.shape[-1])
            param_view=param.data.view(-1,param.shape[-2],param.shape[-1])
            sparse_adam_update(param_view, param.grad._values().reshape(param_view.shape[0],visible_chunk.shape[0],param_view.shape[-1]), exp_avg, exp_avg_sq, visible_chunk, lr, 0.9, 0.999, eps)