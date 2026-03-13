import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

from .. import arguments
from ..utils.wrapper import sparse_adam_update,litegs_fused
from ..utils.CompactedTensor import CompactedTensor

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps, bCluster):
        self.bCluster=bCluster
        self.visible_chunk=None
        self.visible_chunks_num=None
        self.visible_primitive=None
        self.visible_primitives_num=None
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def update_sparse_visibility(self,visible_chunk,visible_chunks_num,visible_primitive,visible_primitives_num):
        if self.visible_chunk is not None or self.visible_primitive is not None:
            assert(False,"SparseGaussianAdam do not support multipass backward!")
        self.visible_chunk=visible_chunk
        self.visible_chunks_num=visible_chunks_num
        self.visible_primitive=visible_primitive
        self.visible_primitives_num=visible_primitives_num
        return

    def zero_grad(self, set_to_none: bool = True) -> None:
        super(SparseGaussianAdam,self).zero_grad(set_to_none)
        self.visible_chunk=None
        self.visible_chunks_num=None
        self.visible_primitive=None
        self.visible_primitives_num=None
        return
    
    @torch.no_grad()
    def step(self):
        
        if self.visible_chunk is None and self.visible_primitive is None:
            assert(False,"Call update_sparse_visibility before step!")

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

            if self.bCluster:
                stored_state = self.state.get(param, None)
                exp_avg = stored_state["exp_avg"].view(-1,param.shape[-2],param.shape[-1])
                exp_avg_sq = stored_state["exp_avg_sq"].view(-1,param.shape[-2],param.shape[-1])
                param_view=param.data.view(-1,param.shape[-2],param.shape[-1])
                assert(isinstance(param.grad,CompactedTensor),"expecting CompactedTensor grad")
                sparse_adam_update(
                    param_view, param.grad.compacted_values, 
                    exp_avg, exp_avg_sq, 
                    self.visible_chunk,self.visible_chunks_num, 
                    lr, 0.9, 0.999, eps
                )
            else:
                stored_state = self.state.get(param, None)
                exp_avg = stored_state["exp_avg"]
                exp_avg_sq = stored_state["exp_avg_sq"]
                N=param.shape[-1]
                sparse_adam_update(
                    param.view(-1,N), param.grad.view(-1,N), exp_avg.view(-1,N), exp_avg_sq.view(-1,N), 
                    self.visible_primitive,self.visible_primitives_num, 
                    lr, 0.9, 0.999, eps
                )
        return
    


class Scheduler(_LRScheduler):
    def __init__(self, optimizer:torch.optim.Adam,lr_init, lr_final,max_epochs=10000, last_epoch=-1):
        self.max_epochs=max_epochs
        self.lr_init=lr_init
        self.lr_final=lr_final
        super(Scheduler, self).__init__(optimizer, last_epoch)
        return
    
    def __helper(self):
        if self.last_epoch < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            # Disable this parameter
            return 0.0
        delay_rate = 1.0
        t = np.clip(self.last_epoch / self.max_epochs, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

    def get_lr(self):
        lr_list=[]
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                lr_list.append(self.__helper())
            else:
                lr_list.append(group['initial_lr'])

        return lr_list


def get_optimizer(xyz:torch.nn.Parameter,scale:torch.nn.Parameter,rot:torch.nn.Parameter,
                  sh_0:torch.nn.Parameter,sh_rest:torch.nn.Parameter,opacity:torch.nn.Parameter,
                  spatial_lr_scale:float,
                  opt_setting:arguments.OptimizationParams,bCluster:bool):
    
    params_geo = [
        {'params': [xyz], 'lr': opt_setting.position_lr_init * spatial_lr_scale, "name": "xyz"},
        {'params': [opacity], 'lr': opt_setting.opacity_lr, "name": "opacity"},
        {'params': [scale], 'lr': opt_setting.scaling_lr, "name": "scale"},
        {'params': [rot], 'lr': opt_setting.rotation_lr, "name": "rot"}
    ]
    params_sh = [
        {'params': [sh_0], 'lr': opt_setting.feature_lr, "name": "sh_0"},
        {'params': [sh_rest], 'lr': opt_setting.feature_lr / 10.0, "name": "sh_rest"},
    ]
    if opt_setting.sparse_grad:
        optimizer = SparseGaussianAdam(params_geo, lr=0, eps=1e-15,bCluster=bCluster)
        sh_optimizer = SparseGaussianAdam(params_sh, lr=0, eps=1e-15,bCluster=bCluster)
    else:
        optimizer = torch.optim.Adam(params_geo, lr=0, eps=1e-15)
        sh_optimizer = torch.optim.Adam(params_sh, lr=0, eps=1e-15)
    if bCluster:
        sh_optimizer=ShFusedAdam(sh_0,sh_rest,opt_setting.feature_lr,opt_setting.feature_lr/10,eps=1e-15)
    scheduler = Scheduler(optimizer,opt_setting.position_lr_init*spatial_lr_scale,
              opt_setting.position_lr_final*spatial_lr_scale,
              max_epochs=opt_setting.position_lr_max_steps)
    
    return optimizer,scheduler,sh_optimizer


class ShFusedAdam(torch.optim.Adam):
    def __init__(self, sh_0, sh_rest, lr_0, lr_rest, eps):
        self._color=None
        params = [
            {'params': [sh_0], 'lr': lr_0, "name": "sh_0"},
            {'params': [sh_rest], 'lr':lr_rest, "name": "sh_rest"},
        ]
        super().__init__(params=params, lr=0, eps=eps)
    
    def forward(
        self,
        sh_degree,
        visible_chunkid, visible_chunk_num,
        view_matrix,position
    ):
        if self._color is not None:
            assert(False,"ShFusedAdam do not support multipass backward!")
        sh_0=self.param_groups[0]['params'][0]
        sh_rest=self.param_groups[1]['params'][0]

        color:torch.Tensor = litegs_fused.compact_sh_forward(
            sh_degree,
            visible_chunkid, visible_chunk_num,
            view_matrix,
            position, sh_0, sh_rest
        )

        self._color=color.detach().requires_grad_()
        self._color.retain_grad()
        self._sh_degree=sh_degree
        self._visible_chunkid=visible_chunkid
        self._visible_chunk_num=visible_chunk_num
        self._view_matrix=view_matrix
        self._position=position


        return self._color

    def zero_grad(self, set_to_none: bool = True) -> None:
        super(ShFusedAdam,self).zero_grad(set_to_none)
        self._color=None
        self._sh_degree=None
        self._visible_chunkid=None
        self._visible_chunk_num=None
        self._position=None
        self._view_matrix=None
        return
    
    @torch.no_grad()
    def step(self):
        
        if self._color is None or self._color.grad is None:
            return
        
        sh_0=self.param_groups[0]['params'][0]
        sh_rest=self.param_groups[1]['params'][0]
        
        state = self.state[sh_0]
        if len(state) == 0:
            state['step'] = torch.tensor(0.0, dtype=torch.float32)
            state['exp_avg'] = torch.zeros_like(sh_0, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(sh_0, memory_format=torch.preserve_format)
        state = self.state[sh_rest]
        if len(state) == 0:
            state['step'] = torch.tensor(0.0, dtype=torch.float32)
            state['exp_avg'] = torch.zeros_like(sh_rest, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(sh_rest, memory_format=torch.preserve_format)

        litegs_fused.compact_sh_backward_adam(
            self._sh_degree,
            self._visible_chunkid,self._visible_chunk_num,self._view_matrix,self._position,
            sh_0,sh_rest,
            self._color.grad,
            self.state[sh_0]['exp_avg'],self.state[sh_0]['exp_avg_sq'],
            self.state[sh_rest]['exp_avg'],self.state[sh_rest]['exp_avg_sq'],
            self.param_groups[0]['lr'],self.param_groups[1]['lr'],0.9,0.99,self.defaults['eps']
        )

        
        return