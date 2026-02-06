import torch

class CompactedTensor(torch.Tensor):
    """
    A container for sparse gradients
    """
    
    @staticmethod
    def __new__(cls, full_shape, chunk_ids:torch.Tensor, compacted_values:torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls, 
            full_shape, 
            dtype=compacted_values.dtype, 
            device=compacted_values.device,
            layout=torch.strided, # 伪装成 strided，兼容性更好
            requires_grad=False
        )

    def __init__(self, full_shape, chunk_ids:torch.Tensor, compacted_values:torch.Tensor):
        self.chunk_ids = chunk_ids
        self.compacted_values = compacted_values
        return

    def __repr__(self):
        return f"CompactedTensor(shape={self.shape}, compacted_shape={self.compacted_values.shape})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        if func.overloadpacket == torch.ops.aten.detach:
            old = args[0]
            new_obj = cls(old.shape, old.chunk_ids, old.compacted_values)
            return new_obj

        if func.overloadpacket == torch.ops.aten.clone:
            old = args[0]
            new_obj = cls(old.shape, old.chunk_ids.clone(), old.compacted_values.clone())
            return new_obj
            
        if func.overloadpacket in (torch.ops.aten.add_, torch.ops.aten.add):
            breakpoint()
            return args[0]

        breakpoint()
        return args[0]