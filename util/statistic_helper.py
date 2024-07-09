import torch
import typing
class StatisticsHelper:
    def __init__(self,chunk_num,chunk_size):
        self.reset(chunk_num,chunk_size)
        return
    
    def start(self):
        self.bStart=True

    def pause(self):
        self.bStart=False
    
    def reset(self,chunk_num,chunk_size):
        self.bStart=False
        self.chunk_num=chunk_num
        self.chunk_size=chunk_size
        self.mean_and_std:dict[str,torch.Tensor]={}
        self.max_and_min:dict[str,torch.Tensor]={}

        self.visible_count=torch.zeros(chunk_num*chunk_size,dtype=torch.int32,device='cuda')
        self.compact_mask:torch.Tensor=None

        self.handle_list:list[tuple[str,torch.Tensor,typing.Callable[[torch.Tensor],torch.Tensor],typing.Callable]]=[]
        return
    
    def register_tensor_grad_callback(self,key:str,tensor:torch.Tensor,
                                      statistics_update_func:typing.Callable,
                                      grad_update_func:typing.Callable[[torch.Tensor],torch.Tensor]=None):
        tensor.retain_grad()
        self.handle_list.append((key,tensor,grad_update_func,statistics_update_func))
        return
    
    @torch.no_grad
    def backward_callback(self):
        for (key,tensor,grad_update_func,statistics_update_func) in self.handle_list:
            if tensor.grad is not None:
                if grad_update_func is not None:
                    grad=grad_update_func(tensor.grad)
                else:
                    grad=tensor.grad
                statistics_update_func(self,key,grad)
            else:
                assert(False)
        self.handle_list=[]
        return
    
    @torch.no_grad
    def set_compact_mask(self,compact_mask:torch.Tensor):
        self.compact_mask=compact_mask
        return
    
    @torch.no_grad
    def update_visible_count(self,compacted_visible_mask:torch.Tensor):
        assert(self.compact_mask is not None)
        self.visible_count[self.compact_mask]+=compacted_visible_mask.sum(0)
        return
    

    @torch.no_grad
    def update_mean_std(self,key:str,tensor:torch.Tensor):
        #update dict
        tensor_sum=tensor.sum(0)
        square_sum=(tensor**2).sum(0)
        data=self.mean_and_std.get(key,None)
        if data is not None:
            data[0]+=tensor_sum
            data[1]+=square_sum
        else:
            data=[tensor_sum,square_sum]
            self.mean_and_std[key]=data
        return

    @torch.no_grad
    def update_mean_std_compact(self,key:str,compact_tensor:torch.Tensor):
        assert(self.compact_mask is not None)

        tensor_sum=compact_tensor.sum(0)
        square_sum=(compact_tensor**2).sum(0)

        data=self.mean_and_std.get(key,None)
        if data is not None:
            data[0][self.compact_mask]+=tensor_sum
            data[1][self.compact_mask]+=square_sum
        else:
            tensor_sum_uncompact=torch.zeros((self.chunk_num*self.chunk_size,*tensor_sum.shape[1:]),device=compact_tensor.device)
            square_sum_uncompact=torch.zeros((self.chunk_num*self.chunk_size,*square_sum.shape[1:]),device=compact_tensor.device)
            tensor_sum_uncompact[self.compact_mask]+=tensor_sum
            square_sum_uncompact[self.compact_mask]+=square_sum
            data=[tensor_sum_uncompact,square_sum_uncompact]
            self.mean_and_std[key]=data
        return
    
    @torch.no_grad
    def update_max_min(self,key:str,tensor:torch.Tensor):
        #update dict
        tensor_max=tensor.max(0)[0]
        tensor_min=tensor.min(0)[0]
        data=self.max_and_min.get(key,None)
        if data is not None:
            data[0]=torch.max(tensor_max,data[0])
            data[1]=torch.min(tensor_min,data[1])
        else:
            data=(tensor_max,tensor_min)
            self.mean_and_std[key]=data
        return


    @torch.no_grad
    def update_max_min_compact(self,key:str,compact_tensor:torch.Tensor):
        assert(self.compact_mask is not None)

        tensor_max=compact_tensor.max(0)[0]
        tensor_min=compact_tensor.min(0)[0]
        data=self.max_and_min.get(key,None)
        if data is not None:
            data[0][self.compact_mask ]=torch.max(tensor_max,data[0][self.compact_mask ])
            data[1][self.compact_mask ]=torch.min(tensor_min,data[1][self.compact_mask ])
        else:
            max_uncompact=torch.ones((self.chunk_num*self.chunk_size,*tensor_max.shape[1:]),device=compact_tensor.device)*(-torch.inf)
            min_uncompact=torch.ones((self.chunk_num*self.chunk_size,*tensor_min.shape[1:]),device=compact_tensor.device)*torch.inf
            data=[max_uncompact,min_uncompact]
            data[0][self.compact_mask ]=tensor_max
            data[1][self.compact_mask ]=tensor_min
            self.max_and_min[key]=data
        return 
    

    @torch.no_grad
    def get_max(self,key:str):
        data = self.max_and_min.get(key,None)
        max_val=None
        if data is not None:
            max_val=data[0]
        return max_val
    
    @torch.no_grad
    def get_min(self,key:str):
        data = self.max_and_min.get(key,None)
        min_val=None
        if data is not None:
            min_val=data[1]
        return min_val

    @torch.no_grad
    def get_mean(self,key:str):
        data = self.mean_and_std.get(key,None)
        mean_val=None
        if data is not None:
            mean_val=(data[0].transpose(0,-1)/(self.visible_count+1e-6)).transpose(0,-1)
        return mean_val
    
    @torch.no_grad
    def get_std(self,key:str):

        def calc_std(sum:torch.Tensor,square_sum:torch.Tensor,count:torch.Tensor):
            grad_mean=(sum.transpose(0,-1)/count).transpose(0,-1)
            grad_square_mean=(square_sum.transpose(0,-1)/count).transpose(0,-1)
            grad_std=grad_square_mean-grad_mean**2
            return grad_std
        
        data = self.mean_and_std.get(key,None)
        std_tensor=None
        if data is not None:
            std_tensor=calc_std(data[0],data[1],self.visible_count)
        return std_tensor

StatisticsHelperInst=StatisticsHelper(0,0)