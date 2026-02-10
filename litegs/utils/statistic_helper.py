import torch
import typing
from typing import Optional, Callable
from ..scene import cluster
from ..utils.wrapper import litegs_fused

class MeanStdData:
    def __init__(self,data_shape:list,cluster_shape:list,device):
        self.sum=torch.zeros((*data_shape,*cluster_shape),device=device)
        self.square_sum=torch.zeros((*data_shape,*cluster_shape),device=device)
        self.count=torch.zeros(cluster_shape,device=device,dtype=torch.int32)
        return

class StatisticsHelper:
    def __init__(self,chunk_num:int,chunk_size:int):
        self.cached_tiles_blend_count:dict[str,torch.Tensor]={}
        self.cached_complex_tile:dict[str,torch.Tensor]={}
        self.cached_sorted_tile_list:dict[str,torch.Tensor]={}
        self.cur_sample:Optional[str]=None
    
        self.reset(chunk_num,chunk_size,lambda epoch:False)
        return
    
    
    def reset(self,chunk_num:int,chunk_size:int,statistics_check_handle:Callable[[int],bool]):
        self.bStart=False
        if statistics_check_handle is not None:
            self.is_statistics_enabled=statistics_check_handle
        self.chunk_num=chunk_num
        self.chunk_size=chunk_size
        self.mean_and_std:dict[str,MeanStdData]={}
        self.max_and_min:dict[str,list[torch.Tensor]]={}

        self.visible_count=torch.zeros(chunk_num,chunk_size,dtype=torch.int32,device='cuda')
        self.compact_mask:Optional[torch.Tensor]=None
        self.valid_length:Optional[torch.Tensor]=None

        self.handle_list:list[tuple[torch.Tensor,Callable[[StatisticsHelper, torch.Tensor],None]]]=[]
        return
    
    def try_start(self,epoch:int) -> "StatisticGuard":
        if self.is_statistics_enabled(epoch):
            return StatisticGuard(self)
        return StatisticGuard(None)
    
    def register_tensor_grad_callback(self,tensor:torch.Tensor,statistics_update_func:Callable[["StatisticsHelper", torch.Tensor],None]):
        tensor.retain_grad()
        self.handle_list.append((tensor,statistics_update_func))
        return
    
    @torch.no_grad()
    def backward_callback(self):
        for (tensor,statistics_update_func) in self.handle_list:
            if tensor.grad is not None:
                statistics_update_func(self,tensor.grad)
            else:
                assert(False)
        self.handle_list=[]
        return
    
    @torch.no_grad()
    def set_compact_mask(self,compact_mask:torch.Tensor,valid_length:Optional[torch.Tensor]=None):
        self.compact_mask=compact_mask
        self.valid_length=valid_length
        return
    
    @torch.no_grad()
    def update_tile_blend_count(self,piexel_blend_count:torch.Tensor,tilesize_h:int,tilesize_w:int):
        N,_,H,W=piexel_blend_count.shape
        assert(N==1,"expecting batch size of 1 for blend count")
        tiles_num_h=int((H+tilesize_h-1)/tilesize_h)
        tiles_num_w=int((W+tilesize_w-1)/tilesize_w)
        piexel_blend_count=piexel_blend_count.detach().reshape(N,tiles_num_h,tilesize_h,tiles_num_w,tilesize_w).permute(1,3,0,2,4).reshape(tiles_num_h*tiles_num_w,-1)
        tiles_blend_count=piexel_blend_count.max(dim=1).values
        if self.cur_sample is not None:
            self.cached_tiles_blend_count[self.cur_sample]=tiles_blend_count
            self.cached_sorted_tile_list[self.cur_sample]=tiles_blend_count.sort(descending=True)[1].int()+1
            #self.cached_complex_tile[self.cur_sample]=(self.cached_tiles_blend_count[self.cur_sample]>1024).nonzero()[:,0]+1
        return
    
    @torch.no_grad()
    def update_visible_count(self,visible_mask:torch.Tensor):
        if self.compact_mask is None:
            self.visible_count+=visible_mask.sum(0)
        else:
            if self.valid_length is None:
                self.visible_count[self.compact_mask]+=visible_mask.sum(0).reshape(-1,self.chunk_size)
            else:
                #gpu driven pipeline: the tail of visible_mask is dirty, so we must ignore it!
                visible_count_ref=self.visible_count.view(1,-1,self.chunk_size)
                compacted_visible_mask=visible_mask.sum(0,dtype=torch.int32).reshape(1,-1,self.chunk_size)
                litegs_fused.gpu_driven_pipeline_sparse_op(visible_count_ref,compacted_visible_mask,self.compact_mask,self.valid_length,"add")
        return
    

    @torch.no_grad()
    def update_mean_std(self,key:str,tensor_sum:torch.Tensor,square_sum:torch.Tensor,count:torch.Tensor,bCompacted:Optional[bool]=None):
        if bCompacted is None:
            bCompacted=(self.compact_mask is not None)
        if bCompacted:
            assert(self.compact_mask is not None)
            tensor_sum=tensor_sum.reshape(*tensor_sum.shape[:-1],-1,self.chunk_size)
            square_sum=square_sum.reshape(*square_sum.shape[:-1],-1,self.chunk_size)
            if count.__class__==torch.Tensor:
                count=count.reshape(-1,self.chunk_size)
        else:
            if count.__class__==torch.Tensor:
                count=count.squeeze()

        
        data=self.mean_and_std.get(key,None)
        if data is None:
            if bCompacted:
                data_shape=tensor_sum.shape[:-2]
                cluster_shape=(self.chunk_num,self.chunk_size)
            else:
                data_shape=tensor_sum.shape[:-1]
                cluster_shape=(tensor_sum.shape[-1],)
            data=MeanStdData(list(data_shape),list(cluster_shape),tensor_sum.device)
            self.mean_and_std[key]=data
        
        #update dict
        if bCompacted:
            if self.valid_length is None:
                data.sum[...,self.compact_mask,:]+=tensor_sum
                data.square_sum[...,self.compact_mask,:]+=square_sum
                data.count[self.compact_mask,:]+=count
            else:
                #gpu driven pipeline: the tail of visible_mask is dirty, so we must ignore it!
                chunks_num=data.sum.shape[-2]
                allocated_chunks_num=tensor_sum.shape[-2]
                litegs_fused.gpu_driven_pipeline_sparse_op(
                    data.sum.view(-1,chunks_num,self.chunk_size),
                    tensor_sum.view(-1,allocated_chunks_num,self.chunk_size),
                    self.compact_mask,self.valid_length,
                    "add"
                )
                litegs_fused.gpu_driven_pipeline_sparse_op(
                    data.square_sum.view(-1,chunks_num,self.chunk_size),
                    square_sum.view(-1,allocated_chunks_num,self.chunk_size),
                    self.compact_mask,self.valid_length,
                    "add"
                )
                litegs_fused.gpu_driven_pipeline_sparse_op(
                    data.count.view(-1,chunks_num,self.chunk_size),
                    count.view(-1,allocated_chunks_num,self.chunk_size),
                    self.compact_mask,self.valid_length,
                    "add"
                )

        else:
            data.sum+=tensor_sum
            data.square_sum+=square_sum
            data.count+=count

        return

    @torch.no_grad()
    def update_max_min(self,key:str,tensor:torch.Tensor):
        #update dict
        tensor_max=tensor.max(0)[0]
        tensor_min=tensor.min(0)[0]
        data=self.max_and_min.get(key,None)
        if data is not None:
            data[0]=torch.max(tensor_max,data[0])
            data[1]=torch.min(tensor_min,data[1])
        else:
            data=[tensor_max,tensor_min]
            self.max_and_min[key]=data
        return


    @torch.no_grad()
    def update_max_min_compact(self,key:str,compact_tensor:torch.Tensor):
        assert(self.compact_mask is not None)

        tensor_max=compact_tensor.max(0).values
        tensor_max=tensor_max.reshape(*tensor_max.shape[:-1],-1,self.chunk_size)
        tensor_min=compact_tensor.min(0).values
        tensor_min=tensor_min.reshape(*tensor_min.shape[:-1],-1,self.chunk_size)

        data=self.max_and_min.get(key,None)
        if data is not None:
            data[0][...,self.compact_mask,:]=torch.max(tensor_max,data[0][...,self.compact_mask,: ])
            data[1][...,self.compact_mask,: ]=torch.min(tensor_min,data[1][...,self.compact_mask,: ])
        else:
            max_uncompact=torch.ones((*tensor_max.shape[:-2],self.chunk_num,self.chunk_size),device=compact_tensor.device)*(-torch.inf)
            min_uncompact=torch.ones((*tensor_min.shape[:-2],self.chunk_num,self.chunk_size),device=compact_tensor.device)*torch.inf
            data=[max_uncompact,min_uncompact]
            data[0][...,self.compact_mask,:]=tensor_max
            data[1][...,self.compact_mask,:]=tensor_min
            self.max_and_min[key]=data
        return 
    

    @torch.no_grad()
    def get_max(self,key:str) -> torch.Tensor | None:
        data = self.max_and_min.get(key,None)
        max_val=None
        if data is not None:
            max_val=data[0]
            max_val,=cluster.uncluster(max_val)
        return max_val
    
    @torch.no_grad()
    def get_min(self,key:str) -> torch.Tensor | None:
        data = self.max_and_min.get(key,None)
        min_val=None
        if data is not None:
            min_val=data[1]
            min_val,=cluster.uncluster(min_val)
        return min_val

    @torch.no_grad()
    def get_mean(self,key:str) -> tuple[torch.Tensor, torch.Tensor] | None:
        data = self.mean_and_std.get(key,None)
        mean_val=None
        if data is not None:
            mean_val=(data.sum/(data.count+1e-9))
            mean_val,=cluster.uncluster(mean_val)
            return mean_val,data.count.reshape(-1)
        return None
    
    @torch.no_grad()
    def get_var(self,key:str) -> tuple[torch.Tensor, torch.Tensor] | None:

        def calc_var(sum:torch.Tensor,square_sum:torch.Tensor,count:torch.Tensor):
            grad_mean=sum/(count+1)
            grad_square_mean=square_sum/(count+1)
            grad_var=grad_square_mean-grad_mean**2
            return grad_var.clamp_min(0)
        
        data = self.mean_and_std.get(key,None)
        std_tensor=None
        if data is not None:
            std_tensor=calc_var(data.sum,data.square_sum,data.count)
            if self.compact_mask is not None:
                std_tensor,=cluster.uncluster(std_tensor)
            return std_tensor,data.count.reshape(-1)
        return None
    
    def get_global_culling(self) -> torch.Tensor:
        culled=(self.visible_count==0)
        culled,=cluster.uncluster(culled)
        return culled
    
class StatisticGuard:
    def __init__(self,inst:Optional[StatisticsHelper]):
        self.stats_obj=inst
        return
    
    def __enter__(self):
        if self.stats_obj is not None:
            self.stats_obj.bStart=True
        return

    def __exit__(self, *args):
        if self.stats_obj is not None:
            self.stats_obj.bStart=False
        return

StatisticsHelperInst=StatisticsHelper(0,0)