import torch
import typing
class StatisticsHelper:
    def __init__(self,gaussian_num):
        self.reset(gaussian_num)
        return
    
    def start(self):
        self.bStart=True

    def pause(self):
        self.bStart=False
    
    def reset(self,gaussian_num):
        self.bStart=False
        self.gaussian_num=gaussian_num
        self.mean_and_std:dict[str,torch.Tensor]={}
        self.max_and_min:dict[str,torch.Tensor]={}
        self.visible_count=torch.zeros(gaussian_num,dtype=torch.int32,device='cuda')
        self.cur_batch_visible_mask:torch.Tensor=None
        self.cur_batch_visible_pts:torch.Tensor=None
        self.cur_batch_visible_num:torch.Tensor=None
        self.batch_n=0

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
    def set_cur_batch_visibility(self,cur_visible_points:torch.Tensor,cur_visible_num:torch.Tensor):
        assert(cur_visible_points.max()<self.gaussian_num)
        assert(cur_visible_points.shape[0]==cur_visible_num.shape[0])
        #set batch
        self.batch_n=cur_visible_points.shape[0]
        #update visible_count
        for i in range(self.batch_n):
            points_num=cur_visible_num[i]
            self.visible_count[cur_visible_points[i,:points_num]]+=1
        self.cur_batch_visible_pts=cur_visible_points
        self.cur_batch_visible_num=cur_visible_num
        return
    
    @torch.no_grad
    def update_mean_std(self,key:str,tensor:torch.Tensor):
        #assert(tensor.shape[0]==self.batch_n)
        assert(tensor.shape[1]==self.gaussian_num)

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
    def _update_mean_std_compact_internel(self,key:str,compact_tensor:torch.Tensor,pts:torch.Tensor):
        data=self.mean_and_std.get(key,None)
        if data is not None:
            data[0][pts]+=compact_tensor
            data[1][pts]+=compact_tensor**2
        else:
            shape=list(compact_tensor.shape)
            shape[0]=self.gaussian_num
            data0=torch.zeros(shape,dtype=compact_tensor.dtype,device=compact_tensor.device)
            data1=torch.zeros(shape,dtype=compact_tensor.dtype,device=compact_tensor.device)
            data=[data0,data1]
            data[0][pts]+=compact_tensor
            data[1][pts]+=compact_tensor**2
            self.mean_and_std[key]=data
        return

    @torch.no_grad
    def update_mean_std_compact(self,key:str,compact_tensor:torch.Tensor):
        assert(self.cur_batch_visible_pts is not None)
        assert(self.cur_batch_visible_num is not None)
        assert(compact_tensor.shape[0]==self.batch_n)

        for i in range(self.batch_n):
            points_num=self.cur_batch_visible_num[i]
            points_index=self.cur_batch_visible_pts[i,:points_num]
            self._update_mean_std_compact_internel(key,compact_tensor[i,:points_num],points_index)
        return
    
    @torch.no_grad
    def update_max_min(self,key:str,tensor:torch.Tensor):
        assert(self.cur_batch_visible_pts is not None)
        assert(self.cur_batch_visible_num is not None)
        assert(tensor.shape[0]==self.batch_n)
        assert(tensor.shape[1]==self.gaussian_num)
        
        #filter invisible
        for i in range(self.batch_n):
            pts_num=self.cur_batch_visible_num[i]
            tensor[self.cur_batch_visible_pts[i,:pts_num]]=0

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
    def _update_max_min_compact_internel(self,key:str,compact_tensor:torch.Tensor,pts:torch.Tensor):
        data=self.max_and_min.get(key,None)
        if data is not None:
            data[0][pts]=torch.max(compact_tensor,data[0][pts])
            data[1][pts]=torch.min(compact_tensor,data[1][pts])
        else:
            shape=list(compact_tensor.shape)
            shape[0]=self.gaussian_num
            data0=torch.zeros(shape,dtype=compact_tensor.dtype,device=compact_tensor.device)
            data1=torch.zeros(shape,dtype=compact_tensor.dtype,device=compact_tensor.device)
            data=[data0,data1]
            data[0][pts]=torch.max(compact_tensor,data[0][pts])
            data[1][pts]=torch.min(compact_tensor,data[1][pts])
            self.max_and_min[key]=data
        return

    @torch.no_grad
    def update_max_min_compact(self,key:str,compact_tensor:torch.Tensor):
        assert(self.cur_batch_visible_pts is not None)
        assert(self.cur_batch_visible_num is not None)
        assert(compact_tensor.shape[0]==self.batch_n)

        for i in range(self.batch_n):
            points_num=self.cur_batch_visible_num[i]
            points_index=self.cur_batch_visible_pts[i,:points_num]
            self._update_max_min_compact_internel(key,compact_tensor[i,:points_num],points_index)
        return 
    
    @torch.no_grad
    def update_invisible_compact(self,compacted_invisible_mask:torch.Tensor):
        assert(self.cur_batch_visible_pts is not None)
        assert(self.cur_batch_visible_num is not None)
        assert(compacted_invisible_mask.shape[0]==self.batch_n)
        for i in range(self.batch_n):
            points_num=self.cur_batch_visible_num[i]
            points_index=self.cur_batch_visible_pts[i,:points_num]
            self.visible_count[points_index]-=1*compacted_invisible_mask[i,:points_num]
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

StatisticsHelperInst=StatisticsHelper(0)