from __future__ import annotations
import torch
import copy
import math

class ObjectBase:

    def __init__(self,obj_id:torch.Tensor,position:torch.Tensor=None,device='cuda'):
        self.registered_properties:list[str]=[]

        assert(obj_id is not None)
        self.obj_id=obj_id
        self.registered_properties.append('obj_id')

        if position is None:
            self.position=torch.zeros(3,device)
        else:
            self.position=position
        self.registered_properties.append('position')

        return
    
    @property
    def device(self):
        return self.position.device

    def cpu(self):
        self.position=self.position.cpu()
        return

    def cuda(self):
        self.position=self.position.cuda()
        return

    def _get_extend(self):
        return torch.zeros(3,device=self.position.device)
    
    def get_AABB(self):
        origin=self.position
        extend=self._get_extend()
        return origin,extend
    
    def get_properties_dict(self)->dict[str,torch.Tensor]:
        properties_dict:dict[str,torch.Tensor]={}
        for property_name in self.registered_properties:
            properties_dict[property_name]=self.__getattribute__(property_name)
        
        return properties_dict
    
class ObjectBatchBase(ObjectBase):
    def __init__(self,obj_id:torch.Tensor,position:torch.Tensor,device='cuda'):
        assert(position is not None)
        assert(obj_id is not None)
        super(ObjectBatchBase,self).__init__(obj_id,position,device)
        return
    
    def get_objects_num(self)->int:
        return self.position.shape[0]
    
    @torch.no_grad()
    def insert_obj(self,obj:ObjectBase):
        assert(self.registered_properties==obj.registered_properties)
        properties_dict=obj.get_properties_dict()
        for (property_name,tensor) in properties_dict.items():
            tensor=tensor.unsqueeze(0)
            cur_batch_property=self.__getattribute__(property_name)
            self.__setattr__(property_name,torch.cat(cur_batch_property,tensor))
        return

    @torch.no_grad()
    def merge_batch(self,obj_batch:ObjectBatchBase):
        assert(self.registered_properties==obj_batch.registered_properties)
        properties_dict=obj_batch.get_properties_dict()
        for (property_name,tensor) in properties_dict.items():
            cur_batch_property=self.__getattribute__(property_name)
            self.__setattr__(property_name,torch.cat(cur_batch_property,tensor))
        return
    
    @torch.no_grad()
    def filter(self,mask):
        for property_name in self.registered_properties:
            cur_batch_property=self.__getattribute__(property_name)
            self.__setattr__(property_name,cur_batch_property[mask])
        return

    @torch.no_grad()
    def devide(self,mask)->tuple[ObjectBatchBase,ObjectBatchBase]:
        groupA=copy.deepcopy(self)
        groupA.filter(mask)
        groupB=copy.deepcopy(self)
        groupB.filter(~mask)
        return (groupA,groupB)
    
    @torch.no_grad()
    def get_AABB(self):
        positions=self.position
        extends=self._get_extend()
        max_xyz=(positions+extends).max(dim=0).values
        min_xyz=(positions-extends).min(dim=0).values
        origin=(max_xyz+min_xyz)/2
        extend=(max_xyz-min_xyz)/2
        return origin,extend
    

    
class GSpointBatch(ObjectBatchBase):
    def __init__(self,point_id:torch.Tensor,position:torch.Tensor,cov:torch.Tensor):
        super(GSpointBatch,self).__init__(point_id,position)
        self.cov=cov
        self.registered_properties.append('cov')
        return
    
    @torch.no_grad()
    def _get_extend(self):
        #todo
        eigen_val,eigen_vec=torch.linalg.eigh(self.cov)
        eigen_val=eigen_val.abs()
        coefficient=2*math.log(255)
        extend=((coefficient*eigen_val.unsqueeze(-1)).sqrt()*eigen_vec).abs().sum(dim=-2)
        return extend