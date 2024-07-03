from __future__ import annotations
import torch
from util.BVH.Object import ObjectBatchBase
import math
import typing

class BVHNode:
    def __init__(self,ObjectBatchBase:ObjectBatchBase,terminate_obj_num:int,callback:typing.Callable[[BVHNode],None]=None):
        origin,extend=ObjectBatchBase.get_AABB()
        self.origin=origin
        self.extend=extend

        main_dim=self.extend.max(dim=0).indices

        N=ObjectBatchBase.get_objects_num()
        self.child=None
        self.objs=None
        if N>terminate_obj_num:
            sorted_value,sorted_index=ObjectBatchBase.position[:,main_dim].sort()

            mid_value=(sorted_value.max()+sorted_value.min())/2
            for group_A_num in range(terminate_obj_num,N,terminate_obj_num):
                if sorted_value[group_A_num]>mid_value:
                    break

            mask=torch.zeros(N,device=ObjectBatchBase.device).bool()
            mask[sorted_index[:group_A_num]]=True
            ObjectBatchBase0,ObjectBatchBase1=ObjectBatchBase.devide(mask)
            self.child=[BVHNode(ObjectBatchBase0,terminate_obj_num,callback),BVHNode(ObjectBatchBase1,terminate_obj_num,callback)]
        else:
            self.objs=ObjectBatchBase.obj_id
        if callback is not None:
            callback(self)
        return
    
    def is_leaf(self)->bool:

        return self.child is None

class BVH:
    def __init__(self,object_list:list[ObjectBatchBase]):
        self.object_list:list[ObjectBatchBase]=object_list
        self.root:BVHNode=None
        self.leaf_nodes:list[BVHNode]=[]
        return
    
    def insert(self,object_list:list[ObjectBatchBase]):
        self.object_list+=object_list
        return
    
    def __node_finish_callback(self,node:BVHNode):
        if node.is_leaf():
            self.leaf_nodes.append(node)
        return
    
    def build(self,terminate_obj_num=32):
        assert(self.object_list is not None and len(self.object_list)>0)

        #clear
        self.root=None
        self.leaf_nodes=[]

        #merge match
        entire_batch=self.object_list[0]
        for i in range(1,len(self.object_list)):
            entire_batch.merge_batch(self.object_list[i])
            
        #build bvh
        self.root=BVHNode(entire_batch,terminate_obj_num,self.__node_finish_callback)
       
        return