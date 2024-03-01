import torch
class VisibleInfo:
    @torch.no_grad()
    def __init__(self,batchId:int,visible_points_for_views:torch.Tensor,visible_points_num:torch.Tensor,generation:int):
        self.batchId=batchId
        self.visible_points=visible_points_for_views
        self.visible_points_num=visible_points_num
        self.generation=generation
        return
    
class BinningInfo:
    @torch.no_grad()
    def __init__(self,batchId:int,tile_start_index:torch.Tensor,sorted_pointId:torch.Tensor,sorted_tileId:torch.Tensor,generation:int):
        self.batchId=batchId
        self.start_index=tile_start_index
        self.pointId=sorted_pointId
        self.tileId=sorted_tileId
        self.generation=generation
        return
    
class CachedData:
    def __init__(self,batchId:int,generation:int):
        self._batchId=batchId
        self._visible_info:VisibleInfo=None
        self._binning_info:BinningInfo=None
        self._generation=generation
        return
    
    @property
    def batchId(self):
        return self._batchId
    
    @property
    def generation(self):
        return self._generation
    
    @property
    def visible_info(self):
        if(self._visible_info.generation!=self._generation):
            print("Mismatch:generation")
            return None
        return self._visible_info
    
    @visible_info.setter
    def visible_info(self,input:VisibleInfo):

        if self._visible_info is not None:
            if input.generation<self._visible_info.generation:
                print("Warning: overwrite cached data with elder generation")

        if input.generation>self._generation:
            self._generation=input.generation

        if input.batchId==self.batchId:
            self._visible_info=input
        else:
            print("Mismatch:batchId")
        return
    
    @property
    def binning_info(self):
        if(self._binning_info.generation!=self._generation):
            print("Mismatch:generation")
            return None
        return self._binning_info
    @binning_info.setter
    def binning_info(self,input:BinningInfo):
        if self._binning_info is not None:
            if input.generation<self._binning_info.generation:
                print("Warning: overwrite cached data with elder generation")

        if input.generation>self._generation:
            self._generation=input.generation

        if input.batchId==self.batchId:
            self._binning_info=input
        else:
            print("Mismatch:batchId")
        return
    
    def check(self,generation:int=None):
        if generation is not None:
            self._generation=generation
        if self._visible_info.generation!=self._generation:
            print("Mismatch: visible_info_generation")
        if self._binning_info.generation!=self._generation:
            print("Mismatch: binning_info_generation")
        return