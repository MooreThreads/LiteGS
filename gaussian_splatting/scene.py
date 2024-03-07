import typing
import numpy.typing as npt
import numpy as np
import torch
import math
from plyfile import PlyData, PlyElement
import os


class GaussianScene:
    def __init__(self,sh_degree=None,
                 position:npt.NDArray=None,scale:npt.NDArray=None,rotator:npt.NDArray=None,
                 sh_coefficient:npt.NDArray=None,
                 opacity:npt.NDArray=None):
        
        self.position:npt.NDArray=position
        self.scale:npt.NDArray=scale
        self.rotator:npt.NDArray=rotator
        self.sh_degree=sh_degree
        if sh_coefficient is not None:
            self.sh_coefficient_dc:npt.NDArray=sh_coefficient[:,:,0:1]
            self.sh_coefficient_rest:npt.NDArray=sh_coefficient[:,:,1:]
        else:
            self.sh_coefficient_dc:npt.NDArray=None
            self.sh_coefficient_rest:npt.NDArray=None
        self.opacity:npt.NDArray=opacity

        if self.position is not None:
            assert(self.position.shape[0]==self.rotator.shape[0])
        return
    
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.sh_coefficient_dc.shape[-1]*self.sh_coefficient_dc.shape[-2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.sh_coefficient_rest.shape[-1]*self.sh_coefficient_rest.shape[-2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scale.shape[-1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotator.shape[-1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self,path:str):
        dirname=os.path.dirname(path)
        if os.path.exists(dirname)==False:
            os.makedirs(dirname)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        points_num=self.position.shape[0]
        assert(self.scale.shape[0]==points_num)
        assert(self.rotator.shape[0]==points_num)
        assert(self.sh_coefficient_dc.shape[0]==points_num)
        assert(self.opacity.shape[0]==points_num)
        elements = np.empty(points_num, dtype=dtype_full)
        attributes = np.concatenate((self.position, np.zeros_like(self.position), self.sh_coefficient_dc.reshape(points_num,-1), self.sh_coefficient_rest.reshape(points_num,-1), 
                                     self.opacity, self.scale, self.rotator), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        return
    
    def load_ply(self,path:str):

        return
    


