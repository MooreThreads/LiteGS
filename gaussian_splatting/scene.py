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
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self.position=xyz
        self.sh_coefficient_dc=features_dc
        self.sh_coefficient_rest=features_extra
        self.opacity=opacities
        self.scale=scales
        self.rotator=rots
        return
    


