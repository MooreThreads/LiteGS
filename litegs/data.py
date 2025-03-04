import math
import numpy as np
import numpy.typing as npt
import os
import PIL.Image
import torch
from torch.utils.data import Dataset

from . import utils

class CameraInfo:
    def __init__(self):
        self.id:int=0
        self.model:str=''
        self.width:int=0
        self.height:int=0
        return
    
    def __init__(self,id:int,model_name:str,width:int,height:int):
        self.id:int=id
        self.model:str=model_name
        self.width:int=width
        self.height:int=height
        return
    
    def get_project_matrix(self)->npt.NDArray:
        return None
    def get_focal(self):
        return None
    
class PinHoleCameraInfo(CameraInfo):
    def __init__(self,id:int,width:int,height:int,parameters:list[float],z_near=0.01,z_far=100.0):
        super(PinHoleCameraInfo,self).__init__(id,"PINHOLE",width,height)
        focal_length_x=parameters[0]
        focal_length_y=parameters[1]
        focal_x=focal_length_x/(width*0.5)
        focal_y=focal_length_y/(height*0.5)
        self.proj_matrix=np.array([[focal_x,0,0,0],
                  [0,focal_y,0,0],
                  [0,0,(z_far+z_near)/(z_far-z_near),-2*z_far*z_near/(z_far-z_near)],
                  [0,0,1,0]],dtype=np.float32).transpose()
        return
    
    def get_project_matrix(self):
        return self.proj_matrix
    
WARNED = False

class CameraFrame:
    def __init__(self):
        self.id:int=0
        self.viewtransform_rotation:npt.NDArray=np.array((0,0,0,0))
        self.viewtransform_position:npt.NDArray=np.array((0,0,0))
        self.camera_id:int=0
        self.name:str=None
        self.img_source:str=None
        self.xys=np.array((0,0,0,0))
        return
    
    def __init__(self,id:int,qvec:npt.ArrayLike,tvec:npt.ArrayLike,camera_id:int,name:str,img_source:str,xys:npt.ArrayLike):
        self.id:int=id
        viewtransform_rotation:npt.NDArray=utils.qvec2rotmat(np.array(qvec))
        viewtransform_position:npt.NDArray=np.array(tvec)
        self.view_matrix = utils.get_view_matrix(viewtransform_rotation,viewtransform_position).transpose()
        self.camera_center = -viewtransform_rotation.transpose()@viewtransform_position
        self.camera_id:int=camera_id
        self.name:str=name
        self.img_source:str=img_source
        self.xys:npt.NDArray=np.array(xys)
        self.image={}
        return
    
    def load_image(self,downsample:int=-1):
        if self.image.get(downsample,None) is None:
            image:PIL.Image.Image=PIL.Image.open(self.img_source)

            orig_w, orig_h = image.size
            if downsample in [1, 2, 4, 8]:
                resolution = round(orig_w/ downsample), round(orig_h/ downsample)
            else:  # should be a type that converts to float
                if downsample == -1:
                    if orig_w > 1600:
                        global WARNED
                        if not WARNED:
                            print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                                "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                            WARNED = True
                        global_down = orig_w / 1600
                    else:
                        global_down = 1
                else:
                    global_down = orig_w / downsample

                scale = float(global_down)
                resolution = (int(orig_w / scale), int(orig_h / scale))  
            self.image[downsample]=np.array(image.resize(resolution),dtype=np.float32).transpose(2,0,1)/255.0
        return self.image[downsample]
    
    def get_viewmatrix(self)->npt.NDArray:
        return self.view_matrix
    
    def get_camera_center(self)->npt.NDArray:
        return self.camera_center
    
class CameraFrameDataset(Dataset):
    def __init__(self,cameras:list[CameraInfo],frames:list[CameraFrame],downsample:int=-1):
        self.cameras=cameras
        self.frames=frames
        self.downsample=downsample
        return
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,idx:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        image=self.frames[idx].load_image(self.downsample)
        view_matrix=self.frames[idx].get_viewmatrix()
        proj_matrix=self.cameras[self.frames[idx].camera_id].get_project_matrix()
        
        return torch.Tensor(view_matrix),torch.Tensor(proj_matrix),torch.Tensor(image)
    
    def get_norm(self)->tuple[float,float]:
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for frame in self.frames:
            W2C = frame.get_viewmatrix()
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1
        translate = -center
        return translate,radius
        