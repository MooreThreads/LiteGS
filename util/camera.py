import numpy as np
import math
from util import getWorld2View,getProjectionMatrix

class Camera:
    def __init__(self,viewtrans_position,viewtrans_rotation,fovX,fovY,z_near=0.01,z_far=100.0):
        self.fovX=fovX
        self.fovY=fovY
        
        self.z_near=z_near
        self.z_far=z_far
        self.world2view_matrix=getWorld2View(viewtrans_rotation,viewtrans_position).transpose()
        self.project_matrix=getProjectionMatrix(znear=self.z_near, zfar=self.z_far, fovX=self.fovX, fovY=self.fovY).transpose()
        self.viewproj_matrix=np.matmul(self.world2view_matrix,self.project_matrix)
        self.viewproj_inv_matrix=np.linalg.inv(self.viewproj_matrix)
        self.camera_center = np.linalg.inv(self.world2view_matrix)[3, :3]
        return
    
class View(Camera):
    def __init__(self,viewtrans_position,viewtrans_rotation,fovX,fovY,image,image_name):
        super(View,self).__init__(viewtrans_position,viewtrans_rotation,fovX,fovY)
        self.image_name=image_name
        self.image=image

        self.focal_x=self.image.shape[1]/(math.tan(fovX*0.5)*2)
        self.focal_y=self.image.shape[0]/(math.tan(fovY*0.5)*2)
        return

