import typing
from util.camera import View
from loader.InfoLoader import CameraInfo,ImageInfo,PinHoleCameraInfo
import numpy as np

class ViewManager:

    def __init__(self,image_list:typing.List[ImageInfo],camera_dict:typing.Dict[int,PinHoleCameraInfo]):
        self.view_list:typing.List[View]=[]
        self.view_matrix_tensor=None
        self.proj_matrix_tensor=None
        self.camera_center_tensor=None

        view_matrix_list=[]
        proj_matrix_list=[]
        camera_center_list=[]
        camera_focal_list=[]
        gt_list=[]
        for img in image_list:
            camera_info=camera_dict[img.camera_id]
            cur_view=View(img.viewtransform_position,img.viewtransform_rotation,camera_info.fovX,camera_info.fovY,np.array(img.image),img.name)
            self.view_list.append(cur_view)
            view_matrix_list.append(np.expand_dims(cur_view.world2view_matrix,0))
            proj_matrix_list.append(np.expand_dims(cur_view.project_matrix,0))
            camera_center_list.append(np.expand_dims(cur_view.camera_center,0))
            camera_focal_list.append(np.expand_dims((cur_view.focal_x,cur_view.focal_y),0))
            gt_list.append(np.expand_dims(cur_view.image,0))
        self.view_matrix_tensor=np.concatenate(view_matrix_list)
        self.proj_matrix_tensor=np.concatenate(proj_matrix_list)
        self.camera_center_tensor=np.concatenate(camera_center_list)
        self.camera_focal_tensor=np.concatenate(camera_focal_list)
        self.view_gt_tensor=np.concatenate(gt_list)/255.0
        self.view_gt_tensor=self.view_gt_tensor.transpose(0,3,1,2)
        self.image_size=image_list[0].image.size

        return
    