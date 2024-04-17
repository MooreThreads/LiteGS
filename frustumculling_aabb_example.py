from loader.InfoLoader import CameraInfo,ImageInfo
from loader import TrainingDataLoader
from util.camera import View
from util import cg_torch

import typing
import numpy as np
import torch


if __name__ == "__main__":
    cameras_info:typing.Dict[int,CameraInfo]=None
    images_info:typing.List[ImageInfo]=None
    cameras_info,images_info,scene,_,NerfNormRadius=TrainingDataLoader.load('./dataset/garden','images_4',0)

    viewproj_matrix_list=[]
    for img in images_info:
        camera_info=cameras_info[img.camera_id]
        cur_view=View(img.viewtransform_position,img.viewtransform_rotation,camera_info.fovX,camera_info.fovY,np.array(img.image),img.name)
        viewproj_matrix_list.append(np.expand_dims(cur_view.viewproj_matrix,0))
    viewproj_matrix=np.concatenate(viewproj_matrix_list,axis=0)


    viewproj_matrix=torch.Tensor(viewproj_matrix).cuda()
    frustumplane=cg_torch.viewproj_to_frustumplane(viewproj_matrix)
    M=100
    aabb_origin=torch.randint(-50,50,(M,3)).cuda()
    aabb_extend=torch.ones_like(aabb_origin)*0.1
    aabb_visibility=cg_torch.frustum_culling_aabb(frustumplane,aabb_origin,aabb_extend)

    visible_item=aabb_visibility.nonzero()
    for view_id,obj_id in visible_item.cpu():
        cur_viewproj_matrix=viewproj_matrix[view_id]
        cur_aabb_origin=aabb_origin[obj_id]
        cur_aabb_origin=torch.concat([cur_aabb_origin,torch.ones((1)).cuda()])
        hom_pos=cur_aabb_origin@cur_viewproj_matrix
        ndc_xy=hom_pos[0:2]/hom_pos[3]
        print(ndc_xy)
    pass