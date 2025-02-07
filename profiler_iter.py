from loader import TrainingDataLoader
from argparse import ArgumentParser
import typing
from loader.InfoLoader import CameraInfo,ImageInfo
from gaussian_splatting.scene import GaussianScene
from gaussian_splatting.model import GaussianSplattingModel
from training.training import ViewManager
import torch
from util import tiles2img_torch
from matplotlib import pyplot as plt 
from training.arguments import OptimizationParams
import time
import fused_ssim

all_scene=["bicycle","bonsai","counter","flowers","garden","kitchen","room","stump","treehill"]
scene_str="bonsai"#all_scene[8]

#load training data
cameras_info:typing.Dict[int,CameraInfo]=None
images_info:typing.List[ImageInfo]=None
scene:GaussianScene=None
cameras_info,images_info,scene,_,NerfNormRadius=TrainingDataLoader.load('./dataset/{0}/A100_colmap-default_gaussian-splatting-default'.format(scene_str),'images',3,-1)
image_size=images_info[0].image.size

args = OptimizationParams(ArgumentParser(description="Training script parameters"))
scene.load_ply('./output/{0}/point_cloud/finish/point_cloud.ply'.format(scene_str))

gaussian_model=GaussianSplattingModel(scene,3)
gaussian_model.update_tiles_coord(image_size,8)

trainingset=[c for idx, c in enumerate(images_info) if idx % 8 != 0]
testset=[c for idx, c in enumerate(images_info) if idx % 8 == 0]
view_manager=ViewManager(trainingset,cameras_info)
view_manager_testset=ViewManager(testset,cameras_info)

view_matrix=torch.Tensor(view_manager.view_matrix_tensor).cuda()
view_project_matrix=view_matrix@(torch.Tensor(view_manager.proj_matrix_tensor).cuda())
camera_center=torch.Tensor(view_manager.camera_center_tensor).cuda()
camera_focal=torch.Tensor(view_manager.camera_focal_tensor).cuda()
ground_truth=torch.Tensor(view_manager.view_gt_tensor).cuda()
total_views_num=view_matrix.shape[0]


start=time.monotonic()
for i in range(0,10):
    with torch.no_grad():
        view_matrix_batch=view_matrix[i:i+1]
        view_project_matrix_batch=view_project_matrix[i:i+1]
        camera_focal_batch=camera_focal[i:i+1]
        camera_center_batch=camera_center[i:i+1]
        ground_truth_batch=ground_truth[i:i+1].contiguous()

    tile_img,tile_transmitance,_=gaussian_model.render(view_matrix_batch,view_project_matrix_batch,camera_focal_batch,camera_center_batch,None,None)
    img=tiles2img_torch(tile_img,gaussian_model.cached_tiles_size[0],gaussian_model.cached_tiles_size[1])[...,:image_size[1],:image_size[0]].contiguous()
    img.mean().backward()
    #optimizer.zero_grad(set_to_none = True)

torch.cuda.synchronize()
end=time.monotonic()
print("takes:{0} ms".format(end-start))#19ms