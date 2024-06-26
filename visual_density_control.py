from loader import TrainingDataLoader
from argparse import ArgumentParser
import typing
from loader.InfoLoader import CameraInfo,ImageInfo
from gaussian_splatting.scene import GaussianScene
from gaussian_splatting.model import GaussianSplattingModel
from training.training import ViewManager
from training.densitycontroller import DensityControllerOurs
import torch
from util import tiles2img_torch,statistic_helper
from training.utils import loss_utils
from matplotlib import pyplot as plt 
from training.arguments import OptimizationParams
import time

plt_index=[0,14]

#load training data
cameras_info:typing.Dict[int,CameraInfo]=None
images_info:typing.List[ImageInfo]=None
scene:GaussianScene=None
cameras_info,images_info,scene,_,NerfNormRadius=TrainingDataLoader.load('./dataset/garden','images',2,-1)
image_size=images_info[0].image.size

args = OptimizationParams(ArgumentParser(description="Training script parameters"))


gaussian_model=GaussianSplattingModel(scene,NerfNormRadius)
gaussian_model.update_tiles_coord(image_size,16)
(model_params,op_state_dict, first_iter,actived_sh_degree) = torch.load('output/chkpnt200_baseline.pth')
gaussian_model.load_params(model_params)
gaussian_model.actived_sh_degree=actived_sh_degree
l = [
            {'params': [gaussian_model._xyz], 'lr': args.position_lr_init * 0.1, "name": "xyz"},
            {'params': [gaussian_model._features_dc], 'lr': args.feature_lr, "name": "f_dc"},
            {'params': [gaussian_model._features_rest], 'lr': args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [gaussian_model._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            {'params': [gaussian_model._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [gaussian_model._rotation], 'lr': args.rotation_lr, "name": "rotation"}
        ]
optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
optimizer.load_state_dict(op_state_dict)


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
ssim_helper=loss_utils.LossSSIM().cuda()
statistic_helper.StatisticsHelperInst.reset(gaussian_model._xyz.shape[0])
statistic_helper.StatisticsHelperInst.start()
density_controller=DensityControllerOurs(args)
start=time.monotonic()
for i in range(0,view_matrix.shape[0],1):
    with torch.no_grad():
        view_matrix_batch=view_matrix[i:i+1]
        view_project_matrix_batch=view_project_matrix[i:i+1]
        camera_focal_batch=camera_focal[i:i+1]
        camera_center_batch=camera_center[i:i+1]
        ground_truth_batch=ground_truth[i:i+1]

    tile_img,tile_transmitance=gaussian_model.render(None,None,
                              view_matrix_batch,view_project_matrix_batch,camera_focal_batch,camera_center_batch,None,
                              None)
    img=tiles2img_torch(tile_img,gaussian_model.cached_tiles_size[0],gaussian_model.cached_tiles_size[1])[...,:image_size[1],:image_size[0]]
    transmitance=tiles2img_torch(tile_transmitance,gaussian_model.cached_tiles_size[0],gaussian_model.cached_tiles_size[1])[...,:image_size[1],:image_size[0]]

    l1_loss=loss_utils.l1_loss(img,ground_truth_batch)
    ssim_loss=ssim_helper.loss(img,ground_truth_batch)
    loss=(1.0-0.2)*l1_loss+0.2*(1-ssim_loss)
    loss.backward()
    statistic_helper.StatisticsHelperInst.update_mean_std('xyz_grad',gaussian_model._xyz.grad.unsqueeze(0))
    optimizer.zero_grad(set_to_none = True)

    # with torch.no_grad():
    #     if i in plt_index:
    #         diff_img=torch.abs(img - ground_truth_batch)
torch.cuda.synchronize()
end=time.monotonic()
print("takes:{0} ms".format(end-start))
#########################################
# plot abnormal points
#########################################
# with torch.no_grad():
#     clone_mask=density_controller.densify_and_clone(gaussian_model)
#     gaussian_model._features_dc[clone_mask,0,0]=10.0#set r=1.0
#     for i in plt_index:
#         with torch.no_grad():
#             view_matrix_batch=view_matrix[i:i+1]
#             view_project_matrix_batch=view_project_matrix[i:i+1]
#             camera_focal_batch=camera_focal[i:i+1]
#             camera_center_batch=camera_center[i:i+1]
#             ground_truth_batch=ground_truth[i:i+1]

#         tile_img,tile_transmitance=gaussian_model.render(None,None,
#                                 view_matrix_batch,view_project_matrix_batch,camera_focal_batch,camera_center_batch,None,
#                                 None)
#         img=tiles2img_torch(tile_img,gaussian_model.cached_tiles_size[0],gaussian_model.cached_tiles_size[1])[...,:image_size[1],:image_size[0]]
#         plt.imshow(img[0].transpose(0,2).transpose(0,1).cpu())

#########################################
# generate points
#########################################
# with torch.no_grad():
#     density_controller.densify_and_prune(gaussian_model,optimizer,False)

#########################################
# train 10 epoch
#########################################
# statistic_helper.StatisticsHelperInst.reset(gaussian_model._xyz.shape[0])
# statistic_helper.StatisticsHelperInst.start()
# for epoch_i in range(10):
#     for i in range(view_matrix.shape[0]):
#         with torch.no_grad():
#             view_matrix_batch=view_matrix[i:i+1]
#             view_project_matrix_batch=view_project_matrix[i:i+1]
#             camera_focal_batch=camera_focal[i:i+1]
#             camera_center_batch=camera_center[i:i+1]
#             ground_truth_batch=ground_truth[i:i+1]

#         tile_img,tile_transmitance=gaussian_model.render(None,None,
#                                 view_matrix_batch,view_project_matrix_batch,camera_focal_batch,camera_center_batch,None,
#                                 None)
#         img=tiles2img_torch(tile_img,gaussian_model.cached_tiles_size[0],gaussian_model.cached_tiles_size[1])[...,:image_size[1],:image_size[0]]
#         transmitance=tiles2img_torch(tile_transmitance,gaussian_model.cached_tiles_size[0],gaussian_model.cached_tiles_size[1])[...,:image_size[1],:image_size[0]]

#         l1_loss=loss_utils.l1_loss(img,ground_truth_batch)
#         ssim_loss=ssim_helper.loss(img,ground_truth_batch)
#         loss=(1.0-0.2)*l1_loss+0.2*(1-ssim_loss)
#         loss.backward()
#         if epoch_i==4:
#             statistic_helper.StatisticsHelperInst.update_mean_std('xyz_grad',gaussian_model._xyz.grad.unsqueeze(0))
#         optimizer.step()
#         optimizer.zero_grad(set_to_none = True)

#         if epoch_i==4:
#             with torch.no_grad():
#                 if i in plt_index:
#                     diff_img=torch.abs(img - ground_truth_batch)

#########################################
# plot abnormal points
#########################################
# with torch.no_grad():
#     clone_mask=density_controller.densify_and_clone(gaussian_model)
#     gaussian_model._features_dc[clone_mask,0,0]=10.0#set r=1.0
#     for i in plt_index:
#         with torch.no_grad():
#             view_matrix_batch=view_matrix[i:i+1]
#             view_project_matrix_batch=view_project_matrix[i:i+1]
#             camera_focal_batch=camera_focal[i:i+1]
#             camera_center_batch=camera_center[i:i+1]
#             ground_truth_batch=ground_truth[i:i+1]

#         tile_img,tile_transmitance=gaussian_model.render(None,None,
#                                 view_matrix_batch,view_project_matrix_batch,camera_focal_batch,camera_center_batch,None,
#                                 None)
#         img=tiles2img_torch(tile_img,gaussian_model.cached_tiles_size[0],gaussian_model.cached_tiles_size[1])[...,:image_size[1],:image_size[0]]
#         plt.imshow(img[0].transpose(0,2).transpose(0,1).cpu())