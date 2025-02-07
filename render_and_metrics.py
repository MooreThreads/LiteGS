from argparse import ArgumentParser
from loader.InfoLoader import CameraInfo,ImageInfo
from loader import TrainingDataLoader
from gaussian_splatting.scene import GaussianScene
from gaussian_splatting.model import GaussianSplattingModel
from training.view_manager import ViewManager
from training.training import GaussianTrainer
import typing
from training.arguments import ModelParams
import os
import torch
import torchvision
from torchmetrics.image import psnr,ssim,lpip

def report_result(inference_results:list,result_name:str):
    acc_psnr=0
    acc_ssim=0
    acc_lpips=0
    for (img_psnr,img_ssim,img_lpips) in inference_results:
        acc_psnr+=img_psnr
        acc_ssim+=img_ssim
        acc_lpips+=img_lpips
    psnr_mean=acc_psnr/len(inference_results)
    ssim_mean=acc_ssim/len(inference_results)
    lpips_mean=acc_lpips/len(inference_results)
    
    print("  Scene:{0}".format(result_name))
    print("  SSIM : {:>12.7f}".format(float(ssim_mean)))
    print("  PSNR : {:>12.7f}".format(float(psnr_mean)))
    print("  LPIPS: {:>12.7f}".format(float(lpips_mean)))
    print("")
    return

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model_params = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = parser.parse_args()
    model_params=model_params.extract(args)
    
    
    cameras_info:typing.Dict[int,CameraInfo]=None
    images_info:typing.List[ImageInfo]=None
    scene=GaussianScene(model_params.sh_degree)
    iteration_folder:str='finish'
    if args.iteration!=-1:
        iteration_folder='iteration_{0}'.format(args.iteration)

    #load data
    scene.load_ply(os.path.join(model_params.model_path,'point_cloud',iteration_folder,'point_cloud.ply'))
    cameras_info,images_info,_,_,_=TrainingDataLoader.load(model_params.source_path,model_params.images,model_params.sh_degree,model_params.resolution)
    gs_model=GaussianSplattingModel(scene,model_params.sh_degree)
    tile_size=8
    gs_model.update_tiles_coord(images_info[0].image.size,tile_size)

    #eval
    trainingset=[c for idx, c in enumerate(images_info) if idx % 8 != 0]
    testset=[c for idx, c in enumerate(images_info) if idx % 8 == 0]
    view_manager_train=ViewManager(trainingset,cameras_info)
    view_manager_test=ViewManager(testset,cameras_info)

    #render
    output_path=os.path.join(model_params.model_path,'point_cloud',iteration_folder)
    os.makedirs(os.path.join(output_path,'test'), exist_ok=True)
    os.makedirs(os.path.join(output_path,'train'), exist_ok=True)

    #metrics
    ssim_metrics=ssim.StructuralSimilarityIndexMeasure(data_range=(0.0,1.0)).cuda()
    psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
    lpip_metrics=lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    with torch.no_grad():
        # def metrics_train(iter_i:int,img_name:str,out_img:torch.Tensor,ground_truth:torch.Tensor)->torch.Tensor:
        #     torchvision.utils.save_image(ground_truth[0],os.path.join(output_path,'train','{0}_gt.png'.format(img_name)))
        #     torchvision.utils.save_image(out_img[0],os.path.join(output_path,'train','{0}.png'.format(img_name)))
        #     img_psnr=psnr_metrics(out_img,ground_truth)
        #     img_ssim=ssim_metrics(out_img,ground_truth)
        #     img_lpips=lpip_metrics(out_img,ground_truth)
        #     return img_psnr,img_ssim,img_lpips
        # train_result:list=GaussianTrainer.inference(gs_model,view_manager_train,False,metrics_train)
        # report_result(train_result,model_params.model_path+' training set')
        
        def metrics_test(iter_i:int,img_name:str,out_img:torch.Tensor,ground_truth:torch.Tensor)->torch.Tensor:
            torchvision.utils.save_image(ground_truth[0],os.path.join(output_path,'test','{0}_gt.png'.format(img_name)))
            torchvision.utils.save_image(out_img[0],os.path.join(output_path,'test','{0}.png'.format(img_name)))
            img_psnr=psnr_metrics(out_img,ground_truth)
            img_ssim=ssim_metrics(out_img,ground_truth)
            img_lpips=lpip_metrics(out_img,ground_truth)
            return img_psnr,img_ssim,img_lpips
        test_result=GaussianTrainer.inference(gs_model,view_manager_test,False,metrics_test)
        report_result(test_result,model_params.model_path+' testing set')



