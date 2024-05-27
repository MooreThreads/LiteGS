from gaussian_splatting.model import GaussianSplattingModel
import torch
import numpy as np

class StatisticsHelper:
    def __init__(self,model:GaussianSplattingModel):
        self.model=model
        self.gs_num=self.model._xyz.shape[0]
        self.reset()
        return

    def update(self,visible_points):
        self.visible_count[visible_points]+=1
        
        self.grad_xyz_sum+=self.model._xyz.grad
        self.grad_xyz_square_sum+=(self.model._xyz.grad**2)

        self.grad_features_dc_sum+=self.model._features_dc.grad
        self.grad_features_dc_square_sum+=(self.model._features_dc.grad**2)
        #todo
        return
    
    def reset(self):
        self.visible_count=torch.zeros(self.gs_num,device='cuda')

        self.grad_xyz_sum=torch.zeros_like(self.model._xyz)
        self.grad_xyz_square_sum=torch.zeros_like(self.model._xyz)

        self.grad_features_dc_sum=torch.zeros_like(self.model._features_dc)
        self.grad_features_dc_square_sum=torch.zeros_like(self.model._features_dc)
        #todo
        return

    def get_gradient_std(self):

        def calc_std(square_sum:torch.Tensor,sum:torch.Tensor,count:torch.Tensor):
            grad_mean=(sum.transpose(0,1)/count).transpose(0,1)
            grad_square_mean=(square_sum.transpose(0,1)/count).transpose(0,1)
            grad_std=grad_square_mean-grad_mean**2
            return grad_std
        
        grad_xyz_std=calc_std(self.grad_xyz_square_sum,self.grad_xyz_sum,self.visible_count)
        grad_feature_dc_std=calc_std(self.grad_features_dc_square_sum[:,0,:],self.grad_features_dc_sum[:,0,:],self.visible_count)
        return grad_xyz_std,grad_feature_dc_std
    
class DensityController:
    def __init__(self) -> None:
        self.split_offset=torch.tensor([[-1,-1,-1],
                                        [-1,-1,1],
                                        [-1,1,-1],
                                        [-1,1,1],
                                        [1,-1,-1],
                                        [1,-1,1],
                                        [1,1,-1],
                                        [1,1,1]],
                                        dtype=torch.float32,device='cuda').unsqueeze(0)
        self.split_decay=0.5
        self.threshold=2#threshold*std
        return
    
    def get_abnormal(self,helper:StatisticsHelper)->list[int]:
        '''
        check the statistics data and return the indices of abnormal gaussian points.
        '''
        grad_xyz_std,grad_color_std=helper.get_gradient_std()
        xyz_mask=(grad_xyz_std>grad_xyz_std.mean(dim=0)+self.threshold*grad_xyz_std.std(dim=0))
        color_mask=(grad_color_std>grad_color_std.mean(dim=0)+self.threshold*grad_color_std.std(dim=0))
        abnormal_indices=(xyz_mask.sum(-1)+color_mask.sum(-1)).nonzero()[:,0]
        return abnormal_indices
    
    def split(self,abnormal_indices,gaussian_model:GaussianSplattingModel):
        scale_vec=gaussian_model._scaling[abnormal_indices].exp()
        rotator_vec=torch.nn.functional.normalize(gaussian_model._rotation[abnormal_indices],dim=2)
        transform_matrix=gaussian_model.create_transform_matrix(scale_vec,rotator_vec)#[N,3,3]
        mean_offset=self.split_offset@transform_matrix#[N,8,3]
        
        new_xyz=gaussian_model._xyz[abnormal_indices]
        new_xyz=new_xyz.unsqueeze(1)+mean_offset
        new_scaling=gaussian_model._scaling[abnormal_indices].exp()
        new_features_dc=gaussian_model._features_dc[abnormal_indices]


        return