import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import litegs.utils
from litegs.utils.wrapper import litegs_fused

if __name__ == "__main__":
    # ndc_pos=torch.tensor(np.load('./profiler_input_data/ndc.npy'),device='cuda').requires_grad_(True)
    # cov2d=torch.tensor(np.load('./profiler_input_data/cov2d.npy'),device='cuda').requires_grad_(True)
    # color=torch.tensor(np.load('./profiler_input_data/color.npy'),device='cuda').requires_grad_(True)
    # opacity=torch.tensor(np.load('./profiler_input_data/opacity.npy'),device='cuda').requires_grad_(True)
    # output_shape=[1036,1600]
    # tile_size=(8,16)

    # #with torch.no_grad():
    # for i in range(10):
    #     eigen_val,eigen_vec,inv_cov2d=litegs.utils.wrapper.EighAndInverse2x2Matrix.call_fused(cov2d)
    #     tile_start_index,sorted_pointId,b_visible=litegs.utils.wrapper.Binning.call_fused(ndc_pos,eigen_val,eigen_vec,opacity,output_shape,tile_size)
    #     img,transmitance,depth,normal=litegs.utils.wrapper.GaussiansRasterFunc.apply(sorted_pointId,tile_start_index,ndc_pos,inv_cov2d,color,opacity,None,
    #                                             output_shape[0],output_shape[1],tile_size[0],tile_size[1],False,False)
    #     img.mean().backward()
    #     # plt_img=litegs.utils.tiles2img_torch(img,math.ceil(output_shape[1]/tile_size[1]),math.ceil(output_shape[0]/tile_size[0]))[...,:output_shape[0],:output_shape[1]].contiguous()
    #     # plt.imshow(plt_img.detach().cpu()[0].permute(1,2,0))
    #     # plt.show()
    #     pass

    sorted_pointId=torch.tensor(np.load('./nan_data/sorted_pointId.npy'),device='cuda')
    tile_start_index=torch.tensor(np.load('./nan_data/tile_start_index.npy'),device='cuda')
    packed_params=torch.tensor(np.load('./nan_data/packed_params.npy'),device='cuda')
    transmitance=torch.tensor(np.load('./nan_data/transmitance.npy'),device='cuda')
    lst_contributor=torch.tensor(np.load('./nan_data/lst_contributor.npy'),device='cuda')
    grad_rgb_image=torch.tensor(np.load('./nan_data/grad_rgb_image.npy'),device='cuda')
    grad_rgb_image_max=torch.tensor(np.load('./nan_data/grad_rgb_image_max.npy'),device='cuda')

    _img,_trans,_depth,_lst_contributor=litegs_fused.rasterize_forward_packed(sorted_pointId,tile_start_index,packed_params,None,
                                                                              1063, 1600, 8, 16,False,False)

    grad_ndc,grad_cov2d_inv,grad_color,grad_opacities=litegs_fused.rasterize_backward(sorted_pointId,tile_start_index,packed_params,None,
                                                                                    #transmitance,lst_contributor,
                                                                                    _trans,_lst_contributor,
                                                                                    grad_rgb_image,None,None,grad_rgb_image_max,
                                                                                    1063, 1600, 8, 16)
    
    if grad_color.isnan().any() or grad_color.isinf().any() \
            or grad_opacities.isnan().any() or grad_opacities.isinf().any() \
                or grad_cov2d_inv.isnan().any() or grad_cov2d_inv.isinf().any() \
                    or grad_ndc.isnan().any() or grad_ndc.isinf().any():
            breakpoint()
    pass