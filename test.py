import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import litegs.utils

if __name__ == "__main__":
    ndc_pos=torch.tensor(np.load('./profiler_input_data/ndc.npy'),device='cuda').requires_grad_(True)
    cov2d=torch.tensor(np.load('./profiler_input_data/cov2d.npy'),device='cuda').requires_grad_(True)
    color=torch.tensor(np.load('./profiler_input_data/color.npy'),device='cuda').requires_grad_(True)
    opacity=torch.tensor(np.load('./profiler_input_data/opacity.npy'),device='cuda').requires_grad_(True)
    output_shape=[1036,1600]
    tile_size=8


    #test 87526
    # debug_id=87526
    # ndc_pos=ndc_pos[...,debug_id:debug_id+1]
    # cov2d=cov2d[...,debug_id:debug_id+1]
    # color=color[...,debug_id:debug_id+1]
    # opacity=opacity[...,debug_id:debug_id+1]
    for i in range(10):
        eigen_val,eigen_vec,inv_cov2d=litegs.utils.wrapper.EighAndInverse2x2Matrix.call_fused(cov2d)
        tile_start_index,sorted_pointId,b_visible=litegs.utils.wrapper.Binning.call_fused(ndc_pos,eigen_val,eigen_vec,opacity,output_shape,tile_size)
        img,transmitance,depth,normal=litegs.utils.wrapper.GaussiansRasterFunc.apply(sorted_pointId,tile_start_index,ndc_pos,inv_cov2d,color,opacity,None,
                                                output_shape[0],output_shape[1],tile_size,tile_size,False,False)
        img.mean().backward()
        # plt_img=litegs.utils.tiles2img_torch(img,math.ceil(output_shape[1]/tile_size),math.ceil(output_shape[0]/tile_size))[...,:output_shape[0],:output_shape[1]].contiguous()
        # plt.imshow(plt_img.detach().cpu()[0].permute(1,2,0))
        # plt.show()
        pass