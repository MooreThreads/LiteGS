import typing
import numpy as np
import torch
import util.cg_torch
from matplotlib import pyplot as plt 
import util.camera

def make_view_matrix(camera_position):
    view_matrix=torch.eye(4,device=camera_position.device)
    view_matrix[3,0:3]=camera_position
    return view_matrix

if __name__ == "__main__":

    camera=util.camera.Camera(np.array((0,0,5)),np.eye(3,3),90,90,0.01,100)
    viewproj_matrix=torch.tensor(camera.viewproj_matrix,device='cuda',dtype=torch.float32)

    #create aabb mesh
    N=4
    aabb_origin=torch.rand((N,3),device='cuda')
    aabb_origin[:,0:2]=(aabb_origin[:,0:2]-0.5)*2*10
    aabb_origin[:,2]=aabb_origin[:,2]*10
    aabb_ext=torch.ones((N,3),device='cuda')
    trangles=util.cg_torch.make_aabb_mesh(aabb_origin,aabb_ext)
    helper=torch.ones((N,12,3,1),device='cuda')

    #project
    trangles=torch.cat((trangles,helper),dim=-1)
    ndc_trangles=util.cg_torch.world_to_ndc(trangles,viewproj_matrix)

    #raster
    ndc_trangles=ndc_trangles[...,:3].reshape(-1,3,3)
    depth,texture=util.cg_torch.raster_large_triangle(ndc_trangles,None,1080,1920)

    #show
    plt.imshow(1-depth[:,:,0].cpu(),'gray')
    plt.show()

    pass