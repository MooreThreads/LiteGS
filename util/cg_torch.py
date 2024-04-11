import torch
import math

def world_to_ndc(position,view_project_matrix):
    hom_pos=torch.matmul(position,view_project_matrix)
    ndc_pos=hom_pos/(hom_pos[...,3:4]+1e-7)
    return ndc_pos

def world_to_view(position,view_matrix):
    return position@view_matrix

@torch.no_grad
def viewproj_to_frustumplane(viewproj_matrix:torch.Tensor)->torch.Tensor:
    '''
    Parameters:
        viewproj_matrix - the viewproj transform matrix. [N,4,4]
    Returns:
        frustumplane - the planes of view frustum. [N,6,4]
    '''
    N=viewproj_matrix.shape[0]
    frustumplane=torch.zeros((N,6,4),device=viewproj_matrix.device)
    #left plane
    frustumplane[:,0,0]=viewproj_matrix[:,0,3]+viewproj_matrix[:,0,0]
    frustumplane[:,0,1]=viewproj_matrix[:,1,3]+viewproj_matrix[:,1,0]
    frustumplane[:,0,2]=viewproj_matrix[:,2,3]+viewproj_matrix[:,2,0]
    frustumplane[:,0,3]=viewproj_matrix[:,3,3]+viewproj_matrix[:,3,0]
    #right plane
    frustumplane[:,1,0]=viewproj_matrix[:,0,3]-viewproj_matrix[:,0,0]
    frustumplane[:,1,1]=viewproj_matrix[:,1,3]-viewproj_matrix[:,1,0]
    frustumplane[:,1,2]=viewproj_matrix[:,2,3]-viewproj_matrix[:,2,0]
    frustumplane[:,1,3]=viewproj_matrix[:,3,3]-viewproj_matrix[:,3,0]

    #bottom plane

    #top plane

    #near plane

    #far plane
    return

@torch.no_grad
def frustum_culling_aabb(frustumplane,aabb_origin,aabb_ext):
    '''
    Parameters:
        frustumplane - the planes of view frustum. [N,6,4]
        aabb_origin - the origin of Axis-Aligned Bounding Boxes. [M,3]
        aabb_ext - the extension of Axis-Aligned Bounding Boxes. [M,3]
    Returns:
        visibility - is visible. [N,M]
    '''
    #project origin to plane normal

    #project extension to plane normal

    #push out the origin

    #is completely outside

    return
