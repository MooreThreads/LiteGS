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
    frustumplane[:,2,0]=viewproj_matrix[:,0,3]+viewproj_matrix[:,0,1]
    frustumplane[:,2,1]=viewproj_matrix[:,1,3]+viewproj_matrix[:,1,1]
    frustumplane[:,2,2]=viewproj_matrix[:,2,3]+viewproj_matrix[:,2,1]
    frustumplane[:,2,3]=viewproj_matrix[:,3,3]+viewproj_matrix[:,3,1]

    #top plane
    frustumplane[:,3,0]=viewproj_matrix[:,0,3]-viewproj_matrix[:,0,1]
    frustumplane[:,3,1]=viewproj_matrix[:,1,3]-viewproj_matrix[:,1,1]
    frustumplane[:,3,2]=viewproj_matrix[:,2,3]-viewproj_matrix[:,2,1]
    frustumplane[:,3,3]=viewproj_matrix[:,3,3]-viewproj_matrix[:,3,1]

    #near plane
    frustumplane[:,4,0]=viewproj_matrix[:,0,2]
    frustumplane[:,4,1]=viewproj_matrix[:,1,2]
    frustumplane[:,4,2]=viewproj_matrix[:,2,2]
    frustumplane[:,4,3]=viewproj_matrix[:,3,2]

    #far plane
    frustumplane[:,5,0]=viewproj_matrix[:,0,3]-viewproj_matrix[:,0,2]
    frustumplane[:,5,1]=viewproj_matrix[:,1,3]-viewproj_matrix[:,1,2]
    frustumplane[:,5,2]=viewproj_matrix[:,2,3]-viewproj_matrix[:,2,2]
    frustumplane[:,5,3]=viewproj_matrix[:,3,3]-viewproj_matrix[:,3,2]

    return frustumplane

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
    assert(aabb_origin.shape[0]==aabb_ext.shape[0])
    N=frustumplane.shape[0]
    M=aabb_origin.shape[0]
    frustumplane=frustumplane.unsqueeze(1)
    aabb_origin=aabb_origin.unsqueeze(1).unsqueeze(0)
    aabb_ext=aabb_ext.unsqueeze(1).unsqueeze(0)
    #project origin to plane normal [M,N,6,1]
    dist_origin=(frustumplane[...,0:3]*aabb_origin).sum(-1)+frustumplane[...,3]
    #project extension to plane normal
    dist_ext=(frustumplane[...,0:3]*aabb_ext).abs().sum(-1)
    #push out the origin
    pushed_origin_dist=dist_origin+dist_ext #M,N,6,1
    #is completely outside
    culling=(pushed_origin_dist<0).sum(-1)
    visibility=(culling==0)
    return visibility
