import torch
import math

def world_to_ndc(position,view_project_matrix):
    hom_pos=torch.matmul(position,view_project_matrix)
    ndc_pos=hom_pos/(hom_pos[...,3:4]+1e-7)
    return ndc_pos

def world_to_view(position,view_matrix):
    return position@view_matrix

def quaternion_to_rotation_matrix(rotator_vec:torch.Tensor)->torch.Tensor:
    rotation_matrix=torch.zeros((*(rotator_vec.shape[0:-1]),3,3),device='cuda')

    r=rotator_vec[...,0]
    x=rotator_vec[...,1]
    y=rotator_vec[...,2]
    z=rotator_vec[...,3]


    rotation_matrix[...,0,0]=1 - 2 * (y * y + z * z)
    rotation_matrix[...,0,1]=2 * (x * y + r * z)
    rotation_matrix[...,0,2]=2 * (x * z - r * y)

    rotation_matrix[...,1,0]=2 * (x * y - r * z)
    rotation_matrix[...,1,1]=1 - 2 * (x * x + z * z)
    rotation_matrix[...,1,2]=2 * (y * z + r * x)

    rotation_matrix[...,2,0]=2 * (x * z + r * y)
    rotation_matrix[...,2,1]=2 * (y * z - r * x)
    rotation_matrix[...,2,2]=1 - 2 * (x * x + y * y)
    return rotation_matrix

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

@torch.no_grad
def make_aabb_mesh(aabb_origin:torch.Tensor,aabb_ext:torch.Tensor):
    '''
    Parameters:
        aabb_origin - the origin of Axis-Aligned Bounding Boxes. [N,3]

        aabb_ext - the extension of Axis-Aligned Bounding Boxes. [N,3]
    Returns:
        triangles - world coords of vertexs for each triangle. [N,6*2(triangle),3(vertex),3(position)]
    '''
    #   011-----111
    #  /|       /|
    # 010----110 |
    # | 001----|101
    # |/       |/
    # 000-----100
    assert(aabb_origin.shape[0]==aabb_ext.shape[0])

    #vertex
    V_000=aabb_origin-aabb_ext

    V_001=V_000.clone()
    V_001[:,2]+=2*aabb_ext[:,2]

    V_010=V_000.clone()
    V_010[:,1]+=2*aabb_ext[:,1]

    V_100=V_000.clone()
    V_100[:,0]+=2*aabb_ext[:,0]

    V_011=V_001.clone()
    V_011[:,1]+=2*aabb_ext[:,1]

    V_101=V_001.clone()
    V_101[:,0]+=2*aabb_ext[:,0]

    V_110=V_010.clone()
    V_110[:,0]+=2*aabb_ext[:,0]

    V_111=aabb_origin+aabb_ext

    #trangle
    N=aabb_origin.shape[0]
    triangles=torch.zeros((N,12,3,3),device=aabb_origin.device)

    triangles[:,0,0]=V_000
    triangles[:,0,1]=V_100
    triangles[:,0,2]=V_010
    triangles[:,1,0]=V_010
    triangles[:,1,1]=V_100
    triangles[:,1,2]=V_110

    triangles[:,2,0]=V_001
    triangles[:,2,1]=V_011
    triangles[:,2,2]=V_101
    triangles[:,3,0]=V_101
    triangles[:,3,1]=V_011
    triangles[:,3,2]=V_111

    triangles[:,4,0]=V_010
    triangles[:,4,1]=V_110
    triangles[:,4,2]=V_011
    triangles[:,5,0]=V_011
    triangles[:,5,1]=V_110
    triangles[:,5,2]=V_111

    triangles[:,6,0]=V_000
    triangles[:,6,1]=V_001
    triangles[:,6,2]=V_100
    triangles[:,7,0]=V_100
    triangles[:,7,1]=V_001
    triangles[:,7,2]=V_101

    triangles[:,8,0]=V_000
    triangles[:,8,1]=V_010
    triangles[:,8,2]=V_001
    triangles[:,9,0]=V_001
    triangles[:,9,1]=V_010
    triangles[:,9,2]=V_011

    triangles[:,10,0]=V_100
    triangles[:,10,1]=V_101
    triangles[:,10,2]=V_110
    triangles[:,11,0]=V_110
    triangles[:,11,1]=V_101
    triangles[:,11,2]=V_111

    return triangles


@torch.no_grad
def raster_large_triangle(vertices_ndc_pos:torch.Tensor,vertices_property:torch.Tensor,H:int,W:int,device='cuda'):
    '''
    Parameters:
        vertices_ndc_pos - NDC coords of vertex.  [N,3,3]

        vertices_property - property of vertex to be rasterized. [M,N,3]

        H,W - screen size
    Returns:
        (texture[H,W,M],depth[H,W,1])  
    '''
    if vertices_property is None:
        N=vertices_ndc_pos.shape[0]
        M=0
    else:
        (M,N,_)=vertices_property.shape
    assert(N*H*W!=0)
    assert(N*H*W<1024*1024*1024)

    texture=torch.zeros((H,W,M),device=vertices_ndc_pos.device)
    pixel_index=torch.meshgrid(torch.arange(0,H,device=device),torch.arange(0,W,device=device))
    pixel_screen_pos=torch.stack((((pixel_index[1]+0.5)/W-0.5)*2,((pixel_index[0]+0.5)/H-0.5)*2),dim=-1).unsqueeze(2)#[H,W,1,2]
    pixel_screen_pos=pixel_screen_pos.repeat((1,1,N,1))#[H,W,N,2]
    
    #Barycentric Coordinates
    AB=vertices_ndc_pos[:,1,:2]-vertices_ndc_pos[:,0,:2]#[N,2]
    BC=vertices_ndc_pos[:,2,:2]-vertices_ndc_pos[:,1,:2]#[N,2]
    AC=vertices_ndc_pos[:,2,:2]-vertices_ndc_pos[:,0,:2]#[N,2]
    AP=pixel_screen_pos-vertices_ndc_pos[:,0,:2]#[H,W,N,2]

    area_B=AP[...,0]*AC[...,1]-AP[...,1]*AC[...,0]#|AP x AC| [H,W,N]
    area_C=AB[...,0]*AP[...,1]-AB[...,1]*AP[...,0]#|AB x AP| [H,W,N]
    triangle_area=AB[:,0]*BC[:,1]-AB[:,1]*BC[:,0]+1e-7#|AB x BC| [N]

    barycentric_beta=area_B/triangle_area
    barycentric_gamma=area_C/triangle_area
    barycentric_alpha=1-barycentric_beta-barycentric_gamma
    barycentric_coord=torch.stack((barycentric_alpha,barycentric_beta,barycentric_gamma),dim=-1)#[H,W,N,3]

    mask=((barycentric_coord>0).sum(dim=-1)==3)

    depth=(barycentric_coord*vertices_ndc_pos[:,:,2]).sum(dim=-1)
    depth=depth*mask+(~mask)*1#[H,W,N]
    if vertices_property is not None:
        texture=(barycentric_coord.unsqueeze(2)*vertices_property).sum(dim=-1)#[H,W,M,N]
    else:
        texture=None

    #occlusion
    visibility_buffer=torch.argmin(depth,dim=2,keepdim=True)
    depth=torch.gather(depth,2,visibility_buffer)
    if vertices_property is not None:
        property_list=[]
        for i in range(M):
            property_list.append(torch.gather(texture[:,:,i,:],2,visibility_buffer))
        texture=torch.concat(property_list,dim=-1)

    return depth,texture