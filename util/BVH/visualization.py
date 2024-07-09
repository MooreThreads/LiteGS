import numpy as np
import open3d.visualization
import torch
from util.BVH.Object import GSpointBatch
from util.BVH.PytorchBVH import BVH
import open3d

class VisualizationHelper:
    def __init__(self,bvh:BVH,obj_positions:torch.Tensor,obj_colors:torch.Tensor=None):
        self.reset(bvh,obj_positions,obj_colors)
        self.geometries=[]
        return
    
    def reset(self,bvh:BVH,obj_positions:torch.Tensor,obj_colors:torch.Tensor=None):
        assert(obj_positions.shape[-1]==3)
        self.bvh=bvh
        self.point_positions=obj_positions
        self.point_colors=obj_colors

        self.position_in_chunks=[]
        self.color_in_chunks=[]
        self.origin_list:list[torch.Tensor]=[]
        self.extend_list:list[torch.Tensor]=[]
        for node_index,node in enumerate(bvh.leaf_nodes):
            positions=self.point_positions[node.objs]
            self.position_in_chunks.append(positions)
            if self.point_colors is None:
                colors=torch.rand((1,3),device=positions.device).repeat(node.objs.shape[0],1)
            else:
                colors=self.point_colors[node.objs]
            self.color_in_chunks.append(colors)

            self.origin_list.append(node.origin)
            self.extend_list.append(node.extend)
        return
    

    def __get_pcd(self):
        pcd_position_data=torch.cat(self.position_in_chunks).cpu()
        pcd_color_data=torch.cat(self.color_in_chunks).cpu()

        pcd=open3d.geometry.PointCloud()
        pcd.points=open3d.utility.Vector3dVector(pcd_position_data)
        pcd.colors=open3d.utility.Vector3dVector(pcd_color_data)
        return pcd
    
    @torch.no_grad()
    def add_points(self):
        self.geometries.append(self.__get_pcd())
        return
    
    def __create_box(self):
        N=len(self.origin_list)
        boxes=[]
        for i in range(N):
            origin=np.array(self.origin_list[i].cpu())
            extend=np.array(self.extend_list[i].cpu())
            box = np.array([
                [origin[0]-extend[0],origin[1]-extend[1],origin[2]-extend[2]],
                [origin[0]-extend[0],origin[1]-extend[1],origin[2]+extend[2]],
                [origin[0]-extend[0],origin[1]+extend[1],origin[2]+extend[2]],
                [origin[0]-extend[0],origin[1]+extend[1],origin[2]-extend[2]],

                [origin[0]+extend[0],origin[1]-extend[1],origin[2]-extend[2]],
                [origin[0]+extend[0],origin[1]-extend[1],origin[2]+extend[2]],
                [origin[0]+extend[0],origin[1]+extend[1],origin[2]+extend[2]],
                [origin[0]+extend[0],origin[1]+extend[1],origin[2]-extend[2]],
            ])

            boxes.append(box)
        return boxes
    
    @torch.no_grad()
    def add_AABB_box(self):
        line=np.array([
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7],
        ])
        colors=np.zeros((12,3))
        colors[:,1]=1
        boxes=self.__create_box()
        for box in boxes:
            line_set = open3d.geometry.LineSet()
            line_set.points = open3d.utility.Vector3dVector(box)
            line_set.lines = open3d.utility.Vector2iVector(line)
            line_set.colors = open3d.utility.Vector3dVector(colors)
            self.geometries.append(line_set)
        return
    
    def draw(self):
        open3d.visualization.draw_geometries(self.geometries)
        return

