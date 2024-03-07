import torch
import math

def world_to_ndc(position,view_project_matrix):
    hom_pos=torch.matmul(position,view_project_matrix)
    ndc_pos=hom_pos/(hom_pos[...,3:4]+1e-6)
    return ndc_pos

def world_to_view(position,view_matrix):
    return position@view_matrix

