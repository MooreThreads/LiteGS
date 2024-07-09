import torch
import numpy as np
from util.BVH.PytorchBVH import BVH
from util.BVH import Object
from util.BVH.visualization import VisualizationHelper
cov=torch.Tensor(np.load("./util/BVH/test_data/cov3d1.npy")).cuda()
positions=torch.Tensor(np.load("./util/BVH/test_data/xyz1.npy")).cuda()

CHUKNNUM=20
positions_to_draw=positions[:,:3]#[:1024*CHUKNNUM,:3]
point_id=torch.arange(positions_to_draw.shape[0],device='cuda')
cov_to_draw=cov#[:1024*CHUKNNUM]

points_batch=Object.GSpointBatch(point_id,positions_to_draw,{'cov':cov_to_draw})
origin,extend=points_batch.get_AABB()

bvh=BVH([points_batch,])
bvh.build(1024)
visual=VisualizationHelper(bvh,positions_to_draw)
visual.add_points()
visual.add_AABB_box()
visual.draw()
