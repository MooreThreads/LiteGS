import torch
import numpy as np
from util.BVH.PytorchBVH import BVH
from util.BVH import Object
from util.BVH.visualization import VisualizationHelper
cov=torch.Tensor(np.load("./util/BVH/test_data/cov3d.npy")).cuda()
positions=torch.Tensor(np.load("./util/BVH/test_data/xyz.npy")).cuda()

CHUKNNUM=10
positions_to_draw=positions[:1024*CHUKNNUM,:3]
point_id=torch.arange(positions_to_draw.shape[0],device='cuda')
cov_to_draw=cov[:1024*CHUKNNUM]

points_batch=Object.GSpointBatch(point_id,positions_to_draw,cov_to_draw)
origin,extend=points_batch.get_AABB()

bvh=BVH([points_batch,])
bvh.build(1024)
visual=VisualizationHelper(bvh,positions_to_draw)
visual.add_points()
visual.add_AABB_box()
visual.draw()
