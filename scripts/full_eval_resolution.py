#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#



# 3DGS-MCMC preprocess
# 1. remove flowers,treehill
# 2. --image images_4 -> --image images --resolution 4


import os
from argparse import ArgumentParser

scene_primitive = {
    "bicycle": 5987095,#54275
    "flowers": 3618411,#38347
    "garden": 5728191,#138766
    "stump": 4867429,#32049
    "treehill": 3770257,#52363
    "room": 1548960,#112627
    "counter": 1190919,#155767
    "kitchen": 1803735,#241367
    "bonsai": 1252367,#206613
    "truck": 2584171,#136029
    "train": 1085480,#182686
    "drjohnson": 3273600,#80861
    "playroom": 2326100#37005
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--output_path", default="./output")
parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
parser.add_argument("--deepblending", "-db", required=True, type=str)
args, _ = parser.parse_known_args()


datasets={
    "mipnerf360_indoor":["bicycle", "garden", "stump"],#remove flowers,treehill
    "mipnerf360_outdoor":["room", "counter", "kitchen", "bonsai"],
    "tanksandtemples":[],#["truck", "train"],
    "deepblending":[]#["drjohnson", "playroom"],
}

img_config={
    "mipnerf360_indoor":" -i images --resolution 4",
    "mipnerf360_outdoor":" -i images --resolution 2",
    "tanksandtemples":" -i images",
    "deepblending":" -i images",
}

fast_config="--iterations 18000 --position_lr_max_steps 18000 --position_lr_final 0.000016 --densification_interval 2"



if not args.skip_training:
    for dataset,scenes in datasets.items():
        for scene_name in scenes:
            scene_input_path=os.path.join(args.__getattribute__(dataset.split('_')[0]),scene_name)
            target_primitives=scene_primitive[scene_name]
            scene_output_path=os.path.join(args.output_path,scene_name+'-{}k-star'.format(int(target_primitives/1000)))
            print("scene:{} #primitive:{}".format(scene_name,target_primitives))
            os.system("python example_train.py -s {0} -m {1} --eval --sh_degree 3 --target_primitives {2} {3} {4}".format(
                    scene_input_path,
                    scene_output_path,
                    target_primitives,
                    img_config[dataset],
                    ""
                ))

for dataset,scenes in datasets.items():
    for scene_name in scenes:
        scene_input_path=os.path.join(args.__getattribute__(dataset.split('_')[0]),scene_name)
        target_primitives=scene_primitive[scene_name]
        scene_output_path=os.path.join(args.output_path,scene_name+'-{}k-star'.format(int(target_primitives/1000)))
        os.system("python example_metrics.py -s {0} -m {1} --sh_degree 3 {2}".format(scene_input_path,scene_output_path,img_config[dataset]))