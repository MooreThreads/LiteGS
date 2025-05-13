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

import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "garden", "stump"]#["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--output_path", default="./output")
parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
parser.add_argument('--colmap_subfolder', default=".", type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)

if not args.skip_training:
    for scene in all_scenes:
        scene_input_path=os.path.join(args.mipnerf360,scene,args.colmap_subfolder)
        scene_output_path=os.path.join(args.output_path,scene)
        os.system("python example_train.py -s " + scene_input_path + " -i images -m " + scene_output_path + " --eval --sh_degree 3")

for scene in all_scenes:
    scene_input_path=os.path.join(args.mipnerf360,scene,args.colmap_subfolder)
    scene_output_path=os.path.join(args.output_path,scene)
    os.system("python example_metrics.py -s " + scene_input_path + " -i images -m " + scene_output_path + " --sh_degree 3")