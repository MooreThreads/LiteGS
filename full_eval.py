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

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

# Scene-specific budgets for "big" mode (final_count)
big_budgets = {
    "bicycle": 5987095,
    "flowers": 3618411,
    "garden": 5728191,
    "stump": 4867429,
    "treehill": 3770257,
    "room": 1548960,
    "counter": 1190919,
    "kitchen": 1803735,
    "bonsai": 1252367,
    "truck": 2584171,
    "train": 1085480,
    "playroom": 2326100,
    "drjohnson": 3273600
}

# Scene-specific budgets for "budget" mode (multiplier)
budget_multipliers = {
    "bicycle": 15,
    "flowers": 15,
    "garden": 15,
    "stump": 15,
    "treehill": 15,
    "room": 2,
    "counter": 2,
    "kitchen": 2,
    "bonsai": 2,
    "truck": 2,
    "train": 2,
    "playroom": 5,
    "drjohnson": 5
}

images={
    "bicycle": "images_4",
    "flowers":  "images_4",
    "garden":  "images_4",
    "stump":  "images_4",
    "treehill": "images_4",
    "room": "images_2",
    "counter": "images_2",
    "kitchen": "images_2",
    "bonsai": "images_2",
    "truck": "images",
    "train": "images",
    "playroom": "images",
    "drjohnson": "images",
}

budget_dict={
    "big":big_budgets,
    "budget":budget_multipliers
}

densify_mode_dict={
    "big":"final_count",
    "budget":"multiplier"
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--output_path", default="./output")
parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
parser.add_argument("--deepblending", "-db", required=True, type=str)
parser.add_argument("--mode", required=True, type=str)
args, _ = parser.parse_known_args()

datasets={
    "mipnerf360":mipnerf360_outdoor_scenes+mipnerf360_indoor_scenes,
    "tanksandtemples":tanks_and_temples_scenes,
    "deepblending":deep_blending_scenes
}



if not args.skip_training:
    for dataset,scenes in datasets.items():
        for scene_name in scenes:
            scene_input_path=os.path.join(args.__getattribute__(dataset),scene_name)
            scene_output_path=os.path.join(args.output_path,scene_name)
            os.system("python example_train.py -s {0} -i {1} -m {2} --eval --sh_degree 3 --budget {3} --densify_mode {4}".format(
                scene_input_path,
                images[scene_name],
                scene_output_path,
                budget_dict[args.mode][scene_name],
                densify_mode_dict[args.mode]
            ))

for dataset,scenes in datasets.items():
    for scene_name in scenes:
        scene_input_path=os.path.join(args.__getattribute__(dataset),scene_name)
        scene_output_path=os.path.join(args.output_path,scene_name)
        os.system("python example_metrics.py -s {0} -i {1} -m {2} --sh_degree 3".format(
            scene_input_path,
            images[scene_name],
            scene_output_path
        ))