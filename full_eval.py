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

# Scene-specific budgets for "big" mode (final_count)
target_primitives_list = {
    "bicycle": [i for i in range(500_000,5_000_000+1,500_000)],
    "flowers": [i for i in range(500_000,3_000_000+1,500_000)],
    "garden": [i for i in range(500_000,5_000_000+1,500_000)],
    "stump":[i for i in range(200_000,2_000_000+1,200_000)],
    "treehill":[i for i in range(200_000,2_000_000+1,200_000)],
    "room": [i for i in range(200_000,1_000_000+1,200_000)],
    "counter": [i for i in range(200_000,1_000_000+1,200_000)],
    "kitchen": [i for i in range(300_000,1_000_000+1,100_000)],
    "bonsai": [i for i in range(300_000,1_000_000+1,100_000)],
    "truck": [i for i in range(200_000,2_000_000+1,200_000)],
    "train": [i for i in range(200_000,1_000_000+1,200_000)],
    "playroom": [i for i in range(100_000,1_000_000+1,100_000)],
    "drjohnson": [i for i in range(100_000,1_000_000+1,100_000)]
}

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
    "bicycle": 15,#54275
    "flowers": 15,#38347
    "garden": 15,#138766
    "stump": 15,#32049
    "treehill": 15,#52363
    "room": 2,#112627
    "counter": 2,#155767
    "kitchen": 2,#241367
    "bonsai": 2,#206613
    "truck": 2,#136029
    "train": 2,#182686
    "playroom": 5,#37005
    "drjohnson": 5#80861
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
args, _ = parser.parse_known_args()


datasets={
    "mipnerf360_indoor":["bicycle", "flowers", "garden", "stump", "treehill"],
    "mipnerf360_outdoor":["room", "counter", "kitchen", "bonsai"],
    "tanksandtemples":["truck", "train"],
    "deepblending":["drjohnson", "playroom"],
}

img_config={
    "mipnerf360_indoor":" -i images_4",
    "mipnerf360_outdoor":" -i images_2",
    "tanksandtemples":" -i images",
    "deepblending":" -i images",
}

reg_config={
    "mipnerf360_indoor":" --reg_weight 0 ",
    "mipnerf360_outdoor":" --reg_weight 0 ",
    "tanksandtemples":" --reg_weight 0 ",
    "deepblending":" --reg_weight 0 ",
}



if not args.skip_training:
    for dataset,scenes in datasets.items():
        for scene_name in scenes:
            scene_input_path=os.path.join(args.__getattribute__(dataset.split('_')[0]),scene_name)
            #curve
            for target_primitives in target_primitives_list[scene_name]:
                scene_output_path=os.path.join(args.output_path,scene_name+'-{}k'.format(int(target_primitives/1000)))
                print("scene:{} #primitive:{}".format(scene_name,target_primitives))
                os.system("python example_train.py -s {0} -m {1} --eval --sh_degree 3 --target_primitives {2} {3} {4}".format(
                    scene_input_path,
                    scene_output_path,
                    target_primitives,
                    img_config[dataset],
                    reg_config[dataset]
                ))
            #full
            scene_output_path=os.path.join(args.output_path,scene_name+'-big')
            os.system("python example_train.py -s {0} -m {1} --eval --sh_degree 3 --target_primitives {2} {3} {4}".format(
                    scene_input_path,
                    scene_output_path,
                    target_primitives,
                    img_config[dataset],
                    reg_config[dataset]
                ))

for dataset,scenes in datasets.items():
    for scene_name in scenes:
        scene_input_path=os.path.join(args.__getattribute__(dataset.split('_')[0]),scene_name)
        for target_primitives in target_primitives_list[scene_name]:
            scene_output_path=os.path.join(args.output_path,scene_name+'-{}k'.format(int(target_primitives/1000)))
            os.system("python example_metrics.py -s {0} -m {1} --sh_degree 3 {2}  >> output.txt".format(scene_input_path,scene_output_path,img_config[dataset]))
        scene_output_path=os.path.join(args.output_path,scene_name+'-big')
        os.system("python example_metrics.py -s {0} -m {1} --sh_degree 3 {2}  >> output.txt".format(scene_input_path,scene_output_path,img_config[dataset]))