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
import subprocess
import re
import csv

scene_primitive = {
    "bicycle": 1000000,
    "flowers": 1000000,
    "garden": 1000000,
    "stump": 1000000,
    "treehill": 1000000,
    "room": 1000000,
    "counter": 1000000,
    "kitchen": 1000000,
    "bonsai": 1000000,
    "truck": 1000000,
    "train": 1000000,
    "drjohnson": 1000000,
    "playroom": 1000000
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

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--output_path", default="./output")
parser.add_argument("--repeat", default=3, type=int)
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

img_folder={
    "mipnerf360_indoor":"images_4",
    "mipnerf360_outdoor":"images_2",
    "tanksandtemples":"images",
    "deepblending":"images",
}

training_args_tempalte="-s {0} -m {1} --dataset.eval --densify.target_primitives {2} -i {3} --config-path ./litegs/config/aggressive.json"
eval_args_template="--config-path {0}"
take_time_pattern = r"takes:\s*([+-]?\d+(?:\.\d+)?)"
eval_pattern = r"(SSIM|PSNR|LPIPS)\s*:\s*([+-]?\d+(?:\.\d+)?)"
csv_header=["scene","primitives","repeat_i","time","SSIM_train","PSNR_train","LPIPS_train","SSIM_test","PSNR_test","LPIPS_test"]
csv_file=open(os.path.join(args.output_path,"litegs_aggressive_results.csv"), 'w', newline="")
result_csv_writer=csv.writer(csv_file)
result_csv_writer.writerow(csv_header)


if not args.skip_training:
    for dataset,scenes in datasets.items():
        for scene_name in scenes:
            for i in range(args.repeat):
                
                target_primitives=scene_primitive[scene_name]
                scene_input_path=os.path.join(args.__getattribute__(dataset.split('_')[0]),scene_name)
                scene_output_path=os.path.join(args.output_path,scene_name+'-aggressive-{}k-{}'.format(int(target_primitives/1000), i))
                print("------------ scene:{} #primitive:{} ------------".format(scene_name,target_primitives))

                #training
                training_args=training_args_tempalte.format(scene_input_path,scene_output_path,target_primitives,img_folder[dataset])
                process = subprocess.Popen(["python","example_train.py"]+training_args.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()
                match = re.search(take_time_pattern, stdout)
                if match:
                    training_time=float(match.group(1))
                    print("takes: {} s".format(training_time))
                else:
                    print("Failed to extract training time from output: {}".format(stdout))
                    print(stderr)
                
                #evaluation
                if not os.path.exists(scene_output_path): 
                    print("Output path {} does not exist, skipping evaluation for scene {}, primitives {}, repeat {}".format(scene_output_path, scene_name, target_primitives, i))
                    continue
                eval_args=eval_args_template.format(os.path.join(scene_output_path,"finish","eval_config.json"))
                process = subprocess.Popen(["python","example_metrics.py"]+eval_args.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()
                matches = re.findall(eval_pattern, stdout)
                if len(matches)==6:
                    SSIM_train=float(matches[0][1])
                    PSNR_train=float(matches[1][1])
                    LPIPS_train=float(matches[2][1])
                    SSIM_test=float(matches[3][1])
                    PSNR_test=float(matches[4][1])
                    LPIPS_test=float(matches[5][1])
                    result_csv_writer.writerow([scene_name,target_primitives,i,training_time,SSIM_train,PSNR_train,LPIPS_train,SSIM_test,PSNR_test,LPIPS_test])
                    csv_file.flush()
                    print("SSIM_train: {}\n PSNR_train: {}\n LPIPS_train: {}\n SSIM_test: {}\n PSNR_test: {}\n LPIPS_test: {}".format(SSIM_train, PSNR_train, LPIPS_train, SSIM_test, PSNR_test, LPIPS_test))