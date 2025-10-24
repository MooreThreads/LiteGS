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
import numpy as np


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="./output")
parser.add_argument('--source_path', required=True, type=str)
args, _ = parser.parse_known_args()


training_config="--sh_degree 3 --source_type slam -s {0} -m {1} --target_primitives 1000000 --iterations 4500 --position_lr_max_steps 4500 --position_lr_final 0.000016 --densification_interval 2 --learnable_viewproj"

metrics_config="--sh_degree 3 --source_type slam -s {0} -m {1} --learnable_viewproj"

scenes = os.listdir(args.source_path)

for scene in scenes:
    os.system("python example_train.py "+training_config.format(os.path.join(args.source_path,scene),os.path.join(args.output_path,scene)))

results=[]
for scene in scenes:
    process = subprocess.Popen(["python","example_metrics.py"]+metrics_config.format(os.path.join(args.source_path,scene),os.path.join(args.output_path,scene)).split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    print(stderr)
    print(stdout)

    index=stdout.find('  PSNR : ')
    if index!=-1:
        end=stdout[index+9:].find('\n')
        psnr=float(stdout[index+9:index+9+end])
        results.append(psnr)
print('Finish')
print('PSNR avg:',np.array(results).mean())