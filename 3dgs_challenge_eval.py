import os
from argparse import ArgumentParser
import subprocess
import json
import numpy as np

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="./output")
parser.add_argument('--data_path', required=True, type=str)
args, _ = parser.parse_known_args()

metrics_config="--sh_degree 3 -s {0} -m {1} --images images_gt_downsampled --eval --save_image"

scenes = os.listdir(args.data_path)
metrics = json.load(open(os.path.join(args.output_path,"takes_time.json"),"r"))
psnr_list=[]
for scene in scenes:
    if os.path.isdir(os.path.join(args.data_path,scene))==False:
        continue
    process = subprocess.Popen(["python","example_metrics.py"]+metrics_config.format(os.path.join(args.data_path,scene),os.path.join(args.output_path,scene)).split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    print(stderr)
    print(stdout)

    label='  PSNR : '
    index=stdout.find('  PSNR : ')#train
    index+=len(label)
    index+=stdout[index:].find('  PSNR : ')#test
    if index!=-1:
        index+=len(label)
        end=stdout[index:].find('\n')
        psnr=float(stdout[index:index+end])
        metrics[scene]["psnr"]=psnr
        psnr_list.append(psnr)
json.dump(metrics, open(os.path.join(args.output_path, 'metrics.json'), 'w'))
print('PSNR avg:',np.array(psnr_list).mean())