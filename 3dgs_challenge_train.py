import os
from argparse import ArgumentParser
import subprocess
import json

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="./output")
parser.add_argument('--data_path', required=True, type=str)
args, _ = parser.parse_known_args()


training_config="--sh_degree 3 -s {0} -m {1} --images images_gt_downsampled --target_primitives 1000000 --iterations 5000 --position_lr_max_steps 5000 --position_lr_final 0.000016 --densification_interval 2 --eval"

scenes = os.listdir(args.data_path)
metrics={}
for scene in scenes:
    if os.path.isdir(os.path.join(args.data_path,scene))==False:
        continue
    training_arg_list=training_config.format(os.path.join(args.data_path,scene),os.path.join(args.output_path,scene)).split(' ')
    process = subprocess.Popen(["python","example_train.py"]+training_arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    print(stderr)
    print(stdout)

    label="{} takes: ".format(scene)
    index=stdout.find(label)
    if index!=-1:
        end=stdout[index+len(label):].find('\n')
        takes_time=float(stdout[index+len(label):index+len(label)+end])
        metrics[scene]={"time":takes_time}
json.dump(metrics, open(os.path.join(args.output_path, 'takes_time.json'), 'w'))

# results=[]
# for scene in scenes:
#     process = subprocess.Popen(["python","3dgs_challenge_metrics.py"]+metrics_config.format(os.path.join(args.source_path,scene),os.path.join(args.output_path,scene)).split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     stdout, stderr = process.communicate()
#     print(stderr)
#     print(stdout)

#     index=stdout.find('  PSNR : ')
#     if index!=-1:
#         end=stdout[index+9:].find('\n')
#         psnr=float(stdout[index+9:index+9+end])
#         results.append(psnr)
# print('Finish')
# print('PSNR avg:',np.array(results).mean())