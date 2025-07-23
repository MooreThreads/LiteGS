import os
from argparse import ArgumentParser

datasets_path={
    "mipnerf360_indoor":"./dataset/mipnerf360",
    "mipnerf360_outdoor":"./dataset/mipnerf360",
    "tanksandtemples":"./dataset/tandt",
    "deepblending":"./dataset/db",
}

output_path="./ablation_output/densify/"

target_primitives = {
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
    "playroom": 1000000,
    "drjohnson": 1000000
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


adc_config=" --densification_interval 1 --opacity_reset_interval 20 --opacity_reset_mode reset --prune_mode threshold"

ablation_configs_list=[
    "",
    adc_config
]

for ablation_configs in ablation_configs_list:
    print("------------{}---------------".format(ablation_configs))
    for dataset_name,scenes in datasets.items():
        for scene_name in scenes:
            scene_input_path=os.path.join(datasets_path[dataset_name],scene_name)
            scene_output_path=os.path.join(output_path,scene_name)
            print("scene:{} #primitive:{}".format(scene_name,target_primitives[scene_name]))
            os.system("python example_train.py -s {0} -m {1} --eval --sh_degree 3 --target_primitives {2} {3} {4}".format(
                    scene_input_path,
                    scene_output_path,
                    target_primitives[scene_name],
                    img_config[dataset_name],
                    ablation_configs
                ))
            

    for dataset_name,scenes in datasets.items():
        for scene_name in scenes:
            scene_input_path=os.path.join(datasets_path[dataset_name],scene_name)
            scene_output_path=os.path.join(output_path,scene_name)
            os.system("python example_metrics.py -s {0} -m {1} --sh_degree 3 {2} ".format(scene_input_path,scene_output_path,img_config[dataset_name]))