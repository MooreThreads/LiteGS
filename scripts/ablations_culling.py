import os
from argparse import ArgumentParser

datasets_path={
    "mipnerf360_indoor":"./dataset/mipnerf360",
    "mipnerf360_outdoor":"./dataset/mipnerf360",
    "tanksandtemples":"./dataset/tandt",
    "deepblending":"./dataset/db",
}

output_path="./ablation_output/wo_culling/"

target_primitives = {
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

reg_config={
    "mipnerf360_indoor":" --reg_weight 0 ",
    "mipnerf360_outdoor":" --reg_weight 0 ",
    "tanksandtemples":" --reg_weight 0 ",
    "deepblending":" --reg_weight 0 ",
}




for dataset_name,scenes in datasets.items():
    for scene_name in scenes:
        scene_input_path=os.path.join(datasets_path[dataset_name],scene_name)
        scene_output_path=os.path.join(output_path,scene_name)
        print("scene:{} #primitive:{}".format(scene_name,target_primitives[scene_name]))
        os.system("python example_train.py -s {0} -m {1} --eval --sh_degree 3 --target_primitives {2} {3} {4} --cluster_size 0".format(
                scene_input_path,
                scene_output_path,
                target_primitives[scene_name],
                img_config[dataset_name],
                reg_config[dataset_name]
            ))
        

for dataset_name,scenes in datasets.items():
    for scene_name in scenes:
        scene_input_path=os.path.join(datasets_path[dataset_name],scene_name)
        scene_output_path=os.path.join(output_path,scene_name)
        os.system("python example_metrics.py -s {0} -m {1} --sh_degree 3 {2} ".format(scene_input_path,scene_output_path,img_config[dataset_name]))