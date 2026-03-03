import pandas as pd


# ==========================================
# 1. 读取并预处理数据
# ==========================================
df = pd.read_csv('./output/litegs_results.csv')
df = df.groupby(['scene', 'primitives']).mean().reset_index()
if 'repeat_i' in df.columns:
    df = df.drop(columns=['repeat_i'])

big_conf = {
    "bicycle": 2000000,#54275
    "flowers": 1000000,#38347
    "garden": 2000000,#138766
    "stump": 1000000,#32049
    "treehill": 800000,#52363
    "room": 800000,#112627
    "counter": 600000,#155767
    "kitchen": 1000000,#241367
    "bonsai": 800000,#206613
    "truck": 600000,#136029
    "train": 600000,#182686
    "drjohnson": 800000,#80861
    "playroom": 500000#37005
}

datasets={
    "mipnerf360":["bicycle", "flowers", "garden", "stump", "treehill","room", "counter", "kitchen", "bonsai"],
    "tat":["truck", "train"],
    "db":["drjohnson", "playroom"],
}

scene_results=pd.DataFrame(columns=['scene', 'primitives', 'time', "SSIM_test","PSNR_test","LPIPS_test"])
for scene,conf in big_conf.items():
    results=df[(df['scene'] == scene) & (df['primitives'] == conf)]
    scene_results= pd.concat([scene_results, results], ignore_index=True)

dataset_results=pd.DataFrame(columns=['dataset', 'primitives', 'time', "SSIM_test","PSNR_test","LPIPS_test"])
for dataset,scenes in datasets.items():
    results=scene_results[scene_results['scene'].isin(scenes)]
    results=results.drop(columns=['scene'])
    results=results.mean()
    print(f"Dataset: {dataset}")
    print(results)
