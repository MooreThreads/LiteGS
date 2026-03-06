import pandas as pd

# ==========================================
# 1. 读取并预处理数据
# ==========================================
df = pd.read_csv('./output/litegs_aggressive_results_4090_freeze.csv')
df = df.groupby(['scene', 'primitives']).mean().reset_index()
if 'repeat_i' in df.columns:
    df = df.drop(columns=['repeat_i'])

datasets = {
    "mipnerf360": ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"],
    "tat": ["truck", "train"],
    "db": ["drjohnson", "playroom"],
}


# ==========================================
# 4. 按数据集计算均值 (Dataset-level Aggregation)
# ==========================================
dataset_rows = []
for dataset_name, scenes in datasets.items():
    # 筛选出属于当前数据集的场景
    ds_data = df[df['scene'].isin(scenes)]
    
    if not ds_data.empty:
        # 计算该数据集下的各项指标均值
        mean_vals = ds_data[['primitives', 'time', 'SSIM_test', 'PSNR_test', 'LPIPS_test']].mean()
        
        dataset_rows.append({
            'dataset': dataset_name,
            'primitives': mean_vals['primitives'],
            'time': mean_vals['time'],
            'PSNR_test': mean_vals['PSNR_test'],
            'SSIM_test': mean_vals['SSIM_test'],
            'LPIPS_test': mean_vals['LPIPS_test']
        })

dataset_results = pd.DataFrame(dataset_rows)

# ==========================================
# 5. 格式化输出 (方便填入 LaTeX 表格)
# ==========================================
print("\n" + "="*50)
print(" 📊 Dataset Averages (Ready for LaTeX Table) ")
print("="*50)

# 对输出进行四舍五入格式化，PSNR保留2位小数，SSIM/LPIPS保留3位小数，Time/Primitives转为整数
for index, row in dataset_results.iterrows():
    print(f"Dataset: {row['dataset'].upper()}")
    print(f"  - Primitives (avg) : {int(row['primitives']):,}")
    print(f"  - Time (avg)       : {int(row['time'])} s")
    print(f"  - PSNR             : {row['PSNR_test']:.2f}")
    print(f"  - SSIM             : {row['SSIM_test']:.3f}")
    print(f"  - LPIPS            : {row['LPIPS_test']:.3f}")
    print("-" * 30)
