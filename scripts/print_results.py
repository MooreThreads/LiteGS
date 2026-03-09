import pandas as pd
import argparse

# ==========================================
# 1. 读取并预处理数据
# ==========================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.groupby(['scene', 'primitives']).mean().reset_index()
    if 'repeat_i' in df.columns:
        df = df.drop(columns=['repeat_i'])
    return df

datasets = {
    "mipnerf360": ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"],
    "tat": ["truck", "train"],
    "db": ["drjohnson", "playroom"],
}

# ==========================================
# 2. 输出格式函数
# ==========================================
def print_csv_format(df):
    """以逗号分隔的CSV格式输出"""
    print("\n--- CSV Format ---")
    # 打印表头
    print("scene,primitives,time,SSIM_test,PSNR_test,LPIPS_test")
    # 打印数据
    for _, row in df.iterrows():
        print(f"{row['scene']},{int(row['primitives'])},{int(row['time'])},{row['SSIM_test']:.4f},{row['PSNR_test']:.2f},{row['LPIPS_test']:.4f}")

def print_latex_format(df):
    """LaTeX表格格式输出"""
    print("\n--- LaTeX Format ---")

    # 表头
    header = "Scene & Primitives & Time(s) & SSIM & PSNR & LPIPS \\\\"
    print(header)
    print("\\hline")

    # 数据行
    for _, row in df.iterrows():
        line = f"{row['scene']} & {int(row['primitives']):,} & {int(row['time'])} & {row['SSIM_test']:.4f} & {row['PSNR_test']:.2f} & {row['LPIPS_test']:.4f} \\\\"
        print(line)

def print_markdown_format(df):
    """Markdown表格格式输出"""
    print("\n--- Markdown Format ---")

    # 表头
    print("| Scene | Primitives | Time(s) | SSIM | PSNR | LPIPS |")
    print("|-------|------------|---------|------|------|-------|")

    # 数据行
    for _, row in df.iterrows():
        print(f"| {row['scene']} | {int(row['primitives']):,} | {int(row['time'])} | {row['SSIM_test']:.4f} | {row['PSNR_test']:.2f} | {row['LPIPS_test']:.4f} |")

# ==========================================
# 3. 按场景计算均值 (Scene-level Aggregation)
# ==========================================
def get_scene_averages(df):
    """按场景分组计算平均值"""
    return df[['scene', 'primitives', 'time', 'SSIM_test', 'PSNR_test', 'LPIPS_test']].copy()

# ==========================================
# 4. 按数据集计算均值 (Dataset-level Aggregation)
# ==========================================
def get_dataset_averages(df, datasets):
    dataset_rows = []
    for dataset_name, scenes in datasets.items():
        ds_data = df[df['scene'].isin(scenes)]

        if not ds_data.empty:
            mean_vals = ds_data[['primitives', 'time', 'SSIM_test', 'PSNR_test', 'LPIPS_test']].mean()

            dataset_rows.append({
                'dataset': dataset_name,
                'primitives': mean_vals['primitives'],
                'time': mean_vals['time'],
                'PSNR_test': mean_vals['PSNR_test'],
                'SSIM_test': mean_vals['SSIM_test'],
                'LPIPS_test': mean_vals['LPIPS_test']
            })

    return pd.DataFrame(dataset_rows)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="./output/litegs_aggressive_results_freeze.csv", help="CSV file path")
    args = parser.parse_args()

    df = load_data(args.csv)

    # 场景级别平均数据
    scene_df = get_scene_averages(df)

    # 数据集级别平均数据
    dataset_df = get_dataset_averages(df, datasets)

    # 输出场景级别数据（三种格式）
    print("=" * 60)
    print(" Scene-Level Averages")
    print("=" * 60)
    print_csv_format(scene_df)
    print_latex_format(scene_df)
    print_markdown_format(scene_df)

    # 输出数据集级别数据（三种格式）
    print("\n" + "=" * 60)
    print(" Dataset-Level Averages")
    print("=" * 60)

    # 重命名列以便格式输出
    dataset_df = dataset_df.rename(columns={'dataset': 'scene'})
    print_csv_format(dataset_df)
    print_latex_format(dataset_df)
    print_markdown_format(dataset_df)
