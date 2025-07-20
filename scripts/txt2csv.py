import re
import csv
from collections import defaultdict

def parse_metrics(metrics_text):
    """解析场景指标数据"""
    pattern = r"Scene:.*[\\/](.+?)-(\d+k)\s+(Trainingset|Testset)\s+" \
              r"SSIM\s*:\s*([\d.]+)\s+" \
              r"PSNR\s*:\s*([\d.]+)\s+" \
              r"LPIPS\s*:\s*([\d.]+)"
    
    data_dict = defaultdict(lambda: {'train': {}, 'test': {}})
    
    for match in re.finditer(pattern, metrics_text):
        scene = match.group(1)
        param_scale = match.group(2)
        dataset_type = 'train' if 'Train' in match.group(3) else 'test'
        
        metrics = {
            'SSIM': float(match.group(4)),
            'PSNR': float(match.group(5)),
            'LPIPS': float(match.group(6))
        }
        data_dict[(scene, param_scale)][dataset_type] = metrics
    
    return data_dict

def parse_training_time(time_text):
    """解析训练耗时数据"""
    time_data = {}
    pattern = r".*[\\/](.+?)-(\d+k)\s+takes:\s*([\d.]+)"
    
    for match in re.finditer(pattern, time_text):
        scene = match.group(1)
        param_scale = match.group(2)
        training_time = float(match.group(3))
        time_data[(scene, param_scale)] = training_time
    
    return time_data

def combine_data(metrics_data, time_data):
    """合并指标和耗时数据"""
    csv_data = []
    
    for (scene, scale), datasets in metrics_data.items():
        # 获取此场景的训练耗时
        training_time = time_data.get((scene, scale), "")
        
        row = {
            'Scene': scene,
            'ParamScale': scale,
            'Training_Time': training_time,
            'Train_SSIM': datasets['train'].get('SSIM', ''),
            'Train_PSNR': datasets['train'].get('PSNR', ''),
            'Train_LPIPS': datasets['train'].get('LPIPS', ''),
            'Test_SSIM': datasets['test'].get('SSIM', ''),
            'Test_PSNR': datasets['test'].get('PSNR', ''),
            'Test_LPIPS': datasets['test'].get('LPIPS', '')
        }
        csv_data.append(row)
    
    return csv_data

def save_to_csv(data, filename):
    """保存为CSV文件"""
    if not data:
        return False
    
    fieldnames = data[0].keys()
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    return True

# 主程序
if __name__ == "__main__":
    # 读取指标文件
    with open('./output.txt', 'r') as file:
        metrics_text = file.read()
    
    # 读取耗时文件
    with open('./takes_time.txt', 'r') as file:
        time_text = file.read()
    
    # 解析数据
    metrics_data = parse_metrics(metrics_text)
    time_data = parse_training_time(time_text)
    
    # 合并数据
    combined_data = combine_data(metrics_data, time_data)
    
    # 保存CSV
    if save_to_csv(combined_data, './results.csv'):
        print(f"成功生成CSV文件，包含{len(combined_data)}个场景的数据")
    else:
        print("未解析到有效数据，请检查输入格式")