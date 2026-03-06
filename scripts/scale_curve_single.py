import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# ==========================================
# 1. 读取并预处理数据
# ==========================================
df = pd.read_csv('./output/litegs_results_freeze.csv')
df = df.groupby(['scene', 'primitives']).mean().reset_index()
if 'repeat_i' in df.columns:
    df = df.drop(columns=['repeat_i'])
df['method'] = 'LiteGS'

others = pd.read_csv('./output/others_work_freeze.csv')
others['primitives'] = others['primitives'] * 1000
df = pd.concat([df, others], ignore_index=True)

# 【核心修改】：仅过滤出 garden 场景的数据
target_scene = 'garden'
scene_data = df[df['scene'] == target_scene].copy()
methods = scene_data['method'].unique()

# ==========================================
# 2. 学术级配色与样式字典
# ==========================================
style_dict = {
    'LiteGS':           {'color': '#E63946', 'marker': 'o', 'ls': '-',  'lw': 2.0, 'zorder': 10, 'ms': 5},
    '3DGS':             {'color': '#1D3557', 'marker': 's', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    'Mini-Splatting':   {'color': '#457B9D', 'marker': '^', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    'Mini-Splatting2':  {'color': '#2A9D8F', 'marker': 'D', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    'Taming-3DGS':      {'color': '#F4A261', 'marker': 'v', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    'FastGS':           {'color': '#E9C46A', 'marker': 'P', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    '3DGS-MCMC':        {'color': '#9C6644', 'marker': 'X', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
}

# ==========================================
# 3. 设置单图布局
# ==========================================
# 【核心修改】：为单图设置合适的 figsize，兼顾紧凑与清晰
fig = plt.figure(figsize=(6, 5)) 

legend_handles = []
legend_labels = []

# 直接使用一个 2行1列 的 GridSpec
inner_gs = gridspec.GridSpec(2, 1, hspace=0.08)

ax_psnr = fig.add_subplot(inner_gs[0])
ax_time = fig.add_subplot(inner_gs[1], sharex=ax_psnr)

outlier_methods = ['3DGS-MCMC', '3DGS']

# ==========================================
# 4. 绘图逻辑
# ==========================================
normal_data = scene_data[~scene_data['method'].isin(outlier_methods)]
if not normal_data.empty:
    max_normal_time = normal_data['time'].astype(float).max()
    y_time_limit = max_normal_time * 1.25 
else:
    y_time_limit = None

for method in methods:
    method_data = scene_data[scene_data['method'] == method].sort_values('primitives')
    if method_data.empty:
        continue
        
    style = style_dict.get(method, {'color': 'gray', 'marker': 'o', 'ls': '--', 'lw': 1.5, 'zorder': 1, 'ms': 4})
    
    x = method_data['primitives'] / 1000  
    y_psnr = method_data['PSNR_test']
    y_time_raw = method_data['time'].astype(float)
    
    y_time_plot = y_time_raw.copy()
    is_outlier = False
    
    if y_time_limit:
        y_time_plot = y_time_plot.apply(lambda val: y_time_limit if val > y_time_limit else val)
        is_outlier = (y_time_raw > y_time_limit).any()

    current_ls = style['ls'] if len(x) > 1 else ''
    
    line_psnr, = ax_psnr.plot(x, y_psnr, color=style['color'], marker=style['marker'], 
                              linestyle=current_ls, linewidth=style['lw'], 
                              markersize=style['ms'], zorder=style['zorder'])
    
    line_time, = ax_time.plot(x, y_time_plot, color=style['color'], marker=style['marker'], 
                              linestyle=current_ls, linewidth=style['lw'], 
                              markersize=style['ms'], zorder=style['zorder'])
    
    # 异常值悬浮文本提示
    if is_outlier:
        for px, p_raw, p_plot in zip(x, y_time_raw, y_time_plot):
            if p_raw > y_time_limit:
                # 针对单图优化文本位置，防止重叠
                ha_align = 'center'
                text_str = f'{int(p_raw)}s'
                ax_time.text(px, p_plot * 0.96, text_str, 
                             color=style['color'], fontsize=10, ha=ha_align, va='top',
                             fontweight='bold', zorder=20)

    legend_handles.append(line_psnr)
    legend_labels.append(method)

if y_time_limit:
    ax_time.set_ylim(bottom=0, top=y_time_limit)

# ==========================================
# 5. 标签与刻度美化
# ==========================================
ax_psnr.set_title(target_scene.capitalize(), fontsize=14, fontweight='bold', pad=10)
ax_psnr.set_ylabel('PSNR (dB)', fontsize=11)
ax_psnr.grid(True, linestyle=':', alpha=0.7)
ax_psnr.tick_params(labelbottom=False)  
ax_psnr.tick_params(axis='both', labelsize=10)

ax_time.set_xlabel('Param Scale (k)', fontsize=11)
ax_time.set_ylabel('Time (s)', fontsize=11)
ax_time.grid(True, linestyle=':', alpha=0.7)
ax_time.tick_params(axis='both', labelsize=10)

# ==========================================
# 6. 全局图例与保存
# ==========================================
# 【核心修改】：由于只有一张图，将图例居中放置在图表正上方
fig.legend(legend_handles, legend_labels, loc='lower center', 
           bbox_to_anchor=(0.5, 0.94), ncol=4, 
           fontsize=10, frameon=True, edgecolor='black')

# 调整边界，给顶部的图例留出空间
plt.subplots_adjust(top=0.82, bottom=0.1, left=0.12, right=0.95)
plt.savefig('doc_img/Teaser_Garden_ParamScale.png', dpi=300, bbox_inches='tight')