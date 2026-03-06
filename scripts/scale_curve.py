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

scenes = sorted(df['scene'].unique())
methods = df['method'].unique()

# ==========================================
# 2. 学术级配色与样式字典
# ==========================================
style_dict = {
    'LiteGS':           {'color': '#E63946', 'marker': 'o', 'ls': '-',  'lw': 2.0, 'zorder': 10, 'ms': 4},
    '3DGS':             {'color': '#1D3557', 'marker': 's', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    'Mini-Splatting':   {'color': '#457B9D', 'marker': '^', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    'Mini-Splatting2':  {'color': '#2A9D8F', 'marker': 'D', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    'Taming-3DGS':      {'color': '#F4A261', 'marker': 'v', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    'FastGS':           {'color': '#E9C46A', 'marker': 'P', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
    '3DGS-MCMC':        {'color': '#9C6644', 'marker': 'X', 'ls': '--', 'lw': 1.5, 'zorder': 5,  'ms': 4},
}

# ==========================================
# 3. 设置布局（极限压缩纵向空间）
# ==========================================
num_scenes = len(scenes)
num_cols = 5
num_rows = int(np.ceil(num_scenes / num_cols))

# 【修改点 1】：将每行的高度从 4.5 极限压缩到 2.6，整体高度将缩减将近一半！
fig = plt.figure(figsize=(16, 2.6 * num_rows))

legend_handles = []
legend_labels = []
plotted_methods = set()

# 【修改点 2】：大幅压缩行间距(hspace)和列间距(wspace)
outer_gs = gridspec.GridSpec(num_rows, num_cols, hspace=0.35, wspace=0.22)

outlier_methods = ['3DGS-MCMC', '3DGS', 'Mini-Splatting']

# ==========================================
# 4. 遍历场景绘图
# ==========================================
for i, scene in enumerate(scenes):
    row_idx = i // num_cols
    col_idx = i % num_cols
    
    # 内部间距保持极小，让 PSNR 和 Time 紧贴
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, 
        subplot_spec=outer_gs[row_idx, col_idx], 
        hspace=0.05
    )
    
    ax_psnr = fig.add_subplot(inner_gs[0])
    ax_time = fig.add_subplot(inner_gs[1], sharex=ax_psnr)
    
    scene_data = df[df['scene'] == scene]
    
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
            
        style = style_dict.get(method, {'color': 'gray', 'marker': 'o', 'ls': '--', 'lw': 1.5, 'zorder': 1, 'ms': 5})
        
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
        
        if is_outlier:
            for px, p_raw, p_plot in zip(x, y_time_raw, y_time_plot):
                if p_raw > y_time_limit:
                    if method == '3DGS' and scene!='garden':
                        ha_align = 'right'
                        text_str = f'{int(p_raw)}s  ' 
                    elif method == '3DGS-MCMC' and scene!='garden':
                        ha_align = 'left'
                        text_str = f'  {int(p_raw)}s'
                    else:
                        ha_align = 'center'
                        text_str = f'{int(p_raw)}s'
                        
                    ax_time.text(px, p_plot * 0.96, text_str, 
                                 color=style['color'], fontsize=8, ha=ha_align, va='top',
                                 fontweight='bold', zorder=20)

        if method not in plotted_methods:
            plotted_methods.add(method)
            legend_handles.append(line_psnr)
            legend_labels.append(method)

    if y_time_limit:
        ax_time.set_ylim(bottom=0, top=y_time_limit)

    # 【修改点 3】：调小字体以适应变扁的子图
    ax_psnr.set_title(scene, fontsize=11, fontweight='bold', pad=4)
    if col_idx==0:
        ax_psnr.set_ylabel('PSNR (dB)', fontsize=9)
    ax_psnr.grid(True, linestyle=':', alpha=0.7)
    ax_psnr.tick_params(axis='both', which='major', labelsize=8)
    ax_psnr.tick_params(labelbottom=False)  
    
    ax_time.set_xlabel('Param Scale (k)', fontsize=9)
    if col_idx==0:
        ax_time.set_ylabel('Time (s)', fontsize=9)
    ax_time.grid(True, linestyle=':', alpha=0.7)
    ax_time.tick_params(axis='both', which='major', labelsize=8)

# ==========================================
# 5. 全局图例与保存
# ==========================================
# 【修改点 4】：调整图例位置与顶部留白
fig.legend(legend_handles, legend_labels, loc='lower center', 
           bbox_to_anchor=(0.5, 0.94), ncol=min(len(legend_labels), 7), 
           fontsize=10, frameon=True, edgecolor='black')

plt.subplots_adjust(top=0.90, bottom=0.08, left=0.05, right=0.97)
plt.savefig('./doc_img/LiteGS_ParamScale.png', dpi=300, bbox_inches='tight')