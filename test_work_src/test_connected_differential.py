import pandas as pd
import matplotlib.pyplot as plt

def plot_sparameter_comparison(file_connected, file_unconnected):
    # 1. 读取数据
    # 由于第一列(ID)没有表头，pandas通常会自动将其设为索引 index_col=0
    try:
        df_conn = pd.read_csv(file_connected, index_col=0)
        df_unconn = pd.read_csv(file_unconnected, index_col=0)
    except FileNotFoundError:
        print("未找到文件，请确保 CSV 文件与此脚本在同一目录下。")
        return

    # 检查 'Freq(Hz)' 是否被正确识别为列，如果没有，重新设置索引
    if 'Freq(Hz)' not in df_conn.columns and df_conn.index.name == 'Freq(Hz)':
        df_conn.reset_index(inplace=True)
        df_unconn.reset_index(inplace=True)

    freq_col = 'Freq(Hz)'
    
    # 定义要绘制的S参数及其虚实部
    s_params = ['S11', 'S21', 'S12', 'S22']
    parts = ['real', 'imag']
    
    # 2. 创建 4行(S11~S22) x 2列(实部、虚部) 的画布
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 20), sharex=True)
    fig.suptitle('S-Parameter Comparison: Connected vs Unconnected', fontsize=20, y=0.92)

    # 3. 循环绘制每个子图
    for i, s in enumerate(s_params):
        for j, part in enumerate(parts):
            col_name = f"{s}_{part}"
            ax = axes[i, j]
            
            # 检查列名是否存在，以防大小写或空格问题
            if col_name in df_conn.columns and col_name in df_unconn.columns:
                # 绘制 connected 曲线 (实线)
                ax.plot(df_conn[freq_col], df_conn[col_name], 
                        label='Connected', color='blue', linestyle='-', linewidth=2)
                # 绘制 unconnected 曲线 (虚线)
                ax.plot(df_unconn[freq_col], df_unconn[col_name], 
                        label='Unconnected', color='red', linestyle='--', linewidth=2)
                
                # 设置标题与标签
                ax.set_title(f"{s} {part.capitalize()}", fontsize=14)
                ax.set_ylabel('Amplitude', fontsize=12)
                ax.grid(True, linestyle=':', alpha=0.7)
                ax.legend(loc='best')
            else:
                ax.text(0.5, 0.5, f"Column {col_name} not found", 
                        ha='center', va='center', transform=ax.transAxes)
                
            # 为最后一行添加横轴标签
            if i == 3:
                ax.set_xlabel('Frequency (Hz)', fontsize=12)

    # 自动调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # 4. 保存或显示图像
    output_filename = 'sparameters_comparison.png'
    plt.savefig(output_filename, dpi=300)
    # plt.show()
    print(f"对比图已生成并保存为: {output_filename}")

# ====== 执行代码 ======
# 确保你的文件名为 'dataset_connected.csv' 和 'dataset_unconnected.csv'
plot_sparameter_comparison('dataset_connected.csv', 'dataset_unconnected.csv')