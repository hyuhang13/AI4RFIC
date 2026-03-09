import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from datetime import datetime

def analyze_and_plot_density(data_dir="beta_filtered_dataset", num_files=5, matrix_size=19):
    print(f"=== 开始分析 Beta 过滤数据集的金属密度分布 ===")
    print(f"读取目录: {os.path.abspath(data_dir)}")
    
    start_time = datetime.now()
    
    densities_all = []
    densities_connected = []
    densities_disconnected = []
    
    total_elements = matrix_size * matrix_size

    # 1. 快速读取所有矩阵并计算密度
    for file_idx in range(1, num_files + 1):
        file_path = os.path.join(data_dir, f"matrices_part{file_idx}.txt")
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}")
            continue
            
        print(f"正在扫描文件: {os.path.basename(file_path)}...")
        
        current_matrix_ones = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if not line.startswith('#'):
                    # 统计当前行的 1 的个数
                    current_matrix_ones += line.count('1')
                elif line.startswith('# Matrix Index:'):
                    # 遇到标签行，计算刚读完的矩阵密度
                    density = (current_matrix_ones / total_elements) * 100
                    densities_all.append(density)
                    
                    # 针对新格式进行解析：# Matrix Index: X | Task: XX | Connected: True | P(1): 0.45...
                    if "Connected: True" in line:
                        densities_connected.append(density)
                    else:
                        densities_disconnected.append(density)
                        
                    # 重置计数器，准备读取下一个矩阵
                    current_matrix_ones = 0

    print(f"\n数据读取完毕，耗时: {(datetime.now() - start_time).total_seconds():.1f} 秒")
    print(f"共统计 {len(densities_all)} 个矩阵样本。")
    print(f"其中连通样本: {len(densities_connected)} 个，断路样本: {len(densities_disconnected)} 个。")

    if not densities_all:
        print("未找到有效数据，请检查数据集路径！")
        return

    # ================= 绘制高分辨率频度曲线图 =================
    print("\n正在生成高分辨率频度分布曲线...")
    plt.figure(figsize=(12, 7))
    
    # 设置网格背景
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # 设定直方图的区间范围 (0% 到 100%，步长1%)
    bins = np.linspace(0, 100, 100)
    
    # 绘制连通样本和断路样本的直方图 (归一化 density=True)
    plt.hist(densities_connected, bins=bins, alpha=0.6, color='royalblue', 
             label='Connected Samples (Target Passives from Filtered)', density=True, zorder=3)
    plt.hist(densities_disconnected, bins=bins, alpha=0.5, color='crimson', 
             label='Disconnected Samples (Negative Cases from Random)', density=True, zorder=3)
    
    # 使用高斯核密度平滑 (Gaussian KDE) 拟合一条整体的连续曲线
    kde_all = gaussian_kde(densities_all, bw_method=0.1)
    x_range = np.linspace(0, 100, 500)
    plt.plot(x_range, kde_all(x_range), color='black', linewidth=2.5, 
             label='Overall Density Smooth Curve', zorder=4)

    # 绘制均值指示线
    mean_density = np.mean(densities_all)
    plt.axvline(mean_density, color='green', linestyle='dashed', linewidth=2.5, 
                label=f'Overall Mean: {mean_density:.1f}%', zorder=5)
    
    # 图表装饰
    plt.title('Metal Density Distribution (Beta-Filtered Rejection Sampling)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Metal Density (%)', fontsize=14)
    plt.ylabel('Frequency (Normalized)', fontsize=14)
    plt.xlim(10, 85)  # 稍微调整了 X 轴范围，完美包裹新的密度分布
    
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    # 保存并显示图片
    output_img = "beta_filtered_density_curve.png"
    plt.savefig(output_img, dpi=300)
    print(f"[✓] 频度曲线图已完美生成并保存为: {os.path.abspath(output_img)}")
    plt.show()

if __name__ == "__main__":
    analyze_and_plot_density()