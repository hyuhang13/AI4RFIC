import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from collections import deque
from datetime import datetime

def check_connectivity_8(matrix, start=(9, 0), end=(9, 18)):
    """
    使用 八连通 (8-connectivity) BFS 算法检测。
    注意：默认终点已修改为右侧正中间 (9, 18)。
    """
    if matrix[start[0]][start[1]] == 0 or matrix[end[0]][end[1]] == 0:
        return False
        
    rows, cols = len(matrix), len(matrix[0])
    visited = set([start])
    queue = deque([start])
    
    # 包含四个正方向和四个对角线方向
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    
    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            return True
            
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if matrix[nr][nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False

def analyze_8conn_density(data_dir="baseline_8conn_dataset", num_files=5, matrix_size=19):
    print(f"=== 开始基于 [八连通] 分析矩阵连通性与密度分布 ===")
    start_time = datetime.now()
    
    densities_all = []
    densities_connected = []
    densities_disconnected = []
    total_elements = matrix_size * matrix_size

    for file_idx in range(1, num_files + 1):
        file_path = os.path.join(data_dir, f"matrices_part{file_idx}.txt")
        if not os.path.exists(file_path):
            continue
            
        print(f"正在扫描文件: {os.path.basename(file_path)}...")
        current_matrix = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if not line.startswith('#'):
                    row_data = [int(x) for x in line.split(',')]
                    current_matrix.append(row_data)
                elif line.startswith('# Matrix Index:'):
                    ones_count = sum(row.count(1) for row in current_matrix)
                    density = (ones_count / total_elements) * 100
                    densities_all.append(density)
                    
                    # 传入更新后的终点坐标 (9, 18)
                    is_connected = check_connectivity_8(current_matrix, start=(9, 0), end=(9, 18))
                    
                    if is_connected:
                        densities_connected.append(density)
                    else:
                        densities_disconnected.append(density)
                        
                    current_matrix = []

    total_count = len(densities_all)
    conn_count = len(densities_connected)
    
    if total_count == 0:
        print("未找到有效数据！")
        return

    print(f"\n================ 数据扫描完毕 ================")
    print(f"总计检查矩阵数: {total_count}")
    print(f"八连通样本数量: {conn_count} (占比 {conn_count/total_count*100:.2f}%)")
    print(f"断路样本数量:   {total_count - conn_count} (占比 {(total_count - conn_count)/total_count*100:.2f}%)")
    print(f"全局平均金属密度: {np.mean(densities_all):.2f}%")
    print(f"连通样本平均密度: {np.mean(densities_connected) if conn_count else 0:.2f}%")

    # ================= 绘制频度曲线图 =================
    plt.figure(figsize=(12, 7))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    bins = np.linspace(35, 65, 60)
    
    if densities_connected:
        plt.hist(densities_connected, bins=bins, alpha=0.6, color='royalblue', 
                 label=f'Connected (8-Conn): {conn_count/total_count*100:.1f}%', density=True, zorder=3)
    if densities_disconnected:
        plt.hist(densities_disconnected, bins=bins, alpha=0.5, color='crimson', 
                 label=f'Disconnected: {(total_count-conn_count)/total_count*100:.1f}%', density=True, zorder=3)
    
    kde_all = gaussian_kde(densities_all, bw_method=0.1)
    x_range = np.linspace(30, 70, 500)
    plt.plot(x_range, kde_all(x_range), color='black', linewidth=2.5, 
             label='Overall Density PDF (p=0.5)', zorder=4)

    plt.title('Metal Density Distribution at p=0.5 (8-Connectivity, Ports Forced)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Metal Density (%)', fontsize=14)
    plt.ylabel('Frequency (Normalized)', fontsize=14)
    plt.xlim(35, 65)
    
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    output_img = "8conn_density_ports_forced.png"
    plt.savefig(output_img, dpi=300)
    print(f"[✓] 频度曲线图已保存为: {os.path.abspath(output_img)}")
    plt.show()

if __name__ == "__main__":
    analyze_8conn_density()