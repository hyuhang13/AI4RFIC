import numpy as np
import os
import random
import sys
from datetime import datetime

# 提高递归深度限制，以防 DFS 走得太深（19x19最多361步）
sys.setrecursionlimit(2000)

def generate_random_path(size, start, end):
    """
    使用带有方向偏好和随机噪声的深度优先搜索 (Biased DFS)。
    确保路径连通的同时，大幅缩短无效的绕圈，将路径密度控制在 10%~15% 左右。
    """
    visited = set()
    path = []
    
    def dfs(curr):
        visited.add(curr)
        path.append(curr)
        
        if curr == end:
            return True
        
        r, c = curr
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        valid_neighbors = [(nr, nc) for nr, nc in neighbors 
                           if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited]
        
        # 核心修复：带噪声的距离评估
        # 计算每个邻居到终点的曼哈顿距离，并加上 [-2.5, 2.5] 的随机噪声
        # 噪声让它偶尔偏离最优路线，产生“蜿蜒”的物理走线特征
        def heuristic(pos):
            dist = abs(pos[0] - end[0]) + abs(pos[1] - end[1])
            return dist + random.uniform(-2.5, 2.5)
            
        # 按照启发式分数从小到大排序，优先走分数低（离终点近）的格子
        valid_neighbors.sort(key=heuristic)
        
        for nxt in valid_neighbors:
            if dfs(nxt):
                return True
        
        path.pop()
        return False
        
    dfs(start)
    return path
def generate_continuous_binary_matrices(total_matrices=100000, num_files=5, matrix_size=19, output_dir="continuous_matrices_dataset"):
    start_time = datetime.now()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    matrices_per_file = total_matrices // num_files
    
    # 设定 PCSNIM 无源网络的端口坐标
    port_start = (9, 0)
    port_end = (9, 18)
    
    global_matrix_idx = 1
    
    print(f"=== 开始生成基于加权概率与连续微扰的矩阵数据集 ===")
    print(f"总样本数: {total_matrices} | 文件数: {num_files}")
    
    for file_idx in range(1, num_files + 1):
        output_file = os.path.join(output_dir, f"matrices_part{file_idx}.txt")
        print(f"\n>> 正在生成第 {file_idx}/{num_files} 个文件...")
        
        n_total = matrices_per_file
        # ---------------------------------------------------------
        # 极简分布设计：90% 强连通，10% 纯随机
        # ---------------------------------------------------------
        n_strong = int(n_total * 0.9)
        n_random = n_total - n_strong
        
        # 标签定义: 1 代表强连通, 0 代表纯随机
        sample_types = [1] * n_strong + [0] * n_random
        random.shuffle(sample_types)
        
        with open(output_file, 'w') as f:
            for s_type in sample_types:
                if s_type == 1:
                    is_connected_label = "True"
                    # 【核心魔法】：使用 Beta 分布生成极其平滑的右偏背景概率
                    # alpha=2.5, beta=8.0 会生成峰值在 0.17 左右，且向右拖尾的连续浮点数
                    p_one = 0#np.random.beta(2.5, 8.0)
                else:
                    is_connected_label = "False"
                    # 断路负样本：使用宽泛的均匀分布覆盖 10% 到 60% 的各种烂结构
                    p_one = np.random.uniform(0.10, 0.60)
                
                # 确保概率在安全范围内
                p_one = max(0.01, min(0.99, p_one))
                
                # 步骤一：按这个完全连续的概率生成背景矩阵
                matrix = np.random.choice([0, 1], size=(matrix_size, matrix_size), p=[1 - p_one, p_one])
                
                # 步骤二：对于强连通类型的样本，叠加曼哈顿距离 DFS 路径
                if s_type == 1:
                    path = generate_random_path(matrix_size, port_start, port_end)
                    for (r, c) in path:
                        matrix[r, c] = 1  
                
                # 步骤三：格式化写入文件
                for row in range(matrix_size):
                    row_str = ','.join(str(matrix[row, col]) for col in range(matrix_size))
                    f.write(row_str + '\n')
                
                # 记录元数据
                f.write(f'# Matrix Index: {global_matrix_idx} | Connected: {is_connected_label} | P(1): {p_one:.4f}\n')
        
                # 显示全局进度
                if global_matrix_idx % 10000 == 0:
                    progress = (global_matrix_idx / total_matrices) * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"   进度: {progress:.1f}% ({global_matrix_idx}/{total_matrices}) - 耗时: {elapsed:.1f}s")
                    
                global_matrix_idx += 1
                
        file_size = os.path.getsize(output_file)
        print(f"   [完成] 文件 {file_idx} 已保存，大小: {file_size / (1024*1024):.2f} MB")

    total_elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n全部生成完毕！总耗时: {total_elapsed:.1f} 秒")
    print(f"平均生成速度: {total_matrices / total_elapsed:.1f} 样本/秒")
    print(f"文件输出目录: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 配置参数
    TOTAL_MATRICES = 100000
    NUM_FILES = 5
    MATRIX_SIZE = 19
    OUTPUT_DIR = "continuous_matrices_dataset"
    
    generate_continuous_binary_matrices(
        total_matrices=TOTAL_MATRICES,
        num_files=NUM_FILES,
        matrix_size=MATRIX_SIZE,
        output_dir=OUTPUT_DIR
    )