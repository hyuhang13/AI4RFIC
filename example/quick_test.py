import numpy as np
import os
import random
import sys
from datetime import datetime

# 提高递归深度限制，以防 DFS 走得太深（19x19最多361步）
sys.setrecursionlimit(2000)

def generate_random_path(size, start, end):
    """
    使用随机深度优先搜索(DFS)生成一条从起点到终点的连通路径。
    模拟射频匹配网络中电感/传输线的曲折走线。
    """
    visited = set()
    path = []
    
    def dfs(curr):
        visited.add(curr)
        path.append(curr)
        
        # 到达终点，停止搜索
        if curr == end:
            return True
        
        r, c = curr
        # 允许上下左右四个方向移动
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        
        # 过滤出在边界内且未被访问过的像素
        valid_neighbors = [(nr, nc) for nr, nc in neighbors 
                           if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited]
        
        # 随机打乱邻居的遍历顺序，保证每次生成的连通路径形状不同
        random.shuffle(valid_neighbors)
        
        for nxt in valid_neighbors:
            if dfs(nxt):
                return True
        
        # 如果走入死胡同，回溯并移除当前点
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
        # 分布设计：90% 强连通干预，10% 纯随机
        # ---------------------------------------------------------
        # 1. 强连通组 (90%)
        n_strong_core = int(n_total * 0.9 * 0.4)  # 核心优质区 (Base P=0.12)
        n_strong_edge = int(n_total * 0.9 * 0.4)  # 边缘探索区 (Base P=0.28)
        n_strong_pen  = int(n_total * 0.9 * 0.2)  # 惩罚长尾区 (Base P=0.45)
        
        # 2. 纯随机断路组 (10%)
        n_random_sparse = int(n_total * 0.1 * 0.5) # 稀疏断路 (Base P=0.20)
        n_random_dense  = n_total - n_strong_core - n_strong_edge - n_strong_pen - n_random_sparse # 密集断路 (Base P=0.50)
        
        # 标签定义
        # 1: 核心连通, 2: 边缘连通, 3: 惩罚连通, 4: 稀疏断路, 5: 密集断路
        sample_types = (
            [1] * n_strong_core +
            [2] * n_strong_edge +
            [3] * n_strong_pen +
            [4] * n_random_sparse +
            [5] * n_random_dense
        )
        
        # 彻底打乱生成顺序，避免同种类型样本扎堆
        random.shuffle(sample_types)
        
        with open(output_file, 'w') as f:
            for s_type in sample_types:
                # 设定基础生成概率 (Base Probability)
                if s_type == 1:   
                    base_p = 0
                    is_connected_label = "True"
                elif s_type == 2: 
                    base_p = 0
                    is_connected_label = "True"
                elif s_type == 3: 
                    base_p = 0
                    is_connected_label = "True"
                elif s_type == 4: 
                    base_p = 0.20
                    is_connected_label = "False"
                else:             
                    base_p = 0.50
                    is_connected_label = "False"
                
                # 【核心逻辑】：在基础概率上叠加连续的均匀噪声扰动 [-0.05, 0.05]
                p_one = base_p + random.uniform(-0.05, 0.05)
                # 严格限制概率在 [0, 1] 之间，防止越界报错
                p_one = max(0.0, min(1.0, p_one))
                
                # 步骤一：按这个连续的浮点概率生成背景矩阵
                matrix = np.random.choice([0, 1], size=(matrix_size, matrix_size), p=[1 - p_one, p_one])
                
                # 步骤二：对于强连通类型的样本，叠加 DFS 随机游走路径
                if s_type in [1, 2, 3]:
                    path = generate_random_path(matrix_size, port_start, port_end)
                    for (r, c) in path:
                        matrix[r, c] = 1  # 强制铺设直流通路
                
                # 步骤三：格式化写入文件
                for row in range(matrix_size):
                    row_str = ','.join(str(matrix[row, col]) for col in range(matrix_size))
                    f.write(row_str + '\n')
                
                # 记录详细的元数据，保留到小数点后四位，方便后续回溯分析
                f.write(f'# Matrix Index: {global_matrix_idx} | Connected: {is_connected_label} | Type: {s_type} | P(1): {p_one:.4f}\n')
                
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