import numpy as np
import os
import random
from collections import deque
from datetime import datetime

def check_connectivity(matrix, start=(9, 0), end=(18, 0)):
    """使用广度优先搜索 (BFS) 验证起点到终点的绝对物理连通性"""
    if matrix[start[0]][start[1]] == 0 or matrix[end[0]][end[1]] == 0:
        return False
        
    rows, cols = len(matrix), len(matrix[0])
    visited = set([start])
    queue = deque([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
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

def generate_p_from_beta_peak_45():
    """
    核心数学引擎：使用 Beta(4, 6) 分布。
    将其线性映射到 [0.3, 0.7] 区间，使得生成概率的峰值精准落在 0.45。
    """
    x = np.random.beta(4, 6)
    p_gen = 0.3 + 0.4 * x  # 跨度为 0.4
    
    # 加入极值保护，防止由于浮点精度问题越界
    return max(0.01, min(0.99, p_gen))

def generate_rigorous_dataset(total_matrices=100000, num_files=5, matrix_size=19, output_dir="beta_filtered_dataset"):
    os.makedirs(output_dir, exist_ok=True)
    matrices_per_file = total_matrices // num_files
    port_start, port_end = (9, 0), (9, 18)
    global_matrix_idx = 1
    
    print(f"=== 开始生成基于 Beta(4,6) 映射过滤的严谨数据集 ===")
    print(f"总样本数: {total_matrices} | 策略: 80% 过滤连通, 20% 纯随机放行")
    start_time = datetime.now()
    
    for file_idx in range(1, num_files + 1):
        output_file = os.path.join(output_dir, f"matrices_part{file_idx}.txt")
        print(f"\n>> 正在处理第 {file_idx}/{num_files} 个文件...")
        
        # 精确划分当前文件的 80/20 任务比例
        n_filtered = int(matrices_per_file * 0.8)
        n_unfiltered = matrices_per_file - n_filtered
        
        # 任务标签：1 代表必须过滤至连通，0 代表纯随机放行
        sample_tasks = [1] * n_filtered + [0] * n_unfiltered
        random.shuffle(sample_tasks)  # 彻底打乱生成顺序
        
        with open(output_file, 'w') as f:
            for task_type in sample_tasks:
                p_gen = generate_p_from_beta_peak_45()
                attempts = 1
                
                if task_type == 1:
                    # 【80% 组】：执行严格的 Rejection Sampling，直到连通为止
                    is_connected = False
                    while not is_connected:
                        matrix = np.random.choice([0, 1], size=(matrix_size, matrix_size), p=[1 - p_gen, p_gen])
                        is_connected = check_connectivity(matrix, port_start, port_end)
                        
                        if not is_connected:
                            attempts += 1
                            # 【安全阀】：如果在 p 接近 0.3 时陷入地狱级死循环
                            if attempts > 50000:
                                # 妥协策略：重新分配一个略高的概率以突破渗流阈值封锁
                                p_gen = random.uniform(0.45, 0.60)
                                attempts = 1
                else:
                    # 【20% 组】：直接生成，顺手检测一下连通性用于打标签
                    matrix = np.random.choice([0, 1], size=(matrix_size, matrix_size), p=[1 - p_gen, p_gen])
                    is_connected = check_connectivity(matrix, port_start, port_end)
                
                # 步骤三：格式化写入文件
                for row in range(matrix_size):
                    row_str = ','.join(str(matrix[row, col]) for col in range(matrix_size))
                    f.write(row_str + '\n')
                
                # 写入极其详尽的元数据
                task_name = "Filtered(80%)" if task_type == 1 else "Unfiltered(20%)"
                connected_label = "True" if is_connected else "False"
                
                f.write(f'# Matrix Index: {global_matrix_idx} | Task: {task_name} | Connected: {connected_label} | P(1): {p_gen:.4f} | Attempts: {attempts}\n')
                
                # 打印进度监控
                if global_matrix_idx % 5000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"   进度: {global_matrix_idx}/{total_matrices} - 已耗时: {elapsed:.1f}s")
                    
                global_matrix_idx += 1

if __name__ == "__main__":
    # 核心配置参数
    TOTAL_MATRICES = 100000
    NUM_FILES = 5
    MATRIX_SIZE = 19
    OUTPUT_DIR = "beta_filtered_dataset"
    
    generate_rigorous_dataset(
        total_matrices=TOTAL_MATRICES,
        num_files=NUM_FILES,
        matrix_size=MATRIX_SIZE,
        output_dir=OUTPUT_DIR
    )