import numpy as np
import os
import random
from collections import deque
from datetime import datetime

def check_connectivity(matrix, start=(9, 0), end=(18, 0)):
    """使用 BFS 检查起点到终点是否连通"""
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

def generate_pure_random_matrices(total_matrices=100000, num_files=5, matrix_size=19, output_dir="pure_random_dataset"):
    os.makedirs(output_dir, exist_ok=True)
    matrices_per_file = total_matrices // num_files
    port_start, port_end = (9, 0), (9, 18)
    global_matrix_idx = 1
    
    print(f"=== 开始基于纯随机过滤生成数据集 ===")
    start_time = datetime.now()
    
    for file_idx in range(1, num_files + 1):
        output_file = os.path.join(output_dir, f"pure_random_part{file_idx}.txt")
        print(f"\n>> 正在生成第 {file_idx}/{num_files} 个文件...")
        
        with open(output_file, 'w') as f:
            for _ in range(matrices_per_file):
                # 为了保证代码能运行结束，我们将概率分布限制在 0.45 到 0.75 之间
                # 若低于 0.45，寻找连通样本的时间将呈指数级增长
                p_one = 0.4 #random.uniform(0.45, 0.75)
                
                attempts = 0
                matrix = None
                is_connected = False
                
                # 核心过滤循环：如果不连通，就一直重新生成
                while not is_connected:
                    attempts += 1
                    # 纯随机生成矩阵
                    matrix = np.random.choice([0, 1], size=(matrix_size, matrix_size), p=[1 - p_one, p_one])
                    # 验证连通性
                    is_connected = check_connectivity(matrix, port_start, port_end)
                    
                    # 加入安全锁，防止低概率下陷入无限死循环
                    if attempts > 10000000:
                        print(f"\n[警告] 概率 p={p_one:.4f} 下尝试了 20000 次仍未生成连通路径，强制跳出以防死机。")
                        break
                
                # 如果因为安全锁跳出，则忽略该样本，进行下一次外部循环（简化处理，这里不严格补齐总数）
                if not is_connected:
                    continue
                
                # 写入矩阵数据
                for row in range(matrix_size):
                    row_str = ','.join(str(matrix[row, col]) for col in range(matrix_size))
                    f.write(row_str + '\n')
                
                # 记录元数据，特别记录了 attempts（为了生成这1个有效样本，丢弃了多少个废样本）
                f.write(f'# Matrix Index: {global_matrix_idx} | P(1): {p_one:.4f} | Attempts: {attempts}\n')
                
                if global_matrix_idx % 5000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"   进度: {global_matrix_idx}/{total_matrices} - 耗时: {elapsed:.1f}s")
                    
                global_matrix_idx += 1

if __name__ == "__main__":
    generate_pure_random_matrices()