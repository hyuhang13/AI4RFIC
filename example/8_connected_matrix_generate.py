import numpy as np
import os
from datetime import datetime

def generate_baseline_matrices(total_matrices=100000, num_files=5, matrix_size=19, output_dir="baseline_8conn_dataset"):
    os.makedirs(output_dir, exist_ok=True)
    matrices_per_file = total_matrices // num_files
    global_matrix_idx = 1
    p_one = 0.5  # 严格固定为 0.5
    
    # 【更新】：设置左右端口坐标
    port_start = (9, 0)
    port_end = (9, 18) 
    
    print(f"=== 开始生成 p=0.5 的纯随机基线数据集 (强制端口置1) ===")
    print(f"输入端口: {port_start} | 输出端口: {port_end}")
    start_time = datetime.now()
    
    for file_idx in range(1, num_files + 1):
        output_file = os.path.join(output_dir, f"matrices_part{file_idx}.txt")
        print(f"\n>> 正在生成第 {file_idx}/{num_files} 个文件...")
        
        with open(output_file, 'w') as f:
            for _ in range(matrices_per_file):
                # 纯随机生成矩阵
                matrix = np.random.choice([0, 1], size=(matrix_size, matrix_size), p=[1 - p_one, p_one])
                
                # 【核心修改】：强行将输入和输出端口的像素置为 1
                matrix[port_start[0], port_start[1]] = 1
                matrix[port_end[0], port_end[1]] = 1
                
                # 写入矩阵数据
                for row in range(matrix_size):
                    row_str = ','.join(str(matrix[row, col]) for col in range(matrix_size))
                    f.write(row_str + '\n')
                
                # 记录元数据
                f.write(f'# Matrix Index: {global_matrix_idx} | P(1): {p_one}\n')
                
                if global_matrix_idx % 10000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"   进度: {global_matrix_idx}/{total_matrices} - 耗时: {elapsed:.1f}s")
                    
                global_matrix_idx += 1

if __name__ == "__main__":
    generate_baseline_matrices()