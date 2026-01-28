import numpy as np
import random  # 添加这行导入语句
def create_matrix_from_data(data_string=None, file_path=None, return_type='list'):
    """
    从字符串或文件创建19x19矩阵
    
    Args:
        data_string: 矩阵字符串，每行用换行分隔，每列用逗号分隔
        file_path: 包含矩阵数据的文件路径
        return_type: 返回类型 ('list' 或 'numpy')
        
    Returns:
        如果return_type='list': Python二维列表 [[1,2,3], [4,5,6], ...]
        如果return_type='numpy': NumPy数组
    """
    import numpy as np
    
    if file_path:
        # 从文件读取
        with open(file_path, 'r') as f:
            lines = f.readlines()
    elif data_string:
        # 从字符串读取
        lines = data_string.strip().split('\n')
    else:
        raise ValueError("必须提供data_string或file_path")
    
    # 清理和转换数据
    matrix_data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 移除行尾逗号（如果有）
        if line.endswith(','):
            line = line[:-1]
        
        # 分割数字
        row_str = line.split(',')
        row = [int(x.strip()) for x in row_str if x.strip()]
        
        # 验证每行长度
        if len(row) != 19:
            print(f"警告: 行长度不是19，而是{len(row)}: {row}")
        
        matrix_data.append(row)
    
    # 验证形状
    if len(matrix_data) != 19:
        print(f"警告: 矩阵行数不是19，而是{len(matrix_data)}")
        # 如果需要，调整大小
        if len(matrix_data) * len(matrix_data[0]) == 361:  # 19x19=361
            # 扁平化然后重塑
            flat_data = [item for sublist in matrix_data for item in sublist]
            matrix_data = [flat_data[i:i+19] for i in range(0, 361, 19)]
    
    print(f"创建矩阵成功，行数: {len(matrix_data)}")
    print(f"矩阵统计: 1的比例={sum(sum(row) for row in matrix_data) / (len(matrix_data) * len(matrix_data[0])):.3f}")
    
    # 返回Python二维列表
    if return_type == 'list':
        print(matrix_data)
        return matrix_data
    elif return_type == 'numpy':
        return np.array(matrix_data, dtype=np.float32)
    else:
        raise ValueError(f"不支持的return_type: {return_type}")
def creat_random_matrix_string():
    import random

def generate_exact_format():
    """生成与问题示例完全相同格式的矩阵"""
    rows = 19
    cols = 19
    
    matrix_str = ""
    for i in range(rows):
        row_values = []
        for j in range(cols):
            row_values.append(str(random.randint(0, 1)))
        matrix_str += ",".join(row_values) + "\n"
    
    return matrix_str

# # 直接生成与示例格式相同的矩阵
# matrix_data = generate_exact_format()

# # 输出时可以包装在triple quotes中
# print(f'matrix_data = """')
# print(matrix_data, end='"""')
matrix_data = """
0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,1,0,0
0,1,1,1,0,0,1,1,1,1,1,1,0,0,1,0,0,0,0
0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1
0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,1
1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,1,1
1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,1
0,0,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0
1,0,0,1,1,1,1,0,1,1,0,1,0,1,1,0,0,1,1
1,0,1,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,1
0,0,0,1,1,0,1,0,1,1,0,1,1,0,0,1,1,1,1
0,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,1,1,1
0,1,0,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,1
0,1,0,0,1,0,0,1,0,1,0,0,1,1,0,0,0,1,0
0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1
1,0,1,1,0,1,1,0,1,1,1,0,0,1,1,0,0,1,0
1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,0,1
1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,0,1
1,0,1,1,0,1,0,0,1,1,0,1,0,1,1,1,1,1,1
1,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,0,1
"""

create_matrix_from_data(data_string=matrix_data)