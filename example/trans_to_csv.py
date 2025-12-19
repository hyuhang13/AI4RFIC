import pandas as pd
import numpy as np

def process_emx_data(input_file, output_file):
    """
    处理EMX导出的特殊格式CSV数据
    """
    # 读取原始文件的所有行
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 提取表头（第一行）
    header_line = lines[0].strip()
    
    # 处理表头 - 原始表头缺少实例编号列
    # 原始表头: Freq(Hz),S11_real,S11_imag,...
    # 我们需要在Freq(Hz)前加上"instance"列
    original_headers = header_line.split(',')
    
    # 创建新的表头：在Freq(Hz)前插入instance列
    new_headers = ['instance'] + original_headers
    
    # 处理数据行
    data_rows = []
    for line in lines[1:]:  # 跳过表头行
        line = line.strip()
        if not line:
            continue
            
        # 分割每一行的数据
        values = line.split(',')
        
        # 第一列是实例编号，第二列开始是表头对应的数据
        instance_num = values[0]  # 实例编号，如"1"
        data_values = values[1:]  # 实际数据
        
        # 组合成一行完整的数据
        row_data = [instance_num] + data_values
        data_rows.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(data_rows, columns=new_headers)
    
    # 转换数据类型
    # 将instance列转换为整数
    df['instance'] = pd.to_numeric(df['instance'], errors='coerce').astype(int)
    
    # 将其余列转换为浮点数
    for col in df.columns:
        if col != 'instance':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"实例编号唯一值: {sorted(df['instance'].unique())}")
    
    # 保存处理后的数据
    df.to_csv(output_file, index=False)
    print(f"\n✅ 处理完成！已保存到: {output_file}")
    
    # 显示数据预览
    print("\n📊 处理后的数据预览:")
    print(df.head())
    
    return df

# 使用示例
if __name__ == "__main__":
    # 文件路径配置
    input_file = "dataset.csv"  # 你的原始文件
    output_file = "processed_dataset.csv"  # 处理后的文件
    
    # 处理数据
    df_processed = process_emx_data(input_file, output_file)
    
    # 可选：统计信息
    print("\n📈 数据统计:")
    print(f"总行数: {len(df_processed)}")
    print(f"实例数量: {df_processed['instance'].nunique()}")
    print(f"频率范围: {df_processed['Freq(Hz)'].min():.2e} - {df_processed['Freq(Hz)'].max():.2e} Hz")