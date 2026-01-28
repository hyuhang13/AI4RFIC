def verify_fixed_file(fixed_file):
    """验证修复后的文件"""
    import pandas as pd
    
    print(f"验证文件: {fixed_file}")
    
    # 读取前几行
    df = pd.read_csv(fixed_file, nrows=5)
    
    print(f"列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    print(f"前3行数据:")
    print(df.head(6))
    
    # 检查是否有空值
    print(f"每列空值数量:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        print(f"  {col}: {null_count} 个空值")
    
    print("验证完成！")

# 运行验证
verify_fixed_file("dataset/dataset_agg_cleaned.csv")