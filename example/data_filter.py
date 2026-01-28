import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def enhanced_preprocess_inductor_data(input_file, output_file, create_plots=True):
    """
    增强版数据预处理函数，包含数据分析和可视化
    """
    
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"原始数据形状: {df.shape}")
    
    # 数据清洗
    original_count = len(df)
    
    # 1. 剔除电阻大于100的行
    df_cleaned = df[df['Reff'] <= 100]
    print(f"剔除电阻>100: {original_count} -> {len(df_cleaned)} 行")
    
    # 2. 剔除四列为负值的行
    columns_to_check = ['Ldiff', 'Qdiff', 'Leff', 'Q']
    
    for column in columns_to_check:
        if column in df_cleaned.columns:
            # 转换为数值类型
            df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            before_count = len(df_cleaned)
            df_cleaned = df_cleaned[df_cleaned[column] >= 0]
            after_count = len(df_cleaned)
            print(f"剔除{column}<0: {before_count} -> {after_count} 行")
    
    # 数据质量报告
    print(f"\n=== 数据质量报告 ===")
    print(f"原始数据行数: {original_count}")
    print(f"最终数据行数: {len(df_cleaned)}")
    print(f"数据保留率: {len(df_cleaned)/original_count*100:.2f}%")
    
    # 创建可视化（可选）
    if create_plots:
        create_visualizations(df, df_cleaned)
    
    # 保存清洗后的数据
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n清洗后的数据已保存到: {output_file}")
    
    return df_cleaned

def create_visualizations(original_df, cleaned_df):
    """
    创建数据清洗前后的对比可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 电阻分布对比
    axes[0, 0].hist(original_df['Reff'], bins=50, alpha=0.7, label='原始数据', color='blue')
    axes[0, 0].hist(cleaned_df['Reff'], bins=50, alpha=0.7, label='清洗后', color='red')
    axes[0, 0].axvline(x=100, color='black', linestyle='--', label='阈值(100)')
    axes[0, 0].set_xlabel('电阻 Reff')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('电阻分布对比')
    axes[0, 0].legend()
    
    # 2. 电感值分布
    if 'Ldiff' in cleaned_df.columns:
        axes[0, 1].hist(cleaned_df['Ldiff'], bins=50, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('电感 Ldiff')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('清洗后电感分布')
    
    # 3. 品质因数分布
    if 'Q' in cleaned_df.columns:
        axes[1, 0].hist(cleaned_df['Q'], bins=50, alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('品质因数 Q')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('清洗后品质因数分布')
    
    # 4. 数据量对比
    categories = ['原始数据', '清洗后数据']
    counts = [len(original_df), len(cleaned_df)]
    axes[1, 1].bar(categories, counts, color=['blue', 'green'])
    axes[1, 1].set_ylabel('数据行数')
    axes[1, 1].set_title('数据量对比')
    
    plt.tight_layout()
    plt.savefig('data_cleaning_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 使用示例
if __name__ == "__main__":
    input_file = "Diff_SQ_XFAB_MJ_All_copy.csv"  # 替换为您的文件路径
    output_file = "cleaned_inductor_data.csv"
    
    # 执行数据预处理
    cleaned_data = enhanced_preprocess_inductor_data(input_file, output_file)
    
    # 显示最终数据的信息
    print("\n=== 清洗后数据信息 ===")
    print(cleaned_data.info())
    print("\n前5行数据:")
    print(cleaned_data.head())