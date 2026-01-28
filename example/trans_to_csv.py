import pandas as pd

# 读取CSV文件
# 注意：根据你的数据，第一列没有表头，所以我们需要手动指定列名
# column_names = ['ID', 'Freq(Hz)', 'S11_real', 'S11_imag', 'S21_real', 'S21_imag', 
#                 'S12_real', 'S12_imag', 'S22_real', 'S22_imag']

# df = pd.read_csv('your_file.csv', header=0, names=column_names)

# 或者如果你的CSV文件已经有表头，但第一列没有列名，可以这样处理：
df = pd.read_csv('dataset.csv')
# 如果第一列没有名称，给它命名为'ID'
df = df.rename(columns={df.columns[0]: 'ID'})

# 删除每个ID的第一行（Freq=0的行）
# 方法1：按ID分组，然后删除每个组的第一行
df_filtered = df.groupby('ID').apply(lambda x: x.iloc[1:]).reset_index(drop=True)

# 方法2：或者直接删除所有Freq(Hz)=0的行（这假设只有第一行是Freq=0）
# df_filtered = df[df['Freq(Hz)'] != 0]

# 保存到新的CSV文件
df_filtered.to_csv('filtered_file.csv', index=False)

print(f"原始数据行数: {len(df)}")
print(f"处理后数据行数: {len(df_filtered)}")
print(f"删除了 {len(df) - len(df_filtered)} 行数据")

# 验证结果
print("\n处理后每个ID的数据行数:")
for id_value in df_filtered['ID'].unique():
    id_count = len(df_filtered[df_filtered['ID'] == id_value])
    print(f"ID {id_value}: {id_count} 行")