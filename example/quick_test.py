import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

# ==================== 配置参数 ====================
Z0 = 50.0                          # 参考阻抗（欧姆）
freq_unit_csv = 'Hz'               # CSV文件中频率的单位，可选 'Hz' 或 'GHz'
json_file =  'genetic_results/result_20260303_195827/optimization_results_20260303_201548.json'              # JSON数据文件路径
csv_real = 'zm11_real.csv'          # 阻抗实部文件路径
csv_imag = 'zm11_imag.csv'          # 阻抗虚部文件路径

# ==================== 读取JSON数据 ====================
with open(json_file, 'r') as f:
    data = json.load(f)

freq_json = np.array([p['freq'] for p in data['freq_predictions']])  # 单位 Hz
s22_real_json = np.array([p['s_params']['S22_real'] for p in data['freq_predictions']])
s22_imag_json = np.array([p['s_params']['S22_imag'] for p in data['freq_predictions']])

# ==================== 读取CSV阻抗数据 ====================
# 跳过第一行标题，指定列名
df_real = pd.read_csv(csv_real, skiprows=1, names=['freq', 'real'])
df_imag = pd.read_csv(csv_imag, skiprows=1, names=['freq', 'imag'])

# 转换为浮点数（确保）
freq_csv = df_real['freq'].astype(float).values
if not np.allclose(freq_csv, df_imag['freq'].astype(float).values):
    raise ValueError("实部和虚部文件的频率不一致，请检查")

z_real = df_real['real'].astype(float).values
z_imag = df_imag['imag'].astype(float).values

# 构建复阻抗
z_csv = z_real - 1j * z_imag

# 频率单位转换：统一为 Hz
if freq_unit_csv.upper() == 'GHZ':
    freq_csv_hz = freq_csv * 1e9
else:  # Hz
    freq_csv_hz = freq_csv

# ==================== 阻抗 -> S参数转换 ====================
# 默认将CSV阻抗视为 Z22（输出端口阻抗）转换为 S22
s22_csv = (z_csv - Z0) / (z_csv + Z0)
s22_real_csv = np.real(s22_csv)
s22_imag_csv = np.imag(s22_csv)

# 如果实际应为 Z11（输入阻抗）转换为 S11，请取消下面两行注释并注释上面两行
# s11_csv = (z_csv - Z0) / (z_csv + Z0)
# s22_real_csv = np.real(s11_csv)   # 注意此时变量名仍是 s22，可根据需要修改
# s22_imag_csv = np.imag(s11_csv)

# ==================== 插值JSON数据（可选） ====================
# 使用线性插值获得与CSV相同频率点上的值，便于直接对比
freq_interp = freq_csv_hz
interp_real = interpolate.interp1d(freq_json, s22_real_json, kind='linear', 
                                    fill_value='extrapolate')
interp_imag = interpolate.interp1d(freq_json, s22_imag_json, kind='linear', 
                                    fill_value='extrapolate')
s22_real_interp = interp_real(freq_interp)
s22_imag_interp = interp_imag(freq_interp)

# ==================== 绘图 ====================
plt.figure(figsize=(12, 5))

# 实部子图
plt.subplot(1, 2, 1)
plt.plot(freq_csv_hz/1e9, s22_real_csv, 'b-', label='From CSV (Z22 -> S22)')
plt.plot(freq_json/1e9, s22_real_json, 'ro', label='JSON points (S22)')
plt.plot(freq_csv_hz/1e9, s22_real_interp, 'r--', label='JSON interpolated', alpha=0.7)
plt.xlabel('Frequency (GHz)')
plt.ylabel('S22 Real Part')
plt.title('S22 Real Part Comparison')
plt.legend()
plt.grid(True)

# 虚部子图
plt.subplot(1, 2, 2)
plt.plot(freq_csv_hz/1e9, s22_imag_csv, 'b-', label='From CSV (Z22 -> S22)')
plt.plot(freq_json/1e9, s22_imag_json, 'ro', label='JSON points (S22)')
plt.plot(freq_csv_hz/1e9, s22_imag_interp, 'r--', label='JSON interpolated', alpha=0.7)
plt.xlabel('Frequency (GHz)')
plt.ylabel('S22 Imag Part')
plt.title('S22 Imag Part Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()