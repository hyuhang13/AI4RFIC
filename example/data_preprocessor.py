# data_loader/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG
from tqdm import tqdm
# import matplotlib as plt
import matplotlib.pyplot as plt
import torch
class DataPreprocessor:
    def __init__(self, matrix_file='binary_matrices.txt', s_param_file='dataset.csv'):
        """
        init dataset: matrix_file, s_param_file
        """
        self.matrix_file = matrix_file
        self.s_param_file = s_param_file
        self.idx = 2000#100000
        self.s_param_dict = None
        self.matrices = None
        self.input_features = DATA_CONFIG['input_features']
        self.output_targets = DATA_CONFIG['output_targets']
        self.X_scaler = None
        self.y_scaler = None
        self.y_scalers = {}

        
    def _load_matrices(self):
        """加载二进制矩阵"""
        print("正在加载二进制矩阵...")
        matrices = []
        current_matrix = []
        
        with open(self.matrix_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # 跳过空行和注释行
                if not line or line.startswith('#'):
                    if current_matrix and len(current_matrix) == 19:
                        matrices.append(np.array(current_matrix, dtype=np.float32))
                        current_matrix = []
                    continue
                
                # 解析矩阵行
                row = [float(x) for x in line.split(',')]
                if len(row) == 19:
                    current_matrix.append(row)
        
        # 添加最后一个矩阵
        if current_matrix and len(current_matrix) == 19:
            matrices.append(np.array(current_matrix, dtype=np.float32))
        
        # 转换为三维数组 (N, 19, 19)
        matrices_array = np.stack(matrices)
        
        # 重塑为适合CNN的格式 (N, 1, 19, 19) - 单通道图像
        matrices_array = matrices_array[:, np.newaxis, :, :]
        
        print(f"矩阵数据形状: {matrices_array.shape}")
        self.matrices = matrices_array
        return matrices_array
    
    def _load_s_params(self):
        """加载S参数"""
        print("正在加载S参数...")
        # 读取CSV文件
        df = pd.read_csv(self.s_param_file)
        
        # 提取所有唯一的矩阵索引
        matrix_indices = df.iloc[:, 0].unique()
        s_param_dict = {}
        # 对于每个矩阵，提取其300个频率点的S参数
        
        for idx in tqdm(matrix_indices, desc="处理S参数"):
            # 获取当前矩阵的所有行
            matrix_data = df[df.iloc[:, 0] == idx]
            
            # 确保有300行数据
            if len(matrix_data) != 300:
                # 如果数据不足300行，使用插值填充
                print(f"警告: 矩阵{idx}只有{len(matrix_data)}行数据")
                # 这里可以添加插值逻辑，但根据您的描述应该是完整的300行
                continue
            
            # 提取S参数（跳过第一列索引和第二列频率）
            frequencies = matrix_data.iloc[:, 1].values #(300,1) 
            #频率归一化
            frequency_norm = self.prepare_frequency_input(frequencies, 'linear')

            s_params = matrix_data.iloc[:, 2:].values   #(300,8)
            s_param_dict[int(idx)] = {
                'frequencies': frequency_norm.astype(np.float32),
                's_params': s_params.astype(np.float32)
            }
        self.s_param_dict = s_param_dict
        return s_param_dict
    
    def __len__(self):
        return len(self.matrices)*300
    
    def __getitem__(self, idx):
        # 计算矩阵索引和频率索引
        matrix_idx = idx // 300
        freq_idx = idx % 300
        
        # 获取矩阵
        matrix = self.matrices[matrix_idx]
        
        # 获取频率和S参数
        matrix_id = matrix_idx + 1  # 因为matrix.txt从索引1开始
        if matrix_id in self.s_param_dict:
            frequencies = self.s_param_dict[matrix_id]['frequencies']
            s_params_all = self.s_param_dict[matrix_id]['s_params']
            
            frequency = frequencies[freq_idx]
            s_params = s_params_all[freq_idx]
        else:
            # 如果找不到，使用近似值
            print("ERROR:wrong idx!\n")
            frequency = np.float32(freq_idx * 0.1)  # 0.1GHz步长
            s_params = np.zeros(8, dtype=np.float32)
        
        # 频率归一化 (0-30GHz归一化到0-1)
        #frequency_norm = frequency / 30e9
        
        return {
            'matrix': torch.FloatTensor(matrix),  # (1, 19, 19)
            'frequency': torch.FloatTensor([frequency]),  # 归一化频率
            's_params': torch.FloatTensor(s_params)  # 8个S参数
        }

    def load_and_clean_data(self):
        """
        改进的标准化预处理：对表现差的目标使用RobustScaler
        """
        self.matrices = self._load_matrices()
        self.s_params = self._load_s_params()
        # 验证数据一致性
        assert len(self.matrices) == len(self.s_params), \
            f"矩阵数量({len(self.matrices)})与S参数数量({len(self.s_params)})不匹配"
        
        print(f"数据集加载完成：{len(self.matrices)}个样本")

    def fit_preprocessors(self, X_train, y_train):
        """
        只在训练集上拟合预处理器
        """
        pass
    
    def transform_data(self, X, y):
        """
        使用训练集拟合的预处理器变换数据
        """
        pass

    def inverse_transform_y(self, y_normalized):
        pass
    
    def get_all_data(self):
        """获取所有数据，用于train_test_split"""
        print("正在准备所有数据用于train_test_split...")
        
        all_matrices = []
        all_frequencies = []
        all_s_params = []
        all_indices = []
        all_matrix_indices = []
        all_freq_indices = []

        total_matrices = len(self.matrices)  # 100,000
        freq_per_matrix = 300
        all_matrix_data = self.matrices
        # 遍历所有样本
        for matrix_idx in tqdm(range(total_matrices), desc="收集数据"):
            matrix = all_matrix_data[matrix_idx]  # (1, 19, 19)
            matrix_id = matrix_idx + 1
            if matrix_id in self.s_param_dict:
                frequencies = self.s_param_dict[matrix_id]['frequencies']  # (300,1)
                s_params_all = self.s_param_dict[matrix_id]['s_params']  # (300, 8)

                for freq_idx in range(freq_per_matrix):
                    all_matrices.append(matrix)
                    all_frequencies.append([frequencies[freq_idx]])  
                    all_s_params.append(s_params_all[freq_idx])
                    all_matrix_indices.append(matrix_idx)
                    all_freq_indices.append(freq_idx)
            else:
                print("can't find S_matrix")
                for freq_idx in range(freq_per_matrix):
                    all_matrices.append(matrix)
                    frequency = freq_idx * 0.1
                    all_frequencies.append([frequency])  
                    all_s_params.append(np.zeros(8, dtype=np.float32))
                    all_matrix_indices.append(matrix_idx)
                    all_freq_indices.append(freq_idx)
        
        # 转换为numpy数组 N = 100000*300,
        print("转换为numpy数组...")
        all_matrices_array = np.stack(all_matrices)  # (N, 1, 19, 19)
        all_frequencies_array = np.stack(all_frequencies)  # (N, 1)
        all_s_params_array = np.stack(all_s_params)  # (N, 8)
        all_matrix_indices_array = np.array(all_matrix_indices)  # (N,)
        all_freq_indices_array = np.array(all_freq_indices)  # (N,)
    
        print(f"数据形状:")
        print(f"  矩阵: {all_matrices_array.shape}")
        print(f"  频率: {all_frequencies_array.shape}")
        print(f"  S参数: {all_s_params_array.shape}")
        print(f"indices: {all_matrix_indices_array.shape}")
        return all_matrices_array, all_frequencies_array, all_s_params_array, all_matrix_indices_array
    def prepare_frequency_input(self, frequencies, method='linear'):
        """准备频率输入"""
        if method == 'log':
            # 对数归一化（适合宽频带）
            log_freq = np.log10(frequencies)
            log_min = np.min(log_freq)
            log_max = np.max(log_freq)
            return (log_freq - log_min) / (log_max - log_min)
        elif method == 'linear':
            # 线性归一化
            return (frequencies - frequencies.min()) / (frequencies.max() - frequencies.min())
        elif method == 'periodic':
            # 周期性编码（适合高频）
            # 将频率转换为正弦和余弦特征
            normalized = (frequencies - frequencies.min()) / (frequencies.max() - frequencies.min())
            angle = 2 * np.pi * normalized
            return np.column_stack([np.sin(angle), np.cos(angle)])  
    def split_data(self, X_matrices, X_frequencies, y_s_params, indices=None,
                   test_size=0.2, val_size=0.1, random_state=42):
        """
        使用train_test_split划分数据
        
        Args:
            X_matrices: 矩阵数据 (N, 1, 19, 19)
            X_frequencies: 频率数据 (N, 1)
            y_s_params: S参数标签 (N, 8)
            indices: 原始索引 (N,)
            test_size: 测试集比例 (默认0.2)
            val_size: 验证集比例 (默认0.1)
            random_state: 随机种子
            
        Returns:
            划分后的数据集
        """
        print(f"数据划分比例: 训练集 {1-test_size-val_size:.0%}, "
              f"验证集 {val_size:.0%}, 测试集 {test_size:.0%}")
        
        # 第一步：划分训练集和临时集（包含验证集和测试集）
        X_temp_matrices, X_test_matrices, X_temp_freq, X_test_freq, \
        y_temp, y_test, temp_idx, test_idx = train_test_split(
            X_matrices, X_frequencies, y_s_params, indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        # 第二步：从临时集中划分验证集
        # 计算验证集在临时集中的比例
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train_matrices, X_val_matrices, X_train_freq, X_val_freq, \
        y_train, y_val, train_idx, val_idx = train_test_split(
            X_temp_matrices, X_temp_freq, y_temp, temp_idx,
            test_size=val_size_adjusted,
            random_state=random_state,
            shuffle=True
        )
        
        # 打印划分结果
        # print(f"训练集: {X_train_matrices.shape[0]} 个样本")
        # print(f"验证集: {X_val_matrices.shape[0]} 个样本")
        # print(f"测试集: {X_test_matrices.shape[0]} 个样本")
        
        # 返回划分结果
        return {
            # N1 + N2 + N3 = 300*100k
            'train': {  # N1 = 21,000,000
                'matrices': X_train_matrices,# (N1, 1, 19, 19)
                'frequencies': X_train_freq, # (N1, 1)
                's_params': y_train,         # (N1, 8)
                'indices': train_idx         # (N1,)
            },
            'val': {    # N2 = 6,000,000
                'matrices': X_val_matrices,
                'frequencies': X_val_freq,
                's_params': y_val,
                'indices': val_idx
            },
            'test': {   # N3 = 3,000,000
                'matrices': X_test_matrices,
                'frequencies': X_test_freq,
                's_params': y_test,
                'indices': test_idx
            }
        }
    def analyze_data_distribution(self, split_data_dict):
        """分析数据分布"""
        # print("\n" + "="*60)
        # print("数据分布分析")
        # print("="*60)
        
        train_size = len(split_data_dict['train']['s_params'])
        val_size = len(split_data_dict['val']['s_params'])
        test_size = len(split_data_dict['test']['s_params'])
        total_size = train_size + val_size + test_size
        
        print(f"训练集: {train_size:,} 个样本 ({train_size/total_size:.1%})")
        print(f"验证集: {val_size:,} 个样本 ({val_size/total_size:.1%})")
        print(f"测试集: {test_size:,} 个样本 ({test_size/total_size:.1%})")
        print(f"总计: {total_size:,} 个样本")
        
        # 分析S参数的统计特性
        print("\nS参数统计特性:")
        for split_name, split_data in split_data_dict.items():
            s_params = split_data['s_params']
            print(f"\n{split_name.capitalize()}集:")
            print(f"  形状: {s_params.shape}")
            print(f"  均值范围: [{s_params.mean(axis=0).min():.4f}, {s_params.mean(axis=0).max():.4f}]")
            print(f"  标准差范围: [{s_params.std(axis=0).min():.4f}, {s_params.std(axis=0).max():.4f}]")
            print(f"  最小值范围: [{s_params.min(axis=0).min():.4f}, {s_params.min(axis=0).max():.4f}]")
            print(f"  最大值范围: [{s_params.max(axis=0).min():.4f}, {s_params.max(axis=0).max():.4f}]")
    

    def visualize_data_distribution(self,split_data_dict, save_path='data_distribution.png'):
        """可视化数据分布"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # S11实部的分布
        for ax_idx, (split_name, split_data) in enumerate(split_data_dict.items()):
            s_params = split_data['s_params']
            s11_real = s_params[:, 0]  # 第一个是S11实部
            
            axes[0, ax_idx].hist(s11_real, bins=50, alpha=0.7, color=['blue', 'green', 'red'][ax_idx])
            axes[0, ax_idx].set_title(f'{split_name.capitalize()} - S11_Real Distribution')
            axes[0, ax_idx].set_xlabel('S11_Real')
            axes[0, ax_idx].set_ylabel('Freq')
            axes[0, ax_idx].grid(True, alpha=0.3)
        
        # 频率分布
        for ax_idx, (split_name, split_data) in enumerate(split_data_dict.items()):
            frequencies = split_data['frequencies']
            
            axes[1, ax_idx].hist(frequencies, bins=50, alpha=0.7, color=['blue', 'green', 'red'][ax_idx])
            axes[1, ax_idx].set_title(f'{split_name.capitalize()} - Freq Distribution')
            axes[1, ax_idx].set_xlabel('Normalization Freq')
            axes[1, ax_idx].set_ylabel('Hz')
            axes[1, ax_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
