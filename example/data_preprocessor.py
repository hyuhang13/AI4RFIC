# data_loader/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG

class DataPreprocessor:
    def __init__(self):
        self.input_features = DATA_CONFIG['input_features']
        self.output_targets = DATA_CONFIG['output_targets']
        self.X_scaler = None
        self.y_scaler = None
        self.removed_feature_indices = []
    
    def load_and_preprocess_data(self, file_path):
        """
        加载和预处理电感数据
        """
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print("原始列名:", df.columns.tolist())
        print("数据形状:", df.shape)
        
        # 数据清洗和转换
        print("数据清洗前:")
        for col in self.input_features + self.output_targets:
            print(f"{col}: 数据类型={df[col].dtype}, 示例={df[col].iloc[:3].tolist()}")
        
        # 专门处理频率列 - 移除"GHz"单位并转换为数值
        if df['freq'].dtype == 'object':
            df['freq'] = df['freq'].str.replace('GHz', '', regex=False).str.strip()
            df['freq'] = pd.to_numeric(df['freq'], errors='coerce')
            print(f"频率列转换后，缺失值数量: {df['freq'].isna().sum()}")
        
        # 处理其他列，确保它们都是数值类型
        for col in self.input_features + self.output_targets:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                na_count = df[col].isna().sum()
                if na_count > 0:
                    print(f"列 {col} 转换后有 {na_count} 个缺失值")
        
        print("数据清洗后:")
        for col in self.input_features + self.output_targets:
            print(f"{col}: 数据类型={df[col].dtype}, 示例={df[col].iloc[:3].tolist()}")
        # # 移除Line_space特征
        # if 'Line_space' in self.input_features:
        #     line_space_idx = self.input_features.index('Line_space')
        #     self.removed_feature_indices.append(line_space_idx)
        #     # 创建新的特征列表，不包含Line_space
        #     self.input_features_used = [feat for feat in self.input_features if feat != 'Line_space']
        #     print(f"移除恒定特征: Line_space")
        #     print(f"使用的输入特征: {self.input_features_used}")
        # else:
        #     self.input_features_used = self.input_features.copy()
        X = df[self.input_features].values
        y = df[self.output_targets].values
        
        # 数据标准化
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        X_normalized = self.X_scaler.fit_transform(X)
        y_normalized = self.y_scaler.fit_transform(y)
        # 打印处理后的特征范围
        print("\n=== 标准化后的特征范围 ===")
        for i, feature in enumerate(self.input_features):#self.input_features_used
            print(f"  {feature}: [{X_normalized[:, i].min():.2f}, {X_normalized[:, i].max():.2f}]")
        return X_normalized, y_normalized
    
    # def split_data(self, X, y):
    #     """划分训练集和测试集"""
    #     return train_test_split(
    #         X, y, 
    #         test_size=DATA_CONFIG['test_size'], 
    #         random_state=DATA_CONFIG['random_state']
    #     )
    def split_data(self, X, y, val_size=0.2, test_size=0.1, random_state=42):
        """
        划分训练集、验证集和测试集
        
        参数:
        - X: 特征数据
        - y: 目标数据
        - val_size: 验证集比例 (默认0.2)
        - test_size: 测试集比例 (默认0.1)
        - random_state: 随机种子，确保结果可复现
        
        返回:
        - X_train, X_val, X_test, y_train, y_val, y_test
        """
        # 计算训练集比例
        train_size = 1 - val_size - test_size
        print(f"数据划分比例: 训练集 {train_size:.0%}, 验证集 {val_size:.0%}, 测试集 {test_size:.0%}")
        
        # 第一步：先分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            shuffle=True  # 打乱数据
        )
        
        # 第二步：从剩余数据中分出验证集
        # 注意：这里需要调整验证集的比例，因为是在剩余数据中的比例
        val_size_adjusted = 0.2#val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            shuffle=True  # 打乱数据
        )
        
        # 输出数据集大小
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本") 
        print(f"测试集: {X_test.shape[0]} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_new_data(self, X_new):
        """
        对新数据进行相同的预处理
        X_new: 新数据，形状为(n_samples, n_features)，特征顺序应与原始input_features一致
        """
        if self.X_scaler is None:
            raise ValueError("必须先调用load_and_preprocess_data方法训练预处理器")
        
        # 移除Line_space特征
        if self.removed_feature_indices:
            X_processed = np.delete(X_new, self.removed_feature_indices, axis=1)
        else:
            X_processed = X_new.copy()
        
        # 标准化
        X_normalized = self.X_scaler.transform(X_processed)
        
        return X_normalized