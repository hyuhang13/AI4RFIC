# data_loader/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG

class DataPreprocessor:
    def __init__(self):
        self.input_features = DATA_CONFIG['input_features']
        self.output_targets = DATA_CONFIG['output_targets']
        self.X_scaler = None
        self.y_scaler = None
        self.y_scalers = {}
        self.removed_feature_indices = []
        self.scale_factor = 1e9
    def load_and_preprocess_data(self, file_path):
        """
        改进的标准化预处理：对表现差的目标使用RobustScaler
        """
        # 读取和清洗数据
        df = pd.read_csv(file_path)
        print("原始列名:", df.columns.tolist())
        print("数据形状:", df.shape)
        
        # 数据清洗和转换
        print("数据清洗前:")
        for col in self.input_features + self.output_targets:
            print(f"{col}: 数据类型={df[col].dtype}, 示例={df[col].iloc[:3].tolist()}")
        
        # 专门处理频率列
        if df['freq'].dtype == 'object':
            df['freq'] = df['freq'].str.replace('GHz', '', regex=False).str.strip()
            df['freq'] = pd.to_numeric(df['freq'], errors='coerce')
            print(f"频率列转换后，缺失值数量: {df['freq'].isna().sum()}")
        
        # 处理其他列
        for col in self.input_features + self.output_targets:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                na_count = df[col].isna().sum()
                if na_count > 0:
                    print(f"列 {col} 转换后有 {na_count} 个缺失值")
        
        print("数据清洗后:")
        for col in self.input_features + self.output_targets:
            print(f"{col}: 数据类型={df[col].dtype}, 示例={df[col].iloc[:3].tolist()}")
        
        # 移除Line_space特征
        if 'Line_space' in self.input_features:
            line_space_idx = self.input_features.index('Line_space')
            self.removed_feature_indices.append(line_space_idx)
            self.input_features_used = [feat for feat in self.input_features if feat != 'Line_space']
            print(f"移除恒定特征: Line_space")
            print(f"使用的输入特征: {self.input_features_used}")
        else:
            self.input_features_used = self.input_features.copy()
        
        # 提取特征和目标
        X = df[self.input_features_used].values
        y = df[self.output_targets].values
        # 数据完整性检查
        print("\n=== 原始数据统计 ===")
        for i, target in enumerate(self.output_targets):
            target_col = y[:, i]
            print(f"{target}: 范围[{target_col.min():.6e}, {target_col.max():.6e}], "
                  f"均值:{target_col.mean():.6e}, 标准差:{target_col.std():.6e}")
        # 输入特征标准化
        self.X_scaler = StandardScaler()
        X_normalized = self.X_scaler.fit_transform(X)
        # 对输出目标分别处理
        y_normalized = np.zeros_like(y)
        
        # 对Ldiff和Leff使用StandardScaler（表现好）
        for i, target in enumerate(self.output_targets):
            # if target in ['Ldiff', 'Leff']:
            #     scaler = StandardScaler()
            #     y_normalized[:, i] = scaler.fit_transform(y[:, i].reshape(-1, 1)).flatten()
            #     self.y_scalers[target] = ('standard', scaler)
            
            # 对Qdiff和Q使用Yeo-Johnson变换（处理负值）
            if target in ['Ldiff', 'Leff']:
                scaler = QuantileTransformer(
                n_quantiles=min(1000, len(y)),
                output_distribution='normal',  # 
                random_state=42
                )
                y_expanded = y[:, i] * self.scale_factor
                y_normalized[:, i] = scaler.fit_transform(y_expanded.reshape(-1, 1)).flatten()
                # y_normalized[:, i] = scaler.fit_transform(y[:, i].reshape(-1, 1)).flatten()
                self.y_scalers[target] = ('multi+quantile', scaler)
            elif target in ['Qdiff', 'Q']:
                # scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                # y_normalized[:, i] = scaler.fit_transform(y[:, i].reshape(-1, 1)).flatten()
                # self.y_scalers[target] = ('yeo-johnson', scaler)
                scaler = QuantileTransformer(
                n_quantiles=min(1000, len(y)),
                output_distribution='normal',  # 
                random_state=42
                )
                y_normalized[:, i] = scaler.fit_transform(y[:, i].reshape(-1, 1)).flatten()
                self.y_scalers[target] = ('quantile', scaler)
            # 对Reff使用对数变换 + 标准化（处理大范围正值）
            elif target == 'Reff':
                # 确保所有值为正
                reff_data = np.maximum(y[:, i], 1e-12)  # 避免0或负值
                # 对数变换
                log_reff = np.log10(reff_data)
                # 标准化
                scaler = StandardScaler()
                y_normalized[:, i] = scaler.fit_transform(log_reff.reshape(-1, 1)).flatten()
                self.y_scalers[target] = ('log+standard', scaler, reff_data.min(), reff_data.max())
        
        # 打印处理后的统计
        print("\n=== 改进预处理后的目标范围 ===")
        for i, target in enumerate(self.output_targets):
            col = y_normalized[:, i]
            transform_type = self.y_scalers[target][0]
            print(f"  {target} ({transform_type}): [{col.min():.2f}, {col.max():.2f}], "
                  f"均值:{col.mean():.3f}, 标准差:{col.std():.3f}")
        
        return X_normalized, y_normalized
    
    def inverse_transform_y(self, y_normalized):
        y_original = np.zeros_like(y_normalized)
        
        for i, target in enumerate(self.output_targets):
            if target in self.y_scalers:
                transform_type, scaler = self.y_scalers[target][0], self.y_scalers[target][1]
                target_data = y_normalized[:, i].reshape(-1, 1)
                
                if transform_type == 'log+standard':
                    # 反标准化 -> 反对数
                    log_data = scaler.inverse_transform(target_data)
                    original_data = 10 ** log_data
                elif transform_type == 'multi+quantile':
                    original_data = scaler.inverse_transform(target_data)
                    original_data  = original_data /self.scale_factor
                else:  # standard
                    original_data = scaler.inverse_transform(target_data)
                
                y_original[:, i] = original_data.flatten()
        
        return y_original
    def preprocess_new_data(self, X_new):
        """
        对新数据进行相同的预处理
        """
        if self.X_scaler is None:
            raise ValueError("必须先调用load_and_preprocess_data方法")
        
        # 移除Line_space特征
        if self.removed_feature_indices:
            X_processed = np.delete(X_new, self.removed_feature_indices, axis=1)
        else:
            X_processed = X_new.copy()
        
        # 标准化
        X_normalized = self.X_scaler.transform(X_processed)
        
        return X_normalized
    
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
    
    # def preprocess_new_data(self, X_new):
    #     """
    #     对新数据进行相同的预处理
    #     X_new: 新数据，形状为(n_samples, n_features)，特征顺序应与原始input_features一致
    #     """
    #     if self.X_scaler is None:
    #         raise ValueError("必须先调用load_and_preprocess_data方法训练预处理器")
        
    #     # 移除Line_space特征
    #     if self.removed_feature_indices:
    #         X_processed = np.delete(X_new, self.removed_feature_indices, axis=1)
    #     else:
    #         X_processed = X_new.copy()
        
    #     # 标准化
    #     X_normalized = self.X_scaler.transform(X_processed)
        
    #     return X_normalized