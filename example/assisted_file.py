###########################################################################################
#main.py
############################################################################################
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import TRAIN_CONFIG, DATA_CONFIG, DEVICE
from data_preprocessor import DataPreprocessor
from data_loader import InductorDataset
from model import InductorNet,ResidualInductorNet 
from train import ModelManager
from optimization_algorithm import GeneticAlgorithm
from utils import TrainingVisualizer

def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_neural_network(model_type='advanced', create_report=True):
    """训练神经网络仿真器"""
    print("=== 训练神经网络仿真器 ===")
    print(f"使用模型类型: {model_type}")
    
    # 1. 加载和预处理数据
    print("加载和预处理数据...")
    preprocessor = DataPreprocessor()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'Diff_SQ_XFAB_MJ_All_copy.csv')
    file_path = os.path.normpath(file_path)
    print(f"reading: {file_path}")
    X, y = preprocessor.load_and_preprocess_data(file_path)
    # X_train, X_test,X_val, y_train, y_test, y_val = preprocessor.split_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    # 创建数据加载器
    train_dataset = InductorDataset(X_train, y_train)
    val_dataset = InductorDataset(X_val, y_val)
    test_dataset = InductorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    
    # 修改评估方法
    def evaluate_model_improved(model_manager, data_loader, preprocessor, output_names):
        """改进的评估方法"""
        model_manager.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                output = model_manager.model(batch_X)
                predictions.append(output.numpy())
                targets.append(batch_y.numpy())
        
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        
        # 使用预处理器的逆变换
        predictions_original = preprocessor.inverse_transform_y(predictions)
        targets_original = preprocessor.inverse_transform_y(targets)
        
        # 计算指标
        metrics_summary = {}
        for i, name in enumerate(output_names):
            pred = predictions_original[:, i]
            true = targets_original[:, i]
            
            mse = mean_squared_error(true, pred)
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)
            
            # 计算相对误差百分比（使用绝对值和一个小常数避免除零）
            relative_errors = np.abs((pred - true) / (np.abs(true) + 1e-12)) * 100
            mean_relative_error = np.mean(relative_errors)
            median_relative_error = np.median(relative_errors)
            
            metrics_summary[name] = {
                'mse': mse,
                'mae': mae, 
                'r2': r2,
                'mean_relative_error': mean_relative_error,
                'median_relative_error': median_relative_error
            }
            
            print(f"{name}: MSE={mse:.2e}, MAE={mae:.2e}, R²={r2:.4f}, "
                  f"平均相对误差={mean_relative_error:.2f}%, 中位数相对误差={median_relative_error:.2f}%")
        
        return predictions_original, targets_original, metrics_summary
    
    # 训练和评估
    model_manager = ModelManager(model)
    model_manager.setup_training()
    
    # 训练模型
    train_losses, val_losses = model_manager.train_model(
        train_loader, val_loader,
        epochs=TRAIN_CONFIG['epochs'],
        resume=TRAIN_CONFIG['resume_training'],
        print_every=20,
        save_every=50
    )
    
    # 使用改进的评估方法
    predictions, targets, metrics_summary = evaluate_model_improved(
        model_manager, test_loader, preprocessor, DATA_CONFIG['output_targets']
    )
    
    return model, preprocessor.X_scaler, preprocessor.y_scaler, metrics_summary

def run_genetic_optimization(model, X_scaler, y_scaler):
    """运行遗传算法优化"""
    print("\n=== 运行遗传算法优化 ===")
    
    # 创建遗传算法优化器
    ga = GeneticAlgorithm(model, X_scaler, y_scaler)
    
    # 设置设计目标
    target_freq = 4.0  # GHz
    target_Leff = 2e-9  # 2nH
    
    # 运行优化
    best_params, best_performance, best_fitness = ga.optimize(
        target_freq=target_freq,
        target_Leff=target_Leff
    )
    
    # 输出优化结果
    print("\n=== 优化结果 ===")
    print(f"目标: Leff = {target_Leff:.2e} at {target_freq} GHz")
    print(f"最佳适应度: {best_fitness:.4f}")
    print("\n最佳参数:")
    for i, param_name in enumerate(ga.input_features):
        print(f"  {param_name}: {best_params[i]}")
    
    print("\n预测性能:")
    for i, target_name in enumerate(ga.output_targets):
        print(f"  {target_name}: {best_performance[i]:.2e}")
    
    return best_params, best_performance

if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
    # 选择模型类型
    model_type = input("选择模型类型 (1: AdvancedDNN, 2: ResidualNet): ").strip()
    if model_type == "2":
        model_type_name = "residual"
    else:
        model_type_name = "advanced"
    
    # 选择要运行的模式
    mode = input("选择运行模式 (1: 仅训练神经网络, 2: 仅运行遗传算法, 3: 完整流程): ")
    
    if mode == "1":
        # 仅训练神经网络
        model, X_scaler, y_scaler, metrics = train_neural_network(model_type_name, create_report=True)
        
        # 打印性能总结
        print("\n" + "="*80)
        print("最终性能总结")
        print("="*80)
        for output_name, metric_dict in metrics.items():
            print(f"{output_name}: R²={metric_dict['R²']:.4f}, MAPE={metric_dict['MAPE']:.2f}%")
    
    elif mode == "2":
        # 仅运行遗传算法（需要已训练好的模型）
        try:
            # 加载已训练的模型
            model_path = f'models/inductor_model_{model_type_name}_final.pth'
            checkpoint = torch.load(model_path)
            
            # 根据保存的配置重建模型
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                model = InductorNet(
                    input_size=config['input_size'],
                    output_size=config['output_size'],
                    hidden_dims=config['hidden_dims'],
                    dropout_rates=config['dropout_rates'],
                    use_batchnorm=config['use_batchnorm']
                )
            else:
                # 向后兼容
                model = InductorNet()
                
            model.load_state_dict(checkpoint['model_state_dict'])
            X_scaler = checkpoint['X_scaler']
            y_scaler = checkpoint['y_scaler']
            
            run_genetic_optimization(model, X_scaler, y_scaler)
        except FileNotFoundError:
            print(f"错误: 未找到训练好的模型 {model_path}，请先运行模式1训练神经网络")
    
    elif mode == "3":
        # 完整流程：训练神经网络 + 遗传算法优化
        model, X_scaler, y_scaler, metrics = train_neural_network(model_type_name, create_report=True)
        run_genetic_optimization(model, X_scaler, y_scaler)
    
    else:
        print("无效的选择，请输入1、2或3")
###########################################################################################
#data_preprocessor.py
############################################################################################
# data_loader/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG


class DataPreprocessor:
    def __init__(self):
        self.input_features = DATA_CONFIG['input_features']
        self.output_targets = DATA_CONFIG['output_targets']
        self.X_scaler = None
        # self.y_scaler = None
        self.y_transformer = None
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
        # self.y_scaler = StandardScaler()
        
        X_normalized = self.X_scaler.fit_transform(X)
        print("\nYeo-JohnsonTrans...")
        self.y_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        # y_normalized = self.y_scaler.fit_transform(y)
        y_transformed = self.y_transformer.fit_transform(y)
        # 打印处理后的特征范围
        print("\n=== 标准化后的特征范围 ===")
        for i, feature in enumerate(self.input_features):#self.input_features_used
            print(f"  {feature}: [{X_normalized[:, i].min():.2f}, {X_normalized[:, i].max():.2f}]")
        for i, target in enumerate(self.output_targets):
            print(f"  {target}: [{y_transformed[:, i].min():.2f}, {y_transformed[:, i].max():.2f}]")
        return X_normalized, y_transformed
    def inverse_transform_y(self, y_transformed):
        """
        将变换后的预测值转换回原始尺度
        """
        if self.y_transformer is None:
            raise ValueError("必须先调用load_and_preprocess_data方法")
        
        return self.y_transformer.inverse_transform(y_transformed)
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