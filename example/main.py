# main.py
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import TRAIN_CONFIG, DATA_CONFIG, DEVICE
from data_preprocessor import DataPreprocessor
from data_loader import DataLoaderCreator
from model import CnnNet, ResidualNet 
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
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def train_neural_network(model_type='advanced', create_report=True):
    """训练神经网络仿真器"""
    print("=== 训练神经网络仿真器 ===")
    print(f"使用模型类型: {model_type}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. 加载和预处理数据
    print("加载和预处理数据...")
    preprocessor = DataPreprocessor()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'cleaned_inductor_data.csv')
    file_path = os.path.normpath(file_path)
    print(f"reading: {file_path}")
    # X, y = preprocessor.load_and_preprocess_data(file_path)
    preprocessor.load_and_clean_data()
    all_matrices, all_frequencies, all_s_params, all_indices = preprocessor.get_all_data()
    print(f"数据形状:")
    print(f"  矩阵: {all_matrices.shape}")
    print(f"  频率: {all_frequencies.shape}")
    print(f"  S参数: {all_s_params.shape}")
    # 2. 使用划分数据
    print("\n划分数据...")
    
    split_data_dict = preprocessor.split_data(
        all_matrices, all_frequencies, all_s_params, all_indices,
        test_size=0.1, val_size=0.2, random_state=42
    )
    # 3. 分析数据分布
    print("\n" + "=" * 60)
    print("步骤3: 分析数据分布")
    print("=" * 60)
    
    preprocessor.analyze_data_distribution(split_data_dict)
    preprocessor.visualize_data_distribution(split_data_dict)
    # 4. 创建DataLoader
    print("\n" + "=" * 60)
    print("步骤4: 创建DataLoader")
    print("=" * 60)
    
    creator = DataLoaderCreator()
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = creator.create_data_loaders(
        split_data_dict, batch_size=TRAIN_CONFIG['batch_size'], num_workers=4
    )
    
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
    """
    preprocess among different dataset!!
    """
    # # 3. 只在训练集上拟合预处理器
    # print("\n在训练集上拟合预处理器...")
    # X_train_normalized, y_train_normalized = preprocessor.fit_preprocessors(X_train, y_train)
    
    # # 4. 使用训练集的预处理器变换验证集和测试集
    # print("\n变换验证集和测试集...")
    # X_val_normalized, y_val_normalized = preprocessor.transform_data(X_val, y_val)
    # X_test_normalized, y_test_normalized = preprocessor.transform_data(X_test, y_test)
    
    # 5. 构建神经网络
    print("\n" + "=" * 60)
    print("步骤5: 创建神经网络...")
    print("=" * 60)
    if model_type == 'residual':
        model = ResidualNet()
        print("使用残差网络")
    else:
        model = CnnNet()
        print("使用Cnn网络")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 6. 训练模型
    print("\n" + "=" * 60)
    print("步骤6: 创建神经网络...")
    print("=" * 60)
    model_manager = ModelManager(model, checkpoint_dir='./frequency_checkpoints')
    model_manager.setup_training()
    
    train_losses, val_losses = model_manager.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=TRAIN_CONFIG['epochs'],
        resume=TRAIN_CONFIG['resume_training'],
        print_every=20,  # 每20个epoch打印一次进度
        save_every=50    # 每50个epoch保存一次检查点
    )
    
    # 7. 评估模型性能
    print("\n在测试集上评估模型性能...")
    
    predictions, targets, frequencies, test_metrics = model_manager.evaluate_model(
        test_loader, dataset_name="测试集"
    )
    model_manager.print_evaluation_results(test_metrics)
    # 8. 详细误差分析
    sample_errors, all_predictions, all_targets = model_manager.evaluate_all_test_samples(test_loader)
    error_stats, min_sample, max_sample = model_manager.analyze_error_statistics(sample_errors)
    
    # 9. 创建误差分布图
    model_manager.create_error_distribution_plot(sample_errors, save_path='error_distribution.png')
    
    # 10. 保存最终模型
    model_manager.save_final_model('final_frequency_model.pth', model_info={
        'input_type': 'matrix_19x19 + frequency',
        'output_type': '8 S-parameters',
        'model_architecture': 'FrequencyAwareSParamCNN'
    })
    
    # 11. 创建训练总结
    # 需要先获取训练集的评估结果
    train_predictions, train_targets, train_freqs, train_metrics = model_manager.evaluate_model(
        train_loader, dataset_name="训练集"
    )
    model_manager.create_training_summary(train_losses, val_losses, train_metrics, test_metrics)
    


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
    
    elif mode == "2":
        # 仅运行遗传算法（需要已训练好的模型）
        try:
            # 加载已训练的模型
            model_path = f'checkpoints/inductor_checkpoints/checkpoint_epoch_999.pth'
            checkpoint = torch.load(model_path)
            print(checkpoint.keys())
            print("\n" + "="*50)
            # 根据保存的配置重建模型
            if 'model_config' in checkpoint:
                print("download")
                config = checkpoint['model_config']
                model = CnnNet(
                    input_size=config['input_size'],
                    output_size=config['output_size'],
                    hidden_dims=config['hidden_dims'],
                    dropout_rates=config['dropout_rates'],
                    use_batchnorm=config['use_batchnorm']
                )
            else:
                # 向后兼容
                print("undownload")
                model = CnnNet()
                
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