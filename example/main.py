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
    
    # 2. 构建神经网络
    print("构建神经网络...")
    if model_type == 'residual':
        model = ResidualInductorNet()
        print("使用残差网络")
    else:
        model = InductorNet()
        print("使用高级深度网络")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 3. 训练模型
    model_manager = ModelManager(model)
    model_manager.setup_training()
    
    train_losses, val_losses = model_manager.train_model(
        train_loader, val_loader, 
        epochs=TRAIN_CONFIG['epochs'],
        resume=TRAIN_CONFIG['resume_training'],
        print_every=20,  # 每20个epoch打印一次进度
        save_every=50    # 每50个epoch保存一次检查点
    )
    
    # 4. 评估模型性能
    print("\n在测试集上评估模型性能...")
    
    predictions, targets, metrics_summary = model_manager.evaluate_model(
        preprocessor,test_loader, preprocessor.y_scaler, DATA_CONFIG['output_targets']
    )
    # 4. 进行三种精度评估
    print("\n=== 进行三种精度评估 ===")
    
    # 4.1 训练集精度评估
    print("1. 评估训练集精度...")
    train_predictions, train_targets, train_metrics = model_manager.evaluate_model(
        preprocessor,train_loader, preprocessor.y_scaler, DATA_CONFIG['output_targets'], "训练集"
    )
    
    # 4.2 测试集精度评估
    print("2. 评估测试集精度...")
    test_predictions, test_targets, test_metrics = model_manager.evaluate_model(
        preprocessor,test_loader, preprocessor.y_scaler, DATA_CONFIG['output_targets'], "测试集"
    )
    
    # 4.3 单个样本精度评估（使用修正后的方法）
    print("3. 评估单个样本精度...")
    
    # 使用测试集中的一个样本
    sample_idx = 109
    sample_input_original = preprocessor.X_scaler.inverse_transform([X_test[sample_idx]])[0]
    sample_target_normalized = y_test[sample_idx]
    for i in range(sample_idx,sample_idx+20):
        print(X_test[sample_idx+i])
        print("******************************************")
        print(sample_idx+i)
        print(y_test[sample_idx+i])
    # 使用修正后的单个样本评估函数
    sample_prediction_original, sample_target_original = model_manager.evaluate_single_sample_corrected(
        preprocessor.X_scaler, 
        preprocessor.y_scalers,  # 注意：这里传递的是y_scalers字典，不是y_scaler对象
        sample_input_original,
        sample_target_normalized,  # 传递归一化的真实值
        DATA_CONFIG['output_targets']
    )
    
    # 计算单个样本误差
    single_sample_metrics = {}
    for i, output_name in enumerate(DATA_CONFIG['output_targets']):
        true_val = sample_target_original[i]
        pred_val = sample_prediction_original[i]
        error_pct = abs(pred_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-12 else float('inf')
        single_sample_metrics[output_name] = (true_val, pred_val, error_pct)
    
    # 打印单个样本的详细信息
    print("\n单个样本详细信息:")
    print("输入参数 (原始尺度):")
    for j, feature in enumerate(['Line_Width', 'Turns','Y_Dimension', 'X_Dimension', 'freq']):  # 注意使用input_features_used
        print(f"  {feature}: {sample_input_original[j]:.6f}")
    
    print("\n预测结果对比:")
    for output_name, (true_val, pred_val, error_pct) in single_sample_metrics.items():
        print(f"  {output_name}:")
        print(f"    真实值: {true_val:.6e}")
        print(f"    预测值: {pred_val:.6e}")
        print(f"    误差: {error_pct:.2f}%")
    
    # 5. 打印精度总结
    model_manager.create_precision_summary(train_metrics, test_metrics, single_sample_metrics)
    # 5. 创建可视化报告
    if create_report:
        print("\n生成训练报告和性能可视化...")
        
        visualizer = TrainingVisualizer()
        
        report_save_path = f'reports/inductor_model_{model_type}'
        os.makedirs('reports', exist_ok=True)
        
        visualizer.create_comprehensive_report(
            preprocessor = preprocessor,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,  # 使用测试集作为验证集进行可视化
            y_scaler=preprocessor.y_scaler,
            output_names=DATA_CONFIG['output_targets'],
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=report_save_path
        )
    
    # 6. 保存模型
    final_model_path = f'models/inductor_model_{model_type}_final.pth'
    os.makedirs('models', exist_ok=True)
    
    model_manager.save_final_model(
        final_model_path, 
        preprocessor.X_scaler, 
        preprocessor.y_scaler,
        train_losses, val_losses
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