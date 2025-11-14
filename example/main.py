# main.py
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os

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
        test_loader, preprocessor.y_scaler, DATA_CONFIG['output_targets']
    )
    
    # 5. 创建可视化报告
    if create_report:
        print("\n生成训练报告和性能可视化...")
        
        visualizer = TrainingVisualizer()
        
        report_save_path = f'reports/inductor_model_{model_type}'
        os.makedirs('reports', exist_ok=True)
        
        visualizer.create_comprehensive_report(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,  # 使用测试集作为验证集进行可视化
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