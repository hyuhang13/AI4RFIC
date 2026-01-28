# main.py
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os
import sys
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List, Tuple, Dict, Optional, Callable
from config import TRAIN_CONFIG, DATA_CONFIG, DEVICE
from data_preprocessor import DataPreprocessor
from data_loader import DataLoaderCreator
from model import CnnNet, ResidualNet 
from train import ModelManager
from optimization_algorithm import GeneticAlgorithm
from utils import TrainingVisualizer
from log import LoggerWriter
import datetime
from dataset_io import DatasetSaver
from dataset_package import MultiModalDataset
from config import TARGET_CONFIG
from s_param_visualizer import SParamVisualizer
# 设置日志记录
def setup_logging(log_file='training_log.txt'):
    """设置日志记录，同时输出到控制台和文件"""
    # 创建日志目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/training_{timestamp}.log"
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的handler
    logger.handlers.clear()
    
    # 创建控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 记录日志文件路径
    logger.info(f"日志文件保存到: {log_filename}")
    
    return log_filename

# 创建自定义的print函数，同时打印到控制台和日志
def log_print(message, level=logging.INFO):
    """自定义打印函数，同时输出到控制台和日志"""
    logger = logging.getLogger()
    
    if level == logging.INFO:
        logger.info(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    elif level == logging.DEBUG:
        logger.debug(message)
    else:
        logger.info(message)
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
def print_model_first_10_params(model):
    """打印模型前10个参数值"""
    print("="*80)
    print("模型参数前10个值检查")
    print("="*80)
    
    all_params = []
    
    # 收集所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data.cpu().numpy().flatten()
            all_params.extend(data)
    
    print(f"模型总参数数量: {len(all_params):,}")
    
    if len(all_params) >= 50:
        print(f"前10个参数值:")
        for i, val in enumerate(all_params[:50]):
            print(f"  参数[{i}] = {val:.8f}")
        
        print(f"\n参数统计:")
        print(f"  最小值: {min(all_params[:50]):.8f}")
        print(f"  最大值: {max(all_params[:50]):.8f}")
        print(f"  平均值: {np.mean(all_params[:50]):.8f}")
        print(f"  标准差: {np.std(all_params[:50]):.8f}")
    else:
        print(f"模型参数少于10个: {len(all_params)}个")
def train_neural_network(model_type='advanced', create_report=True,use_cached_dataset=True, dataset_version='v1'):
    """训练神经网络仿真器"""
    log_print("=== 训练神经网络仿真器 ===")
    log_print(f"使用模型类型: {model_type}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrix_file_path = 'binary_matrices.txt'
    dataset_file_path = 'dataset/dataset_agg_cleaned.csv'
    # 定义数据集保存路径
    save_dir = '/root/aicp-data/saved_datasets'
    os.makedirs(save_dir, exist_ok=True)
    dataset_save_dir = f'/root/aicp-data/saved_datasets/{dataset_version}'
    if use_cached_dataset and os.path.exists(dataset_save_dir):
        log_print(f"使用缓存的预处理数据集: {dataset_save_dir}")
        
        # 直接加载保存的数据集
        data_loaders, datasets, metadata = DatasetSaver.load_dataset(
            load_dir=dataset_save_dir,
            dataset_class=MultiModalDataset
        )
        
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        test_loader = data_loaders['test_loader']
        
        train_dataset = datasets['train_dataset']
        val_dataset = datasets['val_dataset']
        test_dataset = datasets['test_dataset']
        
    else:
        log_print("执行完整的数据预处理流程...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        matrix_file_path = 'binary_matrices.txt'
        dataset_file_path = 'dataset/dataset_agg_cleaned.csv'
        
        # 1. 加载和预处理数据
        log_print("加载和预处理数据...")
        preprocessor = DataPreprocessor(matrix_file_path, dataset_file_path)
        preprocessor.load_and_clean_data()
        all_matrices, all_frequencies, all_s_params, all_indices = preprocessor.get_all_data()

        # 2. 划分数据
        log_print("\n划分数据...")
        split_data_dict = preprocessor.split_data(
            all_matrices, all_frequencies, all_s_params, all_indices,
            test_size=0.1, val_size=0.2, random_state=42
        )
        
        # 3. 分析数据分布
        log_print("\n" + "=" * 60)
        log_print("分析数据分布")
        log_print("=" * 60)
        preprocessor.analyze_data_distribution(split_data_dict)
        preprocessor.visualize_data_distribution(split_data_dict)
        
        # 4. 创建DataLoader
        log_print("\n" + "=" * 60)
        log_print("创建DataLoader")
        log_print("=" * 60)
        creator = DataLoaderCreator()
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = creator.create_data_loaders(
            split_data_dict, batch_size=TRAIN_CONFIG['batch_size'], num_workers=4
        )
        
        # 5. 保存数据集以便后续使用
        log_print("\n" + "=" * 60)
        log_print("保存预处理数据集")
        log_print("=" * 60)
        data_loader_dict = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }
        
        dataset_dict = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }
        
        DatasetSaver.save_dataset(data_loader_dict, dataset_dict, dataset_save_dir)
    # # 1. 加载和预处理数据
    # log_print("加载和预处理数据...")
    # preprocessor = DataPreprocessor(matrix_file_path,dataset_file_path)
    # # current_dir = os.path.dirname(os.path.abspath(__file__))
    # # file_path = os.path.join(current_dir, '..', 'cleaned_inductor_data.csv')
    # # file_path = os.path.normpath(file_path)
    # # log_print(f"reading: {file_path}")
    # preprocessor.load_and_clean_data()
    # all_matrices, all_frequencies, all_s_params, all_indices = preprocessor.get_all_data()

    # # 2. 使用划分数据
    # log_print("\n划分数据...")
    
    # split_data_dict = preprocessor.split_data(
    #     all_matrices, all_frequencies, all_s_params, all_indices,
    #     test_size=0.1, val_size=0.2, random_state=42
    # )
    # # 3. 分析数据分布
    # log_print("\n" + "=" * 60)
    # log_print("步骤3: 分析数据分布")
    # log_print("=" * 60)
    
    # preprocessor.analyze_data_distribution(split_data_dict)
    # preprocessor.visualize_data_distribution(split_data_dict)
    # # 4. 创建DataLoader
    # log_print("\n" + "=" * 60)
    # log_print("步骤4: 创建DataLoader")
    # log_print("=" * 60)
    
    # creator = DataLoaderCreator()
    # train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = creator.create_data_loaders(
    #     split_data_dict, batch_size=TRAIN_CONFIG['batch_size'], num_workers=4
    # )
    log_print(f"训练集批次数量: {len(train_loader)}")
    log_print(f"验证集批次数量: {len(val_loader)}")
    log_print(f"测试集批次数量: {len(test_loader)}")
    """
    preprocess among different dataset!!
    """
    # # 3. 只在训练集上拟合预处理器
    # log_print("\n在训练集上拟合预处理器...")
    # X_train_normalized, y_train_normalized = preprocessor.fit_preprocessors(X_train, y_train)
    
    # # 4. 使用训练集的预处理器变换验证集和测试集
    # log_print("\n变换验证集和测试集...")
    # X_val_normalized, y_val_normalized = preprocessor.transform_data(X_val, y_val)
    # X_test_normalized, y_test_normalized = preprocessor.transform_data(X_test, y_test)
    
    # 5. 构建神经网络
    log_print("\n" + "=" * 60)
    log_print("步骤5: 创建神经网络...")
    log_print("=" * 60)
    if model_type == 'residual':
        model = ResidualNet()
        log_print("使用残差网络")
    else:
        model = CnnNet()
        log_print("使用Cnn网络")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"模型总参数: {total_params:,}")
    log_print(f"可训练参数: {trainable_params:,}")
    
    # 6. 训练模型
    log_print("\n" + "=" * 60)
    log_print("步骤6: 训练...")
    log_print("=" * 60)
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
    log_print("\n" + "=" * 60)
    log_print("步骤7: 在测试集上评估模型性能...")
    log_print("=" * 60)
    print_model_first_10_params(model_manager.model)
    predictions, targets, frequencies, test_metrics = model_manager.evaluate_model(
        test_loader, dataset_name="测试集"
    )

    # 1. 绘制训练历史
    model_manager.plot_training_history(train_losses, val_losses, 
                                       save_path='plots/training_history.png')
    print("???????????????????????????????????1")
    # 2. 绘制散点对比图
    # model_manager.plot_scatter_comparison(predictions, targets, frequencies,
    #                                      dataset_name="Test Set",
    #                                      save_path='plots/Test_loader_scatter_comparison.png')
    # print("???????????????????????????????????3")
    # 3. 绘制dB对比图（新增）
    model_manager.plot_db_comparison(predictions, targets, frequencies,
                                    dataset_name="Test Set",
                                    save_path='plots/Test_loader_db_comparison.png')
    print("???????????????????????????????????dB对比图生成完成")
    # 4. 绘制残差分布图
    model_manager.plot_residual_distribution(predictions, targets,
                                           save_path='plots/Test_loader_residual_distribution.png')
    
    # 5. 按频率分析
    model_manager.plot_frequency_analysis(predictions, targets, frequencies,
                                                         save_path='plots/Test_loader_frequency_analysis.png')
    print("频率分析图生成完成")
    
    # # 6. 综合可视化（一次性生成所有图表）
    # figures = model_manager.create_comprehensive_visualization(
    #     train_losses, val_losses, predictions, targets, frequencies,
    #     output_dir='plots/Test_loader_comprehensive_report'
    # )

    model_manager.print_evaluation_results(test_metrics)

    # 7. 详细误差分析
    # sample_errors, all_predictions, all_targets = model_manager.evaluate_all_test_samples(test_loader)
    # linear_error_stats, db_error_stats, min_sample, max_sample = model_manager.analyze_error_statistics(sample_errors)
    
    # # 8. 创建误差分布图
    # model_manager.create_error_distribution_plot(sample_errors, save_path='error_distribution.png')
    # # 9. 创建dB性能报告
    # model_manager.create_db_performance_report(predictions, targets, output_dir='plots')
    # print("dB性能报告生成完成")

    # 10. 保存最终模型
    model_manager.save_final_model('final_frequency_model.pth', model_info={
        'input_type': 'matrix_19x19 + frequency',
        'output_type': '8 S-parameters',
        'model_architecture': 'FrequencyAwareSParamCNN'
    })
    # print("##############\n")

    # 需要先获取训练集的评估结果
    log_print("\n" + "=" * 60)
    log_print("步骤8: 在训练集上评估模型性能...")
    log_print("=" * 60)
    # print_model_first_10_params(model_manager.model)
    predictions, targets, frequencies, test_metrics = model_manager.evaluate_model(
        train_loader, dataset_name="训练集"
    )

    # 1. 绘制训练历史
    # model_manager.plot_training_history(train_losses, val_losses, 
    #                                    save_path='plots/Train_loader_training_history.png')
    print("???????????????????????????????????1")
    # 2. 绘制散点对比图
    # model_manager.plot_scatter_comparison(predictions, targets, frequencies,
    #                                      dataset_name="Test Set",
    #                                      save_path='plots/Train_loader_scatter_comparison.png')
    # print("???????????????????????????????????3")
    # 3. 绘制dB对比图（新增）
    model_manager.plot_db_comparison(predictions, targets, frequencies,
                                    dataset_name="Test Set",
                                    save_path='plots/Train_loader_db_comparison.png')
    print("???????????????????????????????????dB对比图生成完成")
    # # 4. 绘制残差分布图
    # model_manager.plot_residual_distribution(predictions, targets,
    #                                        save_path='plots/Train_loader_residual_distribution.png')
    
    # 5. 按频率分析
    model_manager.plot_frequency_analysis(predictions, targets, frequencies,
                                                         save_path='plots/Train_loader_frequency_analysis.png')
    print("频率分析图生成完成")
    
    # # 6. 综合可视化（一次性生成所有图表）
    # figures = model_manager.create_comprehensive_visualization(
    #     train_losses, val_losses, predictions, targets, frequencies,
    #     output_dir='plots/Train_loader_comprehensive_report'
    # )

    model_manager.print_evaluation_results(test_metrics)
    
def run_genetic_optimization(model_manager, target_freq=15.0, 
                                   target_gamma_opt = 1,
                                   target_s_params = None,
                                   population_size=1024, generations=100):
    """
    运行矩阵遗传算法优化的主函数接口
    """
    log_print(f"\n{'='*60}")
    log_print("启动矩阵结构遗传优化")
    log_print(f"{'='*60}")
    # 3. 直接查看特定层的权重值
    print("********************************")
    print_model_first_10_params(model_manager.model)
    # 创建遗传算法实例
    ga = GeneticAlgorithm(model_manager, matrix_shape=(19, 19))
    
    # 设置目标参数
    target_params = {
        'gamma_opt': {
            'origin_value': target_gamma_opt,  # 目标S11幅度(dB)
            'magnitude_db': target_gamma_opt,  # 目标S11幅度(dB)
            'weight': 1.0  # 权重
        },
        's11_real': target_s_params['s11_real'],
        's11_imag': target_s_params['s11_imag'],
        's21_real': target_s_params['s21_real'],
        's21_imag': target_s_params['s21_imag'],
        's12_real': target_s_params['s12_real'],
        's12_imag': target_s_params['s12_imag'],
        's22_real': target_s_params['s22_real'],
        's22_imag': target_s_params['s22_imag'],
        'S21': {
            'magnitude_db':0,
            'weight':0
        },
        'S11': {
            'magnitude_db':0,
            'weight':0
        },
        'S22': {
            'magnitude_db':0,
            'weight':0
        },
        'symmetry': {
            'weight': 0.3  # 对称性权重
        },
        'complexity': {
            'weight': 0.1  # 复杂度权重
        }
    }
    
    # 运行优化
    best_matrix, best_info, best_fitness, fitness_history, best_fitness_history = ga.optimize(
        target_freq=target_freq,
        target_params=target_params,
        population_size=population_size,
        generations=generations,
        verbose=True
    )
    # ga.print_matrix(best_matrix)
    # 可视化结果
    ga.visualize_results(best_matrix, fitness_history, best_fitness_history)
    
    # 保存结果
    save_optimization_results(best_matrix, best_info, best_fitness, target_freq)
    
    return best_matrix, best_info, best_fitness

def save_optimization_results(matrix: np.ndarray, info: Dict, 
                            fitness: float, target_freq: float):
    """保存优化结果"""
    import pickle
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'optimized_matrix': matrix,
        'fitness': fitness,
        'target_frequency_ghz': target_freq,
        's_params': {
            'S11_real': info['s_params']['S11_real'],
            'S11_imag': info['s_params']['S11_imag'],
            'S21_real': info['s_params']['S21_real'],
            'S21_imag': info['s_params']['S21_imag'],
            'S12_real': info['s_params']['S12_real'],
            'S12_imag': info['s_params']['S12_imag'],
            'S22_real': info['s_params']['S22_real'],
            'S22_imag': info['s_params']['S22_imag']
        },
        # 's_parameters': {
        #     'S11_mag_db': info['S11_mag_db'],
        #     'S21_mag_db': info['S21_mag_db'],
        #     'S12_mag_db': info['S12_mag_db'],
        #     'S22_mag_db': info['S22_mag_db'],
        #     'S11_phase_deg': info['S11_phase_deg'],
        #     'S21_phase_deg': info['S21_phase_deg'],
        # },
        # 'raw_s_params': info['S_params_raw'],
        'matrix_density': info['matrix_density'],
        'timestamp': timestamp,
        'fitness_components': info.get('fitness_components', {})
    }
    print('预测结果：')
    print(results)
    # # 保存为pickle文件
    # results_file = f'optimization_results_{timestamp}.pkl'
    # with open(results_file, 'wb') as f:
    #     pickle.dump(results, f)
    
    # 保存矩阵为文本文件
    matrix_file = f'optimized_matrix_{timestamp}.txt'
    np.savetxt(matrix_file, matrix, fmt='%d', delimiter=',')
    
    # 保存矩阵为图像
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='binary', interpolation='nearest')
    plt.title(f'Optimized Matrix @ {target_freq} GHz')
    plt.colorbar(label='Value (0/1)')
    plt.savefig(f'optimized_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    log_print(f"\n优化结果已保存:")
    # log_print(f"  - 数据文件: {results_file}")
    log_print(f"  - 矩阵文件: {matrix_file}")
    log_print(f"  - 图像文件: optimized_matrix_{timestamp}.png")


# 使用示例函数
def visualize_s_param_performance(model_manager, matrix, ground_truth_file, 
                                 output_dir='s_param_evaluation'):
    """
    方便的使用函数
    
    Args:
        model_manager: 模型管理器
        matrix: 19x19矩阵
        ground_truth_file: 真实数据文件路径
        output_dir: 输出目录
        
    Returns:
        可视化结果
    """
    # 创建可视化器
    visualizer = SParamVisualizer(model_manager, matrix)
    
    # 执行完整可视化
    results = visualizer.visualize_all(ground_truth_file, output_dir)
    
    return results
if __name__ == "__main__":
    # 设置日志记录
    log_filename = setup_logging()
    
    # 重定向标准输出到日志
    sys.stdout = LoggerWriter(log_print)
    sys.stderr = LoggerWriter(lambda msg: log_print(msg, level=logging.ERROR))
    # 设置随机种子
    set_seed(42)
    
    # 选择模型类型
    model_type = input("选择模型类型 (1: CnnNet, 2: ResidualNet): ").strip()
    if model_type == "2":
        model_type_name = "residual"
    else:
        model_type_name = "advanced"
    
    # 选择要运行的模式
    mode = input("选择运行模式 (1: 仅训练神经网络, 2: 仅运行遗传算法, 3: 测试S参数曲线: ")
    
    if mode == "1":
        # 仅训练神经网络
        train_neural_network(model_type_name, create_report=True,use_cached_dataset=True, dataset_version='v1')
    
    elif mode == "2":
        # 仅运行遗传算法（需要已训练好的模型）
        try:
            # 加载已训练的模型
            model_path = f'frequency_checkpoints/best_model.pth'
            checkpoint = torch.load(model_path)
            # log_print(checkpoint.keys())
            log_print("\n" + "="*50)
            log_print("创建空白模型：")
            log_print("\n" + "="*50)
            model = CnnNet()
            
            log_print("\n" + "="*50)
            log_print("模型加载：")
            log_print("\n" + "="*50)
            model.load_state_dict(checkpoint)
            model = model.to(DEVICE)
            manager = ModelManager(model, checkpoint_dir=model_path)
            # manager.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            # best_matrix, best_prediction, best_fitness = run_genetic_optimization(model,TARGET_CONFIG['target_freq',])
            best_matrix, best_info, best_fitness = run_genetic_optimization(manager, target_freq=TARGET_CONFIG['target_freq'], 
                                   target_gamma_opt = TARGET_CONFIG['target_gamma_opt'],
                                   target_s_params = TARGET_CONFIG['target_s_params'],
                                   population_size=1024, 
                                   generations=100)

        except FileNotFoundError:
            log_print(f"错误: 未找到训练好的模型 {model_path}，请先运行模式1训练神经网络")
    
    elif mode == "3":
        # 完整流程：训练神经网络 + 遗传算法优化
        model_path = f'frequency_checkpoints/best_model.pth'
        checkpoint = torch.load(model_path)
        # log_print(checkpoint.keys())
        log_print("\n" + "="*50)
        log_print("创建空白模型：")
        log_print("\n" + "="*50)
        model = CnnNet()
        
        log_print("\n" + "="*50)
        log_print("模型加载：")
        log_print("\n" + "="*50)
        model.load_state_dict(checkpoint)
        model = model.to(DEVICE)
        manager = ModelManager(model, checkpoint_dir=model_path)
        # # 定义矩阵（示例：随机矩阵）
        test_matrix = [[0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1], [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1]]

        # test_matrix = np.random.randint(
        #         low=0,          
        #         high=2,         
        #         size=(19, 19),  
        #         dtype=np.int32
        #     )
        print(test_matrix)
        test_matrix[9][0]=1
        test_matrix[9][18] = 1

        # 真实数据文件
        ground_truth_file = 's_params_varification.txt'
        
        # 运行可视化
        results = visualize_s_param_performance(
            manager, test_matrix, ground_truth_file, 
            output_dir='s_param_evaluation_results'
        )
    
    else:
        log_print("无效的选择，请输入1、2或3")
    # 恢复标准输出
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
        
    log_print(f"\n程序运行完成，日志已保存到: {log_filename}")
    log_print("=" * 60)