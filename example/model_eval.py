# model_evaluation.py
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessor import DataPreprocessor
from model import InductorNet
from config import DATA_CONFIG

class ModelEvaluator:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.device = next(model.parameters()).device
        
    def evaluate_dataset(self, X, y, dataset_name="Dataset"):
        """评估特定数据集"""
        self.model.eval()
        
        # 转换为Tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions_normalized = self.model(X_tensor)
        
        # 反归一化
        predictions = self.preprocessor.y_scaler.inverse_transform(
            predictions_normalized.cpu().numpy()
        )
        targets = self.preprocessor.y_scaler.inverse_transform(y)
        
        # 计算指标
        metrics = self.calculate_metrics(predictions, targets, dataset_name)
        
        return predictions, targets, metrics
    
    def calculate_metrics(self, predictions, targets, dataset_name):
        """计算评估指标"""
        output_names = DATA_CONFIG['output_targets']
        metrics = {
            'dataset': dataset_name,
            'overall': {},
            'per_output': {}
        }
        
        # 整体指标
        metrics['overall']['R2'] = r2_score(targets, predictions)
        metrics['overall']['MAE'] = mean_absolute_error(targets, predictions)
        metrics['overall']['RMSE'] = np.sqrt(mean_squared_error(targets, predictions))
        metrics['overall']['MSE'] = mean_squared_error(targets, predictions)
        
        # 每个输出的指标
        for i, output_name in enumerate(output_names):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            metrics['per_output'][output_name] = {
                'R2': r2_score(target_i, pred_i),
                'MAE': mean_absolute_error(target_i, pred_i),
                'RMSE': np.sqrt(mean_squared_error(target_i, pred_i)),
                'MSE': mean_squared_error(target_i, pred_i),
                'Mean_Relative_Error': np.mean(np.abs((pred_i - target_i) / (target_i + 1e-12))) * 100
            }
        
        return metrics
    
    def generate_random_data(self, num_samples=1000):
        """生成随机测试数据（在训练数据范围内）"""
        # 获取训练数据的原始范围
        X_train_original = self.preprocessor.X_scaler.inverse_transform(self.preprocessor.X_train)
        
        random_data = []
        for i in range(len(DATA_CONFIG['input_features'])):
            min_val = X_train_original[:, i].min()
            max_val = X_train_original[:, i].max()
            # 在训练数据范围内随机生成
            feature_data = np.random.uniform(min_val, max_val, num_samples)
            random_data.append(feature_data)
        
        random_X_original = np.column_stack(random_data)
        # 归一化
        random_X_normalized = self.preprocessor.X_scaler.transform(random_X_original)
        
        return random_X_normalized, random_X_original
    
    def evaluate_random_data(self, num_samples=1000):
        """评估随机生成的数据"""
        random_X_normalized, random_X_original = self.generate_random_data(num_samples)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            random_predictions_normalized = self.model(
                torch.FloatTensor(random_X_normalized).to(self.device)
            )
        
        # 反归一化预测结果
        random_predictions = self.preprocessor.y_scaler.inverse_transform(
            random_predictions_normalized.cpu().numpy()
        )
        
        print(f"\n随机生成 {num_samples} 个样本的统计信息:")
        print("输入参数范围 (原始尺度):")
        for i, feature in enumerate(DATA_CONFIG['input_features']):
            min_val = random_X_original[:, i].min()
            max_val = random_X_original[:, i].max()
            print(f"  {feature}: [{min_val:.2f}, {max_val:.2f}]")
        
        print("\n预测输出范围:")
        for i, output_name in enumerate(DATA_CONFIG['output_targets']):
            min_val = random_predictions[:, i].min()
            max_val = random_predictions[:, i].max()
            mean_val = random_predictions[:, i].mean()
            print(f"  {output_name}: 范围[{min_val:.2e}, {max_val:.2e}], 均值{mean_val:.2e}")
        
        return random_predictions, random_X_original

def load_model_and_data(checkpoint_path):
    """加载模型和数据"""
    # 加载预处理器和数据
    preprocessor = DataPreprocessor()
    X, y = preprocessor.load_and_preprocess_data("Diff_SQ_XFAB_MJ_All_copy.csv")
    
    # 分割数据（与训练时一致）
    from torch.utils.data import TensorDataset, random_split
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 获取训练集和验证集数据
    X_train, y_train = dataset[train_dataset.indices]
    X_val, y_val = dataset[val_dataset.indices]
    
    # 加载模型
    model = InductorNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, preprocessor, X_train.numpy(), y_train.numpy(), X_val.numpy(), y_val.numpy()

def print_metrics(metrics):
    """打印评估指标"""
    print(f"\n{'='*60}")
    print(f"{metrics['dataset']} 评估结果")
    print(f"{'='*60}")
    
    print("\n整体指标:")
    for metric, value in metrics['overall'].items():
        print(f"  {metric}: {value:.6f}")
    
    print("\n各输出指标:")
    for output_name, output_metrics in metrics['per_output'].items():
        print(f"\n  {output_name}:")
        for metric, value in output_metrics.items():
            if metric == 'Mean_Relative_Error':
                print(f"    {metric}: {value:.2f}%")
            else:
                print(f"    {metric}: {value:.6f}")

def main():
    # 配置
    CHECKPOINT_PATH = "checkpoints/inductor_checkpoints/checkpoint_epoch_499.pth"
    
    print("=== 神经网络模型综合评估 ===")
    
    # 1. 加载模型和数据
    print("加载模型和数据...")
    model, preprocessor, X_train, y_train, X_val, y_val = load_model_and_data(CHECKPOINT_PATH)
    
    # 2. 创建评估器
    evaluator = ModelEvaluator(model, preprocessor)
    
    # 3. 评估训练集
    print("\n评估训练集...")
    train_predictions, train_targets, train_metrics = evaluator.evaluate_dataset(
        X_train, y_train, "训练集"
    )
    print_metrics(train_metrics)
    
    # 4. 评估验证集
    print("\n评估验证集...")
    val_predictions, val_targets, val_metrics = evaluator.evaluate_dataset(
        X_val, y_val, "验证集"
    )
    print_metrics(val_metrics)
    
    # 5. 评估随机生成数据
    print("\n评估随机生成数据...")
    random_predictions, random_inputs = evaluator.evaluate_random_data(1000)
    
    # 6. 对比分析
    print(f"\n{'='*60}")
    print("性能对比分析")
    print(f"{'='*60}")
    
    comparison_data = []
    datasets = [
        ("训练集", train_metrics),
        ("验证集", val_metrics)
    ]
    
    for dataset_name, metrics in datasets:
        comparison_data.append({
            'Dataset': dataset_name,
            'R2_Score': metrics['overall']['R2'],
            'MAE': metrics['overall']['MAE'],
            'RMSE': metrics['overall']['RMSE']
        })
    
    # 创建对比表格
    comparison_df = pd.DataFrame(comparison_data)
    print("\n数据集对比:")
    print(comparison_df.to_string(index=False))
    
    # 7. 可视化结果
    create_comparison_plots(train_predictions, train_targets, 
                           val_predictions, val_targets, 
                           random_predictions)

def create_comparison_plots(train_pred, train_true, val_pred, val_true, random_pred):
    """创建对比可视化"""
    output_names = DATA_CONFIG['output_targets']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('模型性能综合评估', fontsize=16, fontweight='bold')
    
    # 预测vs真实值对比
    for i, (ax, output_name) in enumerate(zip(axes[0, :], output_names)):
        # 训练集
        ax.scatter(train_true[:, i], train_pred[:, i], alpha=0.6, s=20, 
                  label='训练集', color='blue')
        # 验证集
        ax.scatter(val_true[:, i], val_pred[:, i], alpha=0.6, s=20, 
                  label='验证集', color='red')
        
        # 理想线
        min_val = min(train_true[:, i].min(), val_true[:, i].min())
        max_val = max(train_true[:, i].max(), val_true[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        
        ax.set_xlabel(f'真实值 {output_name}')
        ax.set_ylabel(f'预测值 {output_name}')
        ax.set_title(f'{output_name} - 预测vs真实值')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 随机数据预测分布
    for i, (ax, output_name) in enumerate(zip(axes[1, :], output_names)):
        ax.hist(random_pred[:, i], bins=50, alpha=0.7, color='green', 
                edgecolor='black', label='随机数据预测')
        ax.set_xlabel(f'{output_name} 预测值')
        ax.set_ylabel('频数')
        ax.set_title(f'{output_name} - 随机数据预测分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()