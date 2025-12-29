# models/model_manager.py (部分更新)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from utils import save_checkpoint, find_latest_checkpoint, load_model
from utils import print_training_progress
from config import TRAIN_CONFIG, DEVICE, OPTIMIZER_CONFIG, SCHEDULER_CONFIG, MODEL_CONFIG
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')  # 全局设置无GUI后端，必须在import plt前/绘图前执行
import matplotlib.pyplot as plt
class WeightedMSELoss(nn.Module):
    """加权MSE损失函数"""
    def __init__(self, weights=None):
        super().__init__()
        torch.cuda.empty_cache()
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = None
            # self.weights = torch.tensor([1.5, 1.4, 1.0, 1.0, 1.0, 1.0, 1.4, 1.5], dtype=torch.float32)
    
    def forward(self, predictions, targets):
        if self.weights is not None:
            # 将权重移动到相同设备
            weights = self.weights.to(predictions.device)
            # 计算加权MSE
            loss_per_output = torch.mean((predictions - targets) ** 2, dim=0)
            weighted_loss = torch.sum(weights * loss_per_output)
            return weighted_loss
        else:
            # 普通MSE
            return nn.MSELoss()(predictions, targets)
class ModelManager:
    def __init__(self, model, checkpoint_dir=None):
        """
        初始化模型管理器
        
        Args:
            model: 神经网络模型
            checkpoint_dir: 检查点保存目录
            device: 计算设备 (cuda/cpu)
        """
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.device = DEVICE
        self.best_val_loss = float('inf')
        # 创建检查点目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"ModelManager初始化完成，使用设备: {self.device}")
        print(f"检查点目录: {self.checkpoint_dir}")

    def setup_training(self, learning_rate=None, weight_decay=None):
        """设置训练组件 - 使用新的优化器和调度器"""
        learning_rate = learning_rate or OPTIMIZER_CONFIG['lr']
        weight_decay = weight_decay or OPTIMIZER_CONFIG['weight_decay']
        # output_weights = [1.5, 1.4, 1.5, 1.4, 1.0]
        output_weights = None
        # 使用加权MSE损失 
        self.criterion = WeightedMSELoss(weights=output_weights)
        # self.criterion = nn.MSELoss()
        
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 使用ReduceLROnPlateau调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            patience=SCHEDULER_CONFIG['patience'], 
            factor=SCHEDULER_CONFIG['factor'],
            verbose=True  # 打印学习率变化
        )
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        print(f"训练组件设置完成:")
        print(f"使用优化器: {type(self.optimizer).__name__}")
        print(f"损失函数: 加权MSE (权重: {output_weights})")
        print(f"使用调度器: ReduceLROnPlateau")
        print(f"初始学习率: {learning_rate}")
    
    def train_epoch(self, train_loader):
        """训练一个epoch - 适配多模态输入"""
        self.model.train()
        train_loss = 0
        
        for batch in train_loader:
            # 解包多模态输入
            matrices = batch['matrix'].to(self.device)        # (batch, 1, 19, 19)
            frequencies = batch['frequency'].to(self.device)  # (batch, 1)
            s_params = batch['s_params'].to(self.device)      # (batch, 8)
            
            self.optimizer.zero_grad()
            predictions = self.model(matrices, frequencies)   # 模型前向传播
            loss = self.criterion(predictions, s_params)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """验证一个epoch - 适配多模态输入"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                matrices = batch['matrix'].to(self.device)
                frequencies = batch['frequency'].to(self.device)
                s_params = batch['s_params'].to(self.device)
                
                predictions = self.model(matrices, frequencies)
                loss = self.criterion(predictions, s_params)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train_model(self, train_loader, val_loader, epochs=None, resume=True, 
                   print_every=20, save_every=50):
        """
        训练模型的主函数
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            resume: 是否从检查点恢复
            print_every: 每隔多少轮打印一次进度
            save_every: 每隔多少轮保存一次检查点
        """
        epochs = epochs or TRAIN_CONFIG['epochs']
        
        start_epoch = 0
        train_losses = []
        val_losses = []
        
        # 恢复训练
        if resume:
            latest_checkpoint = find_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint:
                print(f"从检查点恢复训练: {latest_checkpoint}")
                start_epoch, train_losses, val_losses = load_model(
                    latest_checkpoint, self.model, self.optimizer
                )
                start_epoch += 1  # 从下一个epoch开始
                
                # 恢复最佳验证损失
                if val_losses:
                    self.best_val_loss = min(val_losses)
                    
                print(f"从 epoch {start_epoch} 继续训练")
            else:
                print("未找到检查点，开始新的训练")
        
        print(f"\n开始训练，总共 {epochs} 个epochs")
        print("=" * 80)
        
        try:
            for epoch in range(start_epoch, epochs):
                # 训练和验证
                avg_train_loss = self.train_epoch(train_loader)
                avg_val_loss = self.validate_epoch(val_loader)
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                # print("diff_loss*************************************************")
                # print(train_losses)
                # print(val_losses)
                # 更新最佳验证损失
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                
                # 学习率调度（基于验证损失）
                self.scheduler.step(avg_val_loss)
                
                # 保存最佳模型
                if avg_val_loss == self.best_val_loss:
                    best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Epoch {epoch}: 保存最佳模型 (损失: {avg_val_loss:.6f})")
                # 打印训练信息
                if epoch % print_every == 0 or epoch == epochs - 1:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print_training_progress(
                        epoch, avg_train_loss, avg_val_loss, current_lr, self.best_val_loss
                    )
                
                # 保存检查点
                if epoch % save_every == 0 or epoch == epochs - 1:
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
                    )
                    save_checkpoint(
                        self.model, self.optimizer, epoch, 
                        train_losses, val_losses, checkpoint_path
                    )
        
        except KeyboardInterrupt:
            print(f"\n训练被用户中断在 epoch {epoch}!")
            # 保存中断时的模型状态
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f'interrupted_epoch_{epoch}.pth'
            )
            save_checkpoint(
                self.model, self.optimizer, epoch, 
                train_losses, val_losses, checkpoint_path
            )
        
        print("=" * 80)
        print(f"训练完成！最佳验证损失: {self.best_val_loss:.6f}")
        
        # 加载最佳模型用于后续评估
        self.load_best_model()
        
        return train_losses, val_losses
    
    def evaluate_model(self, test_loader, dataset_name="测试集"):
        """
        评估模型性能 - 适配多模态输入
        
        Args:
            test_loader: 测试数据加载器
            dataset_name: 数据集名称
        
        Returns:
            predictions, targets, metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_frequencies = []
        i = 0
        with torch.no_grad():
            for batch in test_loader:
                # matrices = batch['matrix'].to(self.device)
                # frequencies = batch['frequency'].to(self.device)
                # s_params = batch['s_params'].to(self.device)
                matrices = batch['matrix'].to(self.device)
                frequencies = batch['frequency'].to(self.device)
                s_params = batch['s_params'].to(self.device)
                predictions = self.model(matrices, frequencies)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(s_params.cpu().numpy())
                all_frequencies.append(frequencies.cpu().numpy())
                # print("##############\n")
                # print(i)
                # i += 1
        # 合并所有批次
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        all_frequencies = np.vstack(all_frequencies)
        
        # 计算评估指标
        metrics = self._calculate_metrics(all_predictions, all_targets, all_frequencies, dataset_name)
        
        return all_predictions, all_targets, all_frequencies, metrics
    
    def _calculate_metrics(self, predictions, targets, frequencies, dataset_name):
        """计算评估指标"""
        # S参数名称
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        metrics = {
            'dataset': dataset_name,
            'overall': {},
            'per_output': {},
            'frequency_based': {}
        }
        
        # 整体指标
        metrics['overall']['R2'] = r2_score(targets, predictions)
        metrics['overall']['MAE'] = mean_absolute_error(targets, predictions)
        metrics['overall']['RMSE'] = np.sqrt(mean_squared_error(targets, predictions))
        metrics['overall']['MSE'] = mean_squared_error(targets, predictions)
        
        # 每个S参数的指标
        for i, name in enumerate(s_param_names):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            metrics['per_output'][name] = {
                'R2': r2_score(target_i, pred_i),
                'MAE': mean_absolute_error(target_i, pred_i),
                'RMSE': np.sqrt(mean_squared_error(target_i, pred_i)),
                'MSE': mean_squared_error(target_i, pred_i),
                'Mean_Relative_Error': np.mean(np.abs((pred_i - target_i) / (np.abs(target_i) + 1e-12))) * 100
            }
        
        # 基于频率的分析（可选）
        # 将频率分组，分析不同频率下的性能
        freq_groups = np.digitize(frequencies.flatten(), bins=[8, 9, 10, 11])
        unique_groups = np.unique(freq_groups)
        
        for group in unique_groups:
            mask = freq_groups == group
            if np.sum(mask) > 10:  # 确保有足够样本
                group_predictions = predictions[mask]
                group_targets = targets[mask]
                
                metrics['frequency_based'][f'freq_group_{group}'] = {
                    'R2': r2_score(group_targets, group_predictions),
                    'MAE': mean_absolute_error(group_targets, group_predictions),
                    'RMSE': np.sqrt(mean_squared_error(group_targets, group_predictions)),
                    'sample_count': np.sum(mask)
                }
        
        return metrics
    def print_evaluation_results(self, metrics):
        """打印评估结果"""
        print("\n" + "="*60)
        print(f"评估结果 - {metrics['dataset']}")
        print("="*60)
        
        print("\n整体指标:")
        print(f"  R²: {metrics['overall']['R2']:.6f}")
        print(f"  MAE: {metrics['overall']['MAE']:.6f}")
        print(f"  RMSE: {metrics['overall']['RMSE']:.6f}")
        print(f"  MSE: {metrics['overall']['MSE']:.6f}")
        
        print("\n各S参数指标:")
        for name, param_metrics in metrics['per_output'].items():
            print(f"\n{name}:")
            print(f"  R²: {param_metrics['R2']:.4f}")
            print(f"  MAE: {param_metrics['MAE']:.6f}")
            print(f"  RMSE: {param_metrics['RMSE']:.6f}")
            print(f"  平均相对误差: {param_metrics['Mean_Relative_Error']:.2f}%")
    
    def predict_single_sample(self, matrix, frequency):
        """
        预测单个样本
        
        Args:
            matrix: 19x19的二进制矩阵
            frequency: 频率值（GHz）
        
        Returns:
            S参数预测值
        """
        self.model.eval()
        
        # 确保输入格式正确
        if isinstance(matrix, np.ndarray):
            matrix = torch.FloatTensor(matrix)
        if isinstance(frequency, (int, float)):
            frequency = torch.FloatTensor([[frequency]])  # 归一化
        print("Sample Input:\n")
        print(matrix)
        print(frequency)
        # 添加批次维度
        if len(matrix.shape) == 3:  # (1, 19, 19)
            matrix = matrix.unsqueeze(0)  # (1, 1, 19, 19)
        
        matrix = matrix.to(self.device)
        frequency = frequency.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(matrix, frequency)
        
        # 转换为numpy数组并解析为字典
        predictions_np = predictions.cpu().numpy().flatten()
        
        s_param_dict = {
            'S11_real': predictions_np[0],
            'S11_imag': predictions_np[1],
            'S21_real': predictions_np[2],
            'S21_imag': predictions_np[3],
            'S12_real': predictions_np[4],
            'S12_imag': predictions_np[5],
            'S22_real': predictions_np[6],
            'S22_imag': predictions_np[7]
        }
        print("Predict:s_param_dict\n")
        print(s_param_dict)
        return s_param_dict
    
    def evaluate_all_test_samples(self, test_loader):
        """
        评估测试集中所有样本的误差
        
        Returns:
            sample_errors: 每个样本的误差信息
            predictions: 所有预测值
            targets: 所有真实值
        """
        print("\n" + "="*60)
        print("评估测试集中所有样本的误差")
        print("="*60)
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_matrices = []
        all_frequencies = []
        
        # 收集所有数据
        with torch.no_grad():
            for batch in test_loader:
                matrices = batch['matrix'].to(self.device)
                frequencies = batch['frequency'].to(self.device)
                s_params = batch['s_params'].to(self.device)
                
                predictions = self.model(matrices, frequencies)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(s_params.cpu().numpy())
                all_matrices.append(matrices.cpu().numpy())
                all_frequencies.append(frequencies.cpu().numpy())
        
        # 合并所有批次
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        all_frequencies = np.vstack(all_frequencies)
        
        # 计算每个样本的误差
        sample_errors = []
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        for i in range(len(all_predictions)):
            sample_error = {}
            total_rel_error = 0
            valid_outputs = 0
            
            for j, name in enumerate(s_param_names):
                true_val = all_targets[i, j]
                pred_val = all_predictions[i, j]
                
                # 计算相对误差
                if abs(true_val) > 1e-12:
                    rel_error = abs(pred_val - true_val) / abs(true_val) * 100
                else:
                    rel_error = float('inf')
                
                sample_error[name] = {
                    'true': true_val,
                    'pred': pred_val,
                    'rel_error': rel_error
                }
                
                if rel_error != float('inf'):
                    total_rel_error += rel_error
                    valid_outputs += 1
            
            # 计算平均相对误差
            if valid_outputs > 0:
                avg_rel_error = total_rel_error / valid_outputs
            else:
                avg_rel_error = float('inf')
            
            sample_errors.append({
                'sample_index': i,
                'errors': sample_error,
                'avg_rel_error': avg_rel_error,
                'frequency': all_frequencies[i][0]  # 反归一化：需要乘以30得到GHz
            })
        
        return sample_errors, all_predictions, all_targets
    
    def analyze_error_statistics(self, sample_errors):
        """分析误差统计信息"""
        print("\n" + "="*60)
        print("测试集误差统计分析")
        print("="*60)
        
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        # 为每个S参数收集误差
        error_stats = {}
        
        for name in s_param_names:
            rel_errors = []
            for sample in sample_errors:
                error = sample['errors'][name]['rel_error']
                if error != float('inf'):
                    rel_errors.append(error)
            
            if rel_errors:
                error_stats[name] = {
                    'min_error': np.min(rel_errors),
                    'max_error': np.max(rel_errors),
                    'mean_error': np.mean(rel_errors),
                    'median_error': np.median(rel_errors),
                    'std_error': np.std(rel_errors),
                    'p95_error': np.percentile(rel_errors, 95),
                    'p99_error': np.percentile(rel_errors, 99)
                }
        
        # 打印每个S参数的误差统计
        for name, stats in error_stats.items():
            print(f"\n{name} 误差统计:")
            print(f"  最小误差: {stats['min_error']:.2f}%")
            print(f"  最大误差: {stats['max_error']:.2f}%")
            print(f"  平均误差: {stats['mean_error']:.2f}%")
            print(f"  中位数误差: {stats['median_error']:.2f}%")
            print(f"  误差标准差: {stats['std_error']:.2f}%")
            print(f"  95%分位数: {stats['p95_error']:.2f}%")
        
        # 找出误差最大和最小的样本
        valid_samples = [s for s in sample_errors if s['avg_rel_error'] != float('inf')]
        
        if valid_samples:
            min_error_sample = min(valid_samples, key=lambda x: x['avg_rel_error'])
            max_error_sample = max(valid_samples, key=lambda x: x['avg_rel_error'])
            
            print(f"\n{'='*60}")
            print("样本误差极值分析")
            print(f"{'='*60}")
            
            print(f"\n🎯 最佳预测样本 (索引: {min_error_sample['sample_index']}):")
            print(f"  平均相对误差: {min_error_sample['avg_rel_error']:.2f}%")
            print(f"  频率: {min_error_sample['frequency']*30:.1f} GHz")  # 反归一化
            
            print(f"\n⚠️ 最差预测样本 (索引: {max_error_sample['sample_index']}):")
            print(f"  平均相对误差: {max_error_sample['avg_rel_error']:.2f}%")
            print(f"  频率: {max_error_sample['frequency']*30:.1f} GHz")
        
        return error_stats, min_error_sample, max_error_sample
    
    def create_error_distribution_plot(self, sample_errors, save_path=None):
        """创建误差分布图"""
        import matplotlib.pyplot as plt
        
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # 绘制每个S参数的误差分布
        for i, name in enumerate(s_param_names):
            if i >= len(axes):
                break
                
            rel_errors = []
            for sample in sample_errors:
                error = sample['errors'][name]['rel_error']
                if error != float('inf'):
                    rel_errors.append(error)
            
            if rel_errors:
                axes[i].hist(rel_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(np.mean(rel_errors), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(rel_errors):.2f}%')
                axes[i].axvline(np.median(rel_errors), color='green', linestyle='--', 
                               label=f'Median: {np.median(rel_errors):.2f}%')
                axes[i].set_xlabel('Relative Error (%)')
                axes[i].set_ylabel('Hz')
                axes[i].set_title(f'{name} Error Distribution')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # 平均误差分布
        avg_errors = [s['avg_rel_error'] for s in sample_errors if s['avg_rel_error'] != float('inf')]
        if avg_errors:
            axes[-1].hist(avg_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[-1].axvline(np.mean(avg_errors), color='red', linestyle='--', 
                            label=f'Mean: {np.mean(avg_errors):.2f}%')
            axes[-1].axvline(np.median(avg_errors), color='green', linestyle='--', 
                            label=f'Median: {np.median(avg_errors):.2f}%')
            axes[-1].set_xlabel('Average Relative Error (%)')
            axes[-1].set_ylabel('Hz')
            axes[-1].set_title('Sample Mean Error Distribution')
            axes[-1].legend()
            axes[-1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"误差分布图已保存到: {save_path}")
        
        # plt.show()
    
    def save_final_model(self, filepath, model_info=None):
        """保存最终模型"""
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'model_info': model_info or {},
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(save_data, filepath)
        print(f"最终模型已保存: {filepath}")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
    
    def load_best_model(self):
        """加载最佳模型"""
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print("已加载最佳模型")
            return True
        else:
            print("未找到最佳模型文件")
            return False
    
    def load_model_from_path(self, model_path):
        """从指定路径加载模型"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            
            print(f"从 {model_path} 加载模型成功")
            return True
        else:
            print(f"模型文件不存在: {model_path}")
            return False
    
    def create_training_summary(self, train_losses, val_losses, train_metrics, test_metrics):
        """创建训练总结报告"""
        print("\n" + "="*80)
        print("神经网络模型训练总结")
        print("="*80)
        
        # 训练过程总结
        print(f"\n📈 训练过程:")
        print(f"  训练轮数: {len(train_losses)}")
        print(f"  最终训练损失: {train_losses[-1]:.6f}")
        print(f"  最终验证损失: {val_losses[-1]:.6f}")
        print(f"  最佳验证损失: {self.best_val_loss:.6f}")
        
        # 训练集评估结果
        print(f"\n📊 训练集评估结果:")
        print(f"  整体R²: {train_metrics['overall']['R2']:.4f}")
        print(f"  整体MAE: {train_metrics['overall']['MAE']:.6f}")
        print(f"  整体RMSE: {train_metrics['overall']['RMSE']:.6f}")
        
        # 测试集评估结果
        print(f"\n📊 测试集评估结果:")
        print(f"  整体R²: {test_metrics['overall']['R2']:.4f}")
        print(f"  整体MAE: {test_metrics['overall']['MAE']:.6f}")
        print(f"  整体RMSE: {test_metrics['overall']['RMSE']:.6f}")
        
        # 各S参数表现对比
        print(f"\n🎯 各S参数表现对比:")
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        for name in s_param_names:
            train_r2 = train_metrics['per_output'][name]['R2']
            test_r2 = test_metrics['per_output'][name]['R2']
            print(f"  {name}: 训练集R²={train_r2:.4f}, 测试集R²={test_r2:.4f}")
        
        print("\n" + "="*80)
    def plot_training_history(self, train_losses, val_losses, save_path=None):
        """
        绘制训练历史曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制损失曲线
        epochs = range(1, len(train_losses) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 绘制对数尺度损失曲线
        plt.subplot(1, 2, 2)
        plt.semilogy(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        plt.semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Log Loss', fontsize=12)
        plt.title('Log Scale Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        
        plt.suptitle(f'Model Training History (Best Val Loss: {self.best_val_loss:.6f})', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        print("???????????????????????????????????2")
        # plt.show()
        return plt.gcf()
    def plot_scatter_comparison(self, predictions, targets, frequencies=None, 
                               dataset_name="Test Set", save_path=None):
        """
        绘制预测值与真实值的散点对比图
        
        Args:
            predictions: 预测值数组 (n_samples, 8)
            targets: 真实值数组 (n_samples, 8)
            frequencies: 频率数组 (n_samples, 1) 可选
            dataset_name: 数据集名称
            save_path: 保存路径（可选）
        """
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        # 计算每个S参数的R²和MAE
        metrics_per_param = []
        for i, name in enumerate(s_param_names):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            r2 = r2_score(target_i, pred_i)
            mae = mean_absolute_error(target_i, pred_i)
            metrics_per_param.append({
                'name': name,
                'r2': r2,
                'mae': mae
            })
        
        # 创建子图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # 为每个S参数绘制散点图
        for i, (name, metrics) in enumerate(zip(s_param_names, metrics_per_param)):
            ax = axes[i]
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            # 绘制散点
            if frequencies is not None:
                # 使用频率作为颜色映射
                scatter = ax.scatter(target_i, pred_i, c=frequencies.flatten(), 
                                   cmap='viridis', alpha=0.6, s=30, 
                                   edgecolors='white', linewidth=0.5)
                # 添加颜色条
                if i == 0:  # 只在第一个子图添加颜色条
                    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
                    cbar.set_label('Normalized Frequency', fontsize=10)
            else:
                ax.scatter(target_i, pred_i, alpha=0.6, s=30, 
                          color='steelblue', edgecolors='white', linewidth=0.5)
            
            # 添加对角线（理想预测线）
            min_val = min(np.min(target_i), np.min(pred_i))
            max_val = max(np.max(target_i), np.max(pred_i))
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', alpha=0.7, linewidth=2, label='Perfect Prediction')
            
            # 设置坐标轴
            ax.set_xlabel('True Values', fontsize=11)
            ax.set_ylabel('Predicted Values', fontsize=11)
            ax.set_title(f'{name}\nR² = {metrics["r2"]:.4f}, MAE = {metrics["mae"]:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', 'box')
            
            # 添加R²和MAE文本
            textstr = f'R² = {metrics["r2"]:.4f}\nMAE = {metrics["mae"]:.4f}'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 移除多余的子图
        for i in range(len(s_param_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'Prediction vs True Values - {dataset_name}\nScatter Plot Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        print("???????????????????????????????????4")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"散点对比图已保存到: {save_path}")
        print("???????????????????????????????????5")
        # plt.show()
        return fig, axes
    def plot_residual_distribution(self, predictions, targets, save_path=None):
        """
        绘制残差分布图
        
        Args:
            predictions: 预测值数组
            targets: 真实值数组
            save_path: 保存路径（可选）
        """
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        residuals = predictions - targets
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, name in enumerate(s_param_names):
            ax = axes[i]
            residual_i = residuals[:, i]
            
            # 绘制残差直方图
            n, bins, patches = ax.hist(residual_i, bins=50, alpha=0.7, 
                                     color='skyblue', edgecolor='black', 
                                     density=True)
            
            # 添加正态分布曲线
            from scipy.stats import norm
            mu, std = norm.fit(residual_i)
            x = np.linspace(min(residual_i), max(residual_i), 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r-', linewidth=2, label=f'Normal Fit\nμ={mu:.4f}, σ={std:.4f}')
            
            # 添加均值和中位数线
            ax.axvline(residual_i.mean(), color='green', linestyle='--', 
                      linewidth=2, label=f'Mean: {residual_i.mean():.4f}')
            ax.axvline(np.median(residual_i), color='orange', linestyle='--', 
                      linewidth=2, label=f'Median: {np.median(residual_i):.4f}')
            
            ax.set_xlabel('Residual', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{name} Residual Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # 移除多余的子图
        for i in range(len(s_param_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Residual Distribution Analysis\n(Prediction - True Value)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"残差分布图已保存到: {save_path}")
        
        # plt.show()
        return fig, axes
    def plot_prediction_comparison_by_frequency(self, predictions, targets, frequencies, 
                                               save_path=None):
        """
        按频率分组的预测对比图
        
        Args:
            predictions: 预测值
            targets: 真实值
            frequencies: 频率值（归一化的）
            save_path: 保存路径
        """
        # 将频率反归一化并分组
        freq_ghz = frequencies.flatten() * 30  # 反归一化，假设原始归一化到0-30GHz
        freq_bins = [0, 9, 10, 11, 12]  # 频率分组 (GHz)
        freq_labels = ['8-9GHz', '9-10GHz', '10-11GHz', '11-12GHz']
        
        freq_groups = pd.cut(freq_ghz, bins=freq_bins, labels=freq_labels, include_lowest=True)
        
        # 计算每个频率组的误差指标
        group_metrics = {}
        for group in freq_labels:
            mask = freq_groups == group
            if np.sum(mask) > 0:
                group_pred = predictions[mask]
                group_true = targets[mask]
                mae = mean_absolute_error(group_true, group_pred)
                rmse = np.sqrt(mean_squared_error(group_true, group_pred))
                r2 = r2_score(group_true, group_pred)
                group_metrics[group] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'count': np.sum(mask)}
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. 各频率组的样本数量
        ax1 = axes[0]
        counts = [group_metrics[g]['count'] for g in freq_labels if g in group_metrics]
        labels = [g for g in freq_labels if g in group_metrics]
        ax1.bar(labels, counts, color=sns.color_palette("husl", len(labels)))
        ax1.set_xlabel('Frequency Range', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title('Sample Distribution by Frequency', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 在柱状图上添加数量标签
        for i, count in enumerate(counts):
            ax1.text(i, count + max(counts)*0.02, str(count), 
                    ha='center', va='bottom', fontsize=11)
        
        # 2. 各频率组的MAE
        ax2 = axes[1]
        mae_values = [group_metrics[g]['MAE'] for g in labels]
        bars = ax2.bar(labels, mae_values, color=sns.color_palette("husl", len(labels)))
        ax2.set_xlabel('Frequency Range', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title('MAE by Frequency Range', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 各频率组的R²
        ax3 = axes[2]
        r2_values = [group_metrics[g]['R2'] for g in labels]
        bars = ax3.bar(labels, r2_values, color=sns.color_palette("husl", len(labels)))
        ax3.set_xlabel('Frequency Range', fontsize=12)
        ax3.set_ylabel('R² Score', fontsize=12)
        ax3.set_title('R² Score by Frequency Range', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)  # R²通常在0-1之间
        
        # 4. 整体预测误差随频率变化
        ax4 = axes[3]
        # 计算每个样本的平均绝对误差
        sample_mae = np.mean(np.abs(predictions - targets), axis=1)
        scatter = ax4.scatter(freq_ghz, sample_mae, alpha=0.6, s=30, 
                             c=sample_mae, cmap='plasma')
        ax4.set_xlabel('Frequency (GHz)', fontsize=12)
        ax4.set_ylabel('Sample MAE', fontsize=12)
        ax4.set_title('Prediction Error vs Frequency', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('MAE', fontsize=11)
        
        plt.suptitle('Frequency-based Performance Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"频率分析图已保存到: {save_path}")
        
        # plt.show()
        return fig, axes, group_metrics
    
    def create_comprehensive_visualization(self, train_losses, val_losses, 
                                         test_predictions, test_targets, 
                                         test_frequencies, output_dir=None):
        """
        创建综合可视化报告，包含所有图表
        
        Args:
            train_losses: 训练损失
            val_losses: 验证损失
            test_predictions: 测试集预测值
            test_targets: 测试集真实值
            test_frequencies: 测试集频率
            output_dir: 输出目录
            
        Returns:
            包含所有图形的字典
        """
        import os
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        figures = {}
        
        print("开始生成可视化报告...")
        
        # 1. 训练历史图
        print("1. 生成训练历史图...")
        fig_loss = self.plot_training_history(train_losses, val_losses)
        figures['training_history'] = fig_loss
        
        if output_dir:
            fig_loss.savefig(os.path.join(output_dir, 'training_history.png'), 
                           dpi=300, bbox_inches='tight')
        
        # 2. 散点对比图
        print("2. 生成散点对比图...")
        fig_scatter, axes_scatter = self.plot_scatter_comparison(
            test_predictions, test_targets, test_frequencies, "Test Set"
        )
        figures['scatter_comparison'] = fig_scatter
        
        if output_dir:
            fig_scatter.savefig(os.path.join(output_dir, 'scatter_comparison.png'), 
                              dpi=300, bbox_inches='tight')
        
        # 3. 残差分布图
        print("3. 生成残差分布图...")
        fig_residual, axes_residual = self.plot_residual_distribution(
            test_predictions, test_targets
        )
        figures['residual_distribution'] = fig_residual
        
        if output_dir:
            fig_residual.savefig(os.path.join(output_dir, 'residual_distribution.png'), 
                               dpi=300, bbox_inches='tight')
        
        # 4. 频率分析图
        print("4. 生成频率分析图...")
        fig_freq, axes_freq, freq_metrics = self.plot_prediction_comparison_by_frequency(
            test_predictions, test_targets, test_frequencies
        )
        figures['frequency_analysis'] = fig_freq
        
        if output_dir:
            fig_freq.savefig(os.path.join(output_dir, 'frequency_analysis.png'), 
                           dpi=300, bbox_inches='tight')
        
        # 5. 创建汇总报告
        print("5. 生成汇总报告...")
        self._create_summary_report(train_losses, val_losses, test_predictions, 
                                  test_targets, output_dir)
        
        print("可视化报告生成完成!")
        
        return figures
    def _create_summary_report(self, train_losses, val_losses, 
                              test_predictions, test_targets, output_dir=None):
        """创建文本格式的汇总报告"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("模型训练与评估汇总报告")
        report_lines.append("=" * 80)
        report_lines.append(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 训练信息
        report_lines.append("\n" + "-" * 40)
        report_lines.append("训练过程信息")
        report_lines.append("-" * 40)
        report_lines.append(f"总训练轮数: {len(train_losses)}")
        report_lines.append(f"最终训练损失: {train_losses[-1]:.6f}")
        report_lines.append(f"最终验证损失: {val_losses[-1]:.6f}")
        report_lines.append(f"最佳验证损失: {self.best_val_loss:.6f}")
        
        # 整体性能指标
        overall_r2 = r2_score(test_targets, test_predictions)
        overall_mae = mean_absolute_error(test_targets, test_predictions)
        overall_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
        
        report_lines.append("\n" + "-" * 40)
        report_lines.append("整体测试性能")
        report_lines.append("-" * 40)
        report_lines.append(f"R² 分数: {overall_r2:.6f}")
        report_lines.append(f"平均绝对误差 (MAE): {overall_mae:.6f}")
        report_lines.append(f"均方根误差 (RMSE): {overall_rmse:.6f}")
        
        # 各S参数性能
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        report_lines.append("\n" + "-" * 40)
        report_lines.append("各S参数详细性能")
        report_lines.append("-" * 40)
        
        for i, name in enumerate(s_param_names):
            pred_i = test_predictions[:, i]
            target_i = test_targets[:, i]
            
            r2 = r2_score(target_i, pred_i)
            mae = mean_absolute_error(target_i, pred_i)
            mre = np.mean(np.abs((pred_i - target_i) / (np.abs(target_i) + 1e-12))) * 100
            
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  R²: {r2:.6f}")
            report_lines.append(f"  MAE: {mae:.6f}")
            report_lines.append(f"  平均相对误差: {mre:.2f}%")
        
        # 将报告写入文件
        if output_dir:
            report_path = os.path.join(output_dir, 'summary_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"汇总报告已保存到: {report_path}")
        
        # 在控制台打印报告
        print('\n'.join(report_lines))