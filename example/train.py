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
class HuberWithPhysicsLoss(nn.Module):
    """
    Huber损失（平滑L1）结合物理约束
    Huber损失对异常值比MSE更鲁棒，比L1更平滑
    """
    def __init__(self, delta=0.1, physics_weight=0.05):
        super().__init__()
        self.delta = delta  # Huber损失阈值
        self.physics_weight = physics_weight
        
    def huber_loss(self, pred, target):
        """Huber损失实现"""
        diff = torch.abs(pred - target)
        condition = diff < self.delta
        loss = torch.where(
            condition,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        return torch.mean(loss)
    
    def physics_constraints(self, pred):
        """物理约束：确保S参数满足无源网络条件"""
        # 提取实部和虚部
        real = pred[:, [0, 2, 4, 6]]  # S11_real, S21_real, S12_real, S22_real
        imag = pred[:, [1, 3, 5, 7]]  # S11_imag, S21_imag, S12_imag, S22_imag
        
        # 约束1：幅度<=1（无源网络）
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        mag_constraint = torch.mean(torch.clamp(magnitude - 1.0, min=0)**2)
        
        # 约束2：互易性（S12 ≈ S21），如果您的网络是互易的
        s21 = pred[:, 2:4]  # S21
        s12 = pred[:, 4:6]  # S12
        reciprocity_constraint = F.mse_loss(s21, s12)
        
        # 约束3：能量守恒（可选，复杂）
        # |S11|^2 + |S21|^2 <= 1
        
        return mag_constraint + 0.1 * reciprocity_constraint
    
    def forward(self, pred, target, frequency=None):
        # 基础Huber损失
        huber = self.huber_loss(pred, target)
        
        # 频率加权（可选）
        if frequency is not None:
            # 对低频样本给予更高权重
            freq_weights = 1.0 / (frequency + 0.1)
            weighted_huber = torch.mean(freq_weights * self.huber_loss(pred, target, reduction='none'))
            base_loss = 0.7 * huber + 0.3 * weighted_huber
        else:
            base_loss = huber
        
        # 物理约束
        physics_loss = self.physics_constraints(pred)
        
        # 组合损失
        total_loss = base_loss + self.physics_weight * physics_loss
        
        return total_loss
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
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
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
        
        # # 恢复训练
        # if resume:
        #     latest_checkpoint = find_latest_checkpoint(self.checkpoint_dir)
        #     if latest_checkpoint:
        #         print(f"从检查点恢复训练: {latest_checkpoint}")
        #         start_epoch, train_losses, val_losses = load_model(
        #             latest_checkpoint, self.model, self.optimizer
        #         )
        #         start_epoch += 1  # 从下一个epoch开始
                
        #         # 恢复最佳验证损失
        #         if val_losses:
        #             self.best_val_loss = min(val_losses)
                    
        #         print(f"从 epoch {start_epoch} 继续训练")
        #     else:
        #         print("未找到检查点，开始新的训练")
        
        # print(f"\n开始训练，总共 {epochs} 个epochs")
        # print("=" * 80)
        
        # try:
        #     for epoch in range(start_epoch, epochs):
        #         # 训练和验证
        #         avg_train_loss = self.train_epoch(train_loader)
        #         avg_val_loss = self.validate_epoch(val_loader)
                
        #         train_losses.append(avg_train_loss)
        #         val_losses.append(avg_val_loss)
        #         # print("diff_loss*************************************************")
        #         # print(train_losses)
        #         # print(val_losses)
        #         # 更新最佳验证损失
        #         if avg_val_loss < self.best_val_loss:
        #             self.best_val_loss = avg_val_loss
                
        #         # 学习率调度（基于验证损失）
        #         self.scheduler.step(avg_val_loss)
                
        #         # 保存最佳模型
        #         if avg_val_loss == self.best_val_loss:
        #             best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        #             torch.save(self.model.state_dict(), best_model_path)
        #             print(f"Epoch {epoch}: 保存最佳模型 (损失: {avg_val_loss:.6f})")
        #         # 打印训练信息
        #         if epoch % print_every == 0 or epoch == epochs - 1:
        #             current_lr = self.optimizer.param_groups[0]['lr']
        #             print_training_progress(
        #                 epoch, avg_train_loss, avg_val_loss, current_lr, self.best_val_loss
        #             )
                
        #         # 保存检查点
        #         if epoch % save_every == 0 or epoch == epochs - 1:
        #             checkpoint_path = os.path.join(
        #                 self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
        #             )
        #             save_checkpoint(
        #                 self.model, self.optimizer, epoch, 
        #                 train_losses, val_losses, checkpoint_path
        #             )
        
        # except KeyboardInterrupt:
        #     print(f"\n训练被用户中断在 epoch {epoch}!")
        #     # 保存中断时的模型状态
        #     checkpoint_path = os.path.join(
        #         self.checkpoint_dir, f'interrupted_epoch_{epoch}.pth'
        #     )
        #     save_checkpoint(
        #         self.model, self.optimizer, epoch, 
        #         train_losses, val_losses, checkpoint_path
        #     )
        
        # print("=" * 80)
        # print(f"训练完成！最佳验证损失: {self.best_val_loss:.6f}")
        
        # 加载最佳模型用于后续评估
        self.load_best_model()
        # self.print_model_first_10_params(self.model)
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

                test = (matrices,frequencies)
                print(test[:10])

                predictions = self.model(matrices, frequencies)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(s_params.cpu().numpy())
                all_frequencies.append(frequencies.cpu().numpy())
        # 合并所有批次
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        all_frequencies = np.vstack(all_frequencies)
        print(all_predictions[:10])
        # 计算评估指标
        metrics = self._calculate_metrics(all_predictions, all_targets, all_frequencies, dataset_name)
        print("******************************************************")
        
        return all_predictions, all_targets, all_frequencies, metrics
    
    def _linear_to_db(self, linear_value):
        """线性值转dB值"""
        # 防止log10(0)和负值
        magnitude = np.abs(linear_value)
        db_value = 10 * np.log10(np.maximum(magnitude, 1e-7))  # 防止除0
        return db_value

    def _calculate_metrics(self, predictions, targets, frequencies, dataset_name):
        """计算评估指标"""
        # S参数名称
        s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                        'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
        
        metrics = {
            'dataset': dataset_name,
            'overall': {},
            'per_output': {},
            'frequency_based': {},
            'db_metrics': {},  # 新增dB指标
            'absolute_error_stats': {},  # 新增：绝对误差统计
            'db_error_stats': {}  # 新增：dB误差统计
        }
        
        # 整体指标
        metrics['overall']['R2'] = r2_score(targets, predictions)
        metrics['overall']['MAE'] = mean_absolute_error(targets, predictions)
        metrics['overall']['RMSE'] = np.sqrt(mean_squared_error(targets, predictions))
        metrics['overall']['MSE'] = mean_squared_error(targets, predictions)
        # 计算绝对误差数组
        absolute_errors = np.abs(predictions - targets)
        # 整体绝对误差统计
        metrics['absolute_error_stats']['overall'] = {
            'mean': np.mean(absolute_errors),
            'median': np.median(absolute_errors),
            '95th_percentile': np.percentile(absolute_errors, 95),
            'max': np.max(absolute_errors),
            'min': np.min(absolute_errors),
            'std': np.std(absolute_errors)
        }
        # 每个S参数的指标
        for i, name in enumerate(s_param_names):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            abs_error_i = absolute_errors[:, i]
            metrics['per_output'][name] = {
                'R2': r2_score(target_i, pred_i),
                'MAE': mean_absolute_error(target_i, pred_i),
                'RMSE': np.sqrt(mean_squared_error(target_i, pred_i)),
                'MSE': mean_squared_error(target_i, pred_i),
                'absolute_error_stats': {
                'mean': np.mean(abs_error_i),
                'median': np.median(abs_error_i),
                '95th_percentile': np.percentile(abs_error_i, 95),
                'max': np.max(abs_error_i),
                'min': np.min(abs_error_i),
                'std': np.std(abs_error_i)
            }
            }
        # 3. 新增：dB误差评估
        # 3.1 计算每个S参数的复数值
        s_param_complex_names = ['S11', 'S21', 'S12', 'S22']
        all_db_errors = []  # 收集所有dB误差用于整体统计
        for i, s_name in enumerate(s_param_complex_names):
            real_idx = i * 2
            imag_idx = i * 2 + 1
            
            # 复数形式
            pred_complex = predictions[:, real_idx] + 1j * predictions[:, imag_idx]
            true_complex = targets[:, real_idx] + 1j * targets[:, imag_idx]
            
            # 幅度（线性）
            pred_mag = np.abs(pred_complex)
            true_mag = np.abs(true_complex)
            
            # dB值
            
            # 计算dB绝对误差
            db_errors = self._linear_to_db(np.abs(pred_mag - true_mag))
            
            all_db_errors.extend(db_errors)  # 收集用于整体统计
            # 相位（角度）
            pred_phase = np.angle(pred_complex, deg=True)  # 转换为角度
            true_phase = np.angle(true_complex, deg=True)
            
            # 相位误差（处理环绕问题）
            phase_diff = pred_phase - true_phase
            phase_diff = ((phase_diff + 180) % 360) - 180  # 限制在[-180, 180]度
            
            metrics['db_metrics'][s_name] = {
                'db_mae': np.mean(db_errors),  # dB误差平均值
                'db_rmse': np.sqrt(np.mean((db_errors) ** 2)),
                # 'db_corr': np.corrcoef(pred_db, true_db)[0, 1],
                'phase_mae': np.mean(np.abs(phase_diff)),  # 相位误差
                'phase_rmse': np.sqrt(np.mean(phase_diff ** 2)),
                'magnitude_r2': r2_score(true_mag, pred_mag),  # 幅度R²
                'phase_r2': r2_score(true_phase, pred_phase),  # 相位R²
                'db_error_stats': {
                'mean': np.mean(db_errors),
                'median': np.median(db_errors),
                '95th_percentile': np.percentile(db_errors, 95),
                'max': np.max(db_errors),
                'min': np.min(db_errors),
                'std': np.std(db_errors)
            }
            }
            
            # # 分区评估：大信号和小信号
            # large_signal_mask = true_mag > 0.05  # 线性值大于0.05
            # small_signal_mask = true_mag < 0.01  # 线性值小于0.01
            
            # if np.sum(large_signal_mask) > 0:
            #     metrics['db_metrics'][s_name]['large_signal_db_error'] = np.mean(
            #         np.abs(pred_db[large_signal_mask] - true_db[large_signal_mask]))
            #     metrics['db_metrics'][s_name]['large_signal_count'] = np.sum(large_signal_mask)
            
            # if np.sum(small_signal_mask) > 0:
            #     metrics['db_metrics'][s_name]['small_signal_db_error'] = np.mean(
            #         np.abs(pred_db[small_signal_mask] - true_db[small_signal_mask]))
            #     metrics['db_metrics'][s_name]['small_signal_count'] = np.sum(small_signal_mask)
        # 计算整体dB误差统计
        if all_db_errors:
            all_db_errors = np.array(all_db_errors)
            metrics['db_error_stats']['overall'] = {
                'mean': np.mean(all_db_errors),
                'median': np.median(all_db_errors),
                '95th_percentile': np.percentile(all_db_errors, 95),
                'max': np.max(all_db_errors),
                'min': np.min(all_db_errors),
                'std': np.std(all_db_errors)
            }
        # 基于频率的分析（可选）
        # 将频率分组，分析不同频率下的性能
        freq_groups = np.digitize(frequencies.flatten(), bins=[0.2,0.4,0.6,0.8])
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
        
        print("\n📊 绝对误差统计 (线性值):")
        overall_stats = metrics['absolute_error_stats']['overall']
        print(f"  均值: {overall_stats['mean']:.8f}")
        print(f"  中位数: {overall_stats['median']:.8f}")
        print(f"  95%分位数: {overall_stats['95th_percentile']:.8f}")
        print(f"  最大值: {overall_stats['max']:.8f}")
        print(f"  最小值: {overall_stats['min']:.8f}")
        print(f"  标准差: {overall_stats['std']:.8f}")

        print("\n🔍 dB误差指标:")
        if 'db_error_stats' in metrics and 'overall' in metrics['db_error_stats']:
            db_stats = metrics['db_error_stats']['overall']
            print(f"  整体dB误差统计:")
            print(f"    均值: {db_stats['mean']:.2f} dB")
            print(f"    中位数: {db_stats['median']:.2f} dB")
            print(f"    95%分位数: {db_stats['95th_percentile']:.2f} dB")
            print(f"    最大值: {db_stats['max']:.2f} dB")
            print(f"    最小值: {db_stats['min']:.2f} dB")
            print(f"    标准差: {db_stats['std']:.2f} dB")
        
        for s_name, db_metrics in metrics['db_metrics'].items():
            print(f"\n  {s_name}:")
            print(f"    dB MAE: {db_metrics['db_mae']:.2f} dB")
            print(f"    dB RMSE: {db_metrics['db_rmse']:.2f} dB")
            # print(f"    dB 相关性: {db_metrics['db_corr']:.4f}")
            
            if 'db_error_stats' in db_metrics:
                stats = db_metrics['db_error_stats']
                print(f"    dB误差统计:")
                print(f"      均值: {stats['mean']:.2f} dB")
                print(f"      中位数: {stats['median']:.2f} dB")
                print(f"      95%分位数: {stats['95th_percentile']:.2f} dB")
            
            print(f"    相位 MAE: {db_metrics['phase_mae']:.2f}°")
            print(f"    幅度 R²: {db_metrics['magnitude_r2']:.4f}")
            
            if 'large_signal_db_error' in db_metrics:
                print(f"    大信号(|S|>0.05) dB误差: {db_metrics['large_signal_db_error']:.2f} dB "
                    f"({db_metrics['large_signal_count']}个样本)")
            
            if 'small_signal_db_error' in db_metrics:
                print(f"    小信号(|S|<0.01) dB误差: {db_metrics['small_signal_db_error']:.2f} dB "
                    f"({db_metrics['small_signal_count']}个样本)")
        
        print("\n📈 各S参数线性指标:")
        for name, param_metrics in metrics['per_output'].items():
            print(f"\n{name}:")
            print(f"  R²: {param_metrics['R2']:.4f}")
            print(f"  MAE: {param_metrics['MAE']:.6f}")
            print(f"  RMSE: {param_metrics['RMSE']:.6f}")
            
            if 'absolute_error_stats' in param_metrics:
                stats = param_metrics['absolute_error_stats']
                print(f"  绝对误差统计:")
                print(f"    均值: {stats['mean']:.6f}")
                print(f"    中位数: {stats['median']:.6f}")
                print(f"    95%分位数: {stats['95th_percentile']:.6f}")
        
        print("\n" + "="*60)
        print("总结报告:")
        print("="*60)
        
        # 总结关键指标
        print(f"\n关键性能指标:")
        print(f"  线性绝对误差均值: {overall_stats['mean']:.6f}")
        print(f"  线性绝对误差中位数: {overall_stats['median']:.6f}")
        
        if 'db_error_stats' in metrics and 'overall' in metrics['db_error_stats']:
            db_stats = metrics['db_error_stats']['overall']
            print(f"  dB绝对误差均值: {db_stats['mean']:.2f} dB")
            print(f"  dB绝对误差中位数: {db_stats['median']:.2f} dB")
            print(f"  dB绝对误差95%分位数: {db_stats['95th_percentile']:.2f} dB")
    def print_model_first_10_params(self, model):
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
        
        if len(all_params) >= 500:
            print(f"前10个参数值:")
            for i, val in enumerate(all_params[:500]):
                print(f"  参数[{i}] = {val:.8f}")
            
            print(f"\n参数统计:")
            print(f"  最小值: {min(all_params[:500]):.8f}")
            print(f"  最大值: {max(all_params[:500]):.8f}")
            print(f"  平均值: {np.mean(all_params[:500]):.8f}")
            print(f"  标准差: {np.std(all_params[:500]):.8f}")
        else:
            print(f"模型参数少于10个: {len(all_params)}个")
    def predict_single_sample(self, matrix, frequency):
        """
        预测单个样本
        
        Args:
            matrix: 19x19的二进制矩阵
            frequency: 频率值已经归一化
        
        Returns:
            S参数预测值
        """
        self.model.eval()
        
        # self.print_model_first_10_params(self.model)
        # 确保输入格式正确
        if isinstance(matrix, np.ndarray):
            matrix = torch.FloatTensor(matrix)
        if isinstance(frequency, (int, float)):
            frequency = torch.FloatTensor([[frequency]])  # 已经归一化
        # print("Sample Input:\n")
        # print(matrix)
        # print(frequency)
        # 添加批次维度和通道维度
        if len(matrix.shape) == 2:  # (19, 19) - 2D张量
            matrix = matrix.unsqueeze(0).unsqueeze(0)  # 变为 (1, 1, 19, 19)
        elif len(matrix.shape) == 3:  # (1, 19, 19) 或 (19, 19, 1)
            if matrix.shape[0] == 1:  # (1, 19, 19)
                matrix = matrix.unsqueeze(0)  # 变为 (1, 1, 19, 19)
            elif matrix.shape[2] == 1:  # (19, 19, 1)
                matrix = matrix.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 1, 19, 19)
            else:
                print(f"错误: 无法识别的3D矩阵形状: {matrix.shape}")
                print("请确保矩阵是19x19的2D数组")
                return None
        else:
            print(f"错误: 不支持的矩阵维度: {len(matrix.shape)}")
            print(f"矩阵形状: {matrix.shape}")
            print("请提供19x19的2D数组或张量")
            return None
        
        # print(f"处理后矩阵形状: {matrix.shape}")
        # print(f"处理后频率形状: {frequency.shape}")
        matrix = matrix.to(self.device)
        frequency = frequency.to(self.device)
        test = (matrix, frequency)
        print(test)
        with torch.no_grad():
            predictions = self.model(matrix, frequency)
        
        # 转换为numpy数组并解析为字典
        predictions_np = predictions.cpu().numpy().flatten()
        print(predictions_np)
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
        # print("Predict:s_param_dict\n")
        # print(s_param_dict)
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
            # dB误差 (每个S参数)
            s_param_complex_names = ['S11', 'S21', 'S12', 'S22']
            sample_error['db_errors'] = {}
            for idx, s_name in enumerate(s_param_complex_names):
                real_idx = idx * 2
                imag_idx = idx * 2 + 1
                
                pred_complex = all_predictions[i, real_idx] + 1j * all_predictions[i, imag_idx]
                true_complex = all_targets[i, real_idx] + 1j * all_targets[i, imag_idx]
                
                pred_db = self._linear_to_db(np.abs(pred_complex))
                true_db = self._linear_to_db(np.abs(true_complex))
                db_error = abs(pred_db - true_db)
                
                sample_error['db_errors'][s_name] = {
                    'pred_db': pred_db,
                    'true_db': true_db,
                    'db_error': db_error
                }
                
                total_db_error += db_error
                valid_db_outputs += 1

            # 计算平均相对误差
            if valid_outputs > 0:
                avg_rel_error = total_rel_error / valid_outputs
            else:
                avg_rel_error = float('inf')
            
            # 计算平均dB误差
            if valid_db_outputs > 0:
                avg_db_error = total_db_error / valid_db_outputs
            else:
                avg_db_error = float('inf')

            sample_errors.append({
                'sample_index': i,
                'errors': sample_error,
                'avg_rel_error': avg_rel_error,
                'avg_db_error': avg_db_error,
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
        linear_error_stats = {}
        db_error_stats = {}
        
        for name in s_param_names:
            rel_errors = []
            for sample in sample_errors:
                error = sample['errors'][name]['rel_error']
                if error != float('inf'):
                    rel_errors.append(error)
            
            if rel_errors:
                linear_error_stats[name] = {
                    'min_error': np.min(rel_errors),
                    'max_error': np.max(rel_errors),
                    'mean_error': np.mean(rel_errors),
                    'median_error': np.median(rel_errors),
                    'std_error': np.std(rel_errors),
                    'p95_error': np.percentile(rel_errors, 95),
                    'p99_error': np.percentile(rel_errors, 99)
                }
        # dB误差统计
        s_param_complex_names = ['S11', 'S21', 'S12', 'S22']
        for s_name in s_param_complex_names:
            db_errors = []
            for sample in sample_errors:
                if s_name in sample['errors']['db_errors']:
                    db_error = sample['errors']['db_errors'][s_name]['db_error']
                    if db_error != float('inf'):
                        db_errors.append(db_error)
            
            if db_errors:
                db_error_stats[s_name] = {
                    'min_error': np.min(db_errors),
                    'max_error': np.max(db_errors),
                    'mean_error': np.mean(db_errors),
                    'median_error': np.median(db_errors),
                    'std_error': np.std(db_errors),
                    'p95_error': np.percentile(db_errors, 95),
                    'p99_error': np.percentile(db_errors, 99)
                }
        # 打印线性误差统计
        print("\n📊 线性相对误差统计:")
        for name, stats in linear_error_stats.items():
            print(f"\n  {name}:")
            print(f"    平均误差: {stats['mean_error']:.2f}%")
            print(f"    中位数误差: {stats['median_error']:.2f}%")
            print(f"    95%分位数: {stats['p95_error']:.2f}%")
        
        # 打印dB误差统计
        print("\n🔍 dB误差统计 (工程上更有意义):")
        for s_name, stats in db_error_stats.items():
            print(f"\n  {s_name}:")
            print(f"    平均dB误差: {stats['mean_error']:.2f} dB")
            print(f"    中位数dB误差: {stats['median_error']:.2f} dB")
            print(f"    95%分位数: {stats['p95_error']:.2f} dB")
            print(f"    最大dB误差: {stats['max_error']:.2f} dB")
        
        # 找出误差最大和最小的样本 (基于dB误差)
        valid_samples = [s for s in sample_errors if s['avg_db_error'] != float('inf')]
        
        if valid_samples:
            min_error_sample = min(valid_samples, key=lambda x: x['avg_db_error'])
            max_error_sample = max(valid_samples, key=lambda x: x['avg_db_error'])
            
            print(f"\n{'='*80}")
            print("样本误差极值分析 (基于dB误差)")
            print(f"{'='*80}")
            
            print(f"\n🎯 最佳预测样本 (索引: {min_error_sample['sample_index']}):")
            print(f"  平均dB误差: {min_error_sample['avg_db_error']:.2f} dB")
            print(f"  平均相对误差: {min_error_sample['avg_rel_error']:.2f}%")
            print(f"  频率: {min_error_sample['frequency']:.1f} GHz")
            
            print(f"\n⚠️ 最差预测样本 (索引: {max_error_sample['sample_index']}):")
            print(f"  平均dB误差: {max_error_sample['avg_db_error']:.2f} dB")
            print(f"  平均相对误差: {max_error_sample['avg_rel_error']:.2f}%")
            print(f"  频率: {max_error_sample['frequency']:.1f} GHz")
        
        return linear_error_stats, db_error_stats, min_error_sample, max_error_sample
    
    def create_error_distribution_plot(self, sample_errors, save_path=None):
        """创建误差分布图 - 增强版包含dB误差"""
        s_param_names = ['S11', 'S21', 'S12', 'S22']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 绘制每个S参数的dB误差分布
        for i, s_name in enumerate(s_param_names):
            if i >= len(axes):
                break
                
            db_errors = []
            for sample in sample_errors:
                if s_name in sample['errors']['db_errors']:
                    error = sample['errors']['db_errors'][s_name]['db_error']
                    if error != float('inf'):
                        db_errors.append(error)
            
            if db_errors:
                ax = axes[i]
                # 绘制直方图
                n, bins, patches = ax.hist(db_errors, bins=50, alpha=0.7, 
                                          color='skyblue', edgecolor='black')
                ax.axvline(np.mean(db_errors), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(db_errors):.2f} dB')
                ax.axvline(np.median(db_errors), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(db_errors):.2f} dB')
                
                # 添加文本信息
                ax.text(0.02, 0.98, f'Mean: {np.mean(db_errors):.2f} dB\n'
                        f'Std: {np.std(db_errors):.2f} dB\n'
                        f'Max: {np.max(db_errors):.2f} dB',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('dB Error (dB)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'{s_name} dB Error Distribution', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('S-Parameter dB Error Distribution', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"dB误差分布图已保存到: {save_path}")
        
        return fig, axes
    
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
        print(f"  最终训练损失: {train_losses[-1]:.10f}")
        print(f"  最终验证损失: {val_losses[-1]:.10f}")
        print(f"  最佳验证损失: {self.best_val_loss:.10f}")
        
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
    # 保留其他方法不变，但可以添加dB相关可视化
    def plot_db_comparison(self, predictions, targets, frequencies=None, 
                          dataset_name="Test Set", save_path=None):
        """
        绘制dB值对比图
        
        Args:
            predictions: 预测值数组 (n_samples, 8)
            targets: 真实值数组 (n_samples, 8)
            frequencies: 频率数组 (n_samples, 1) 可选
            dataset_name: 数据集名称
            save_path: 保存路径（可选）
        """
        s_param_names = ['S11', 'S21', 'S12', 'S22']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, s_name in enumerate(s_param_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            real_idx = i * 2
            imag_idx = i * 2 + 1
            
            # 转换为复数和dB
            pred_complex = predictions[:, real_idx] + 1j * predictions[:, imag_idx]
            true_complex = targets[:, real_idx] + 1j * targets[:, imag_idx]
            
            pred_db = self._linear_to_db(np.abs(pred_complex))
            true_db = self._linear_to_db(np.abs(true_complex))
            
            # 绘制散点图
            if frequencies is not None:
                scatter = ax.scatter(true_db, pred_db, c=frequencies.flatten(), 
                                   cmap='viridis', alpha=0.6, s=30, 
                                   edgecolors='white', linewidth=0.5)
                if i == 0:
                    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
                    cbar.set_label('Normalized Frequency', fontsize=10)
            else:
                ax.scatter(true_db, pred_db, alpha=0.6, s=30, 
                          color='steelblue', edgecolors='white', linewidth=0.5)
            
            # 添加对角线
            min_db = min(np.min(true_db), np.min(pred_db))
            max_db = max(np.max(true_db), np.max(pred_db))
            ax.plot([min_db, max_db], [min_db, max_db], 
                   'r--', alpha=0.7, linewidth=2, label='Perfect Prediction')
            
            # 计算dB误差指标
            db_mae = np.mean(self._linear_to_db(np.abs(pred_complex - true_complex)))
            db_rmse = np.sqrt(np.mean(self._linear_to_db(np.abs(pred_complex - true_complex)) ** 2))
            # db_corr = np.corrcoef(pred_db, true_db)[0, 1]
            
            # 设置图形属性
            ax.set_xlabel('True dB Value', fontsize=12)
            ax.set_ylabel('Predicted dB Value', fontsize=12)
            ax.set_title(f'{s_name} dB Comparison\n'
                        f'MAE: {db_mae:.2f} dB, RMSE: {db_rmse:.2f} dB', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # 添加统计信息文本框
            textstr = f'MAE: {db_mae:.2f} dB\nRMSE: {db_rmse:.2f} dB\n'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.suptitle(f'Predicted vs True dB Values - {dataset_name}\n'
                    f'S-Parameter Magnitude Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"dB对比图已保存到: {save_path}")
        
        return fig, axes
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
    def create_db_performance_report(self, predictions, targets, output_dir=None):
        """
        创建dB性能报告
        
        Args:
            predictions: 预测值
            targets: 真实值
            output_dir: 输出目录
        """
        print("\n" + "="*80)
        print("dB性能评估报告")
        print("="*80)
        
        s_param_names = ['S11', 'S21', 'S12', 'S22']
        
        report_lines = []
        report_lines.append("dB Performance Evaluation Report")
        report_lines.append("="*80)
        
        for i, s_name in enumerate(s_param_names):
            real_idx = i * 2
            imag_idx = i * 2 + 1
            
            # 转换为复数和dB
            pred_complex = predictions[:, real_idx] + 1j * predictions[:, imag_idx]
            true_complex = targets[:, real_idx] + 1j * targets[:, imag_idx]
            
            pred_db = self._linear_to_db(np.abs(pred_complex))
            true_db = self._linear_to_db(np.abs(true_complex))
            
            # 计算各种指标
            db_mae = np.mean(np.abs(pred_db - true_db))
            db_rmse = np.sqrt(np.mean((pred_db - true_db) ** 2))
            # db_corr = np.corrcoef(pred_db, true_db)[0, 1]
            
            # 分区评估
            pred_mag = np.abs(pred_complex)
            true_mag = np.abs(true_complex)
            
            large_signal_mask = true_mag > 0.05
            small_signal_mask = true_mag < 0.01
            
            large_signal_error = np.mean(np.abs(pred_db[large_signal_mask] - true_db[large_signal_mask])) \
                if np.sum(large_signal_mask) > 0 else np.nan
            
            small_signal_error = np.mean(np.abs(pred_db[small_signal_mask] - true_db[small_signal_mask])) \
                if np.sum(small_signal_mask) > 0 else np.nan
            
            report_lines.append(f"\n{s_name}:")
            report_lines.append(f"  Overall dB MAE: {db_mae:.2f} dB")
            report_lines.append(f"  Overall dB RMSE: {db_rmse:.2f} dB")
            # report_lines.append(f"  dB Correlation: {db_corr:.4f}")
            
            if not np.isnan(large_signal_error):
                report_lines.append(f"  Large Signal (|S|>0.05) dB MAE: {large_signal_error:.2f} dB "
                                   f"({np.sum(large_signal_mask)} samples)")
            
            if not np.isnan(small_signal_error):
                report_lines.append(f"  Small Signal (|S|<0.01) dB MAE: {small_signal_error:.2f} dB "
                                   f"({np.sum(small_signal_mask)} samples)")
        
        # 整体dB性能
        all_pred_db = []
        all_true_db = []
        for i in range(4):
            real_idx = i * 2
            imag_idx = i * 2 + 1
            pred_complex = predictions[:, real_idx] + 1j * predictions[:, imag_idx]
            true_complex = targets[:, real_idx] + 1j * targets[:, imag_idx]
            all_pred_db.extend(self._linear_to_db(np.abs(pred_complex)))
            all_true_db.extend(self._linear_to_db(np.abs(true_complex)))
        
        overall_db_mae = np.mean(np.abs(np.array(all_pred_db) - np.array(all_true_db)))
        overall_db_rmse = np.sqrt(np.mean((np.array(all_pred_db) - np.array(all_true_db)) ** 2))
        
        report_lines.append(f"\n{'='*80}")
        report_lines.append("Overall dB Performance:")
        report_lines.append(f"  Average dB MAE: {overall_db_mae:.2f} dB")
        report_lines.append(f"  Average dB RMSE: {overall_db_rmse:.2f} dB")
        
        # 打印到控制台
        for line in report_lines:
            print(line)
        
        # 保存到文件
        if output_dir:
            report_path = os.path.join(output_dir, 'db_performance_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"\ndB性能报告已保存到: {report_path}")
        
        return report_lines
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
        print("residuals的样本总数：", len(residuals))
        print(f"残差最大值：{np.max(residuals):.4f}")
        print(f"残差最小值：{np.min(residuals):.4f}")
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
    def plot_frequency_analysis(self, predictions, targets, frequencies, 
                           save_path=None, freq_range=(0.1, 30)):
        """
        频率分析绘图函数 - 针对0-30GHz频率范围
        
        Args:
            predictions: 模型预测值
            targets: 真实值
            frequencies: 归一化的频率值 (0-1之间)
            save_path: 保存路径
            freq_range: 原始频率范围 (GHz) - 默认为(0, 30)
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        # 1. 反归一化 - 使用正确的公式
        freq_min, freq_max = freq_range
        
        freq_ghz = frequencies.flatten() * (freq_max - freq_min) + freq_min
        
        print(f"📊 频率分析信息:")
        print(f"  归一化频率范围: [{frequencies.min():.3f}, {frequencies.max():.3f}]")
        print(f"  反归一化后范围: [{freq_ghz.min():.2f}, {freq_ghz.max():.2f}] GHz")
        print(f"  使用的原始频率范围: {freq_min}-{freq_max} GHz")
        
        # 2. 定义0-30GHz的分组边界和标签
        # 可以根据需要调整分组粒度
        freq_bins = [0, 5, 10, 15, 20, 25, 30]  # 0-30GHz，每5GHz一组
        freq_labels = ['0-5GHz', '5-10GHz', '10-15GHz', '15-20GHz', '20-25GHz', '25-30GHz']
        
        # 或者如果数据分布不均匀，可以按分位数分组
        # percentiles = [0, 20, 40, 60, 80, 100]
        # freq_bins = np.percentile(freq_ghz, percentiles)
        # freq_labels = [f'{freq_bins[i]:.1f}-{freq_bins[i+1]:.1f}GHz' for i in range(len(freq_bins)-1)]
        
        # 3. 检查频率是否在预期范围内
        if freq_ghz.min() < 0 or freq_ghz.max() > 30:
            print(f"⚠️ 警告: 频率数据超出预期范围0-30GHz!")
            print(f"  数据范围: {freq_ghz.min():.2f}-{freq_ghz.max():.2f} GHz")
            
            # 动态调整边界以包含所有数据
            freq_bins = [
                0,
                max(5, np.floor(freq_ghz.min()/5)*5 + 5),
                max(10, np.floor(freq_ghz.min()/10)*10 + 10),
                max(15, np.floor(freq_ghz.min()/15)*15 + 15),
                max(20, np.floor(freq_ghz.min()/20)*20 + 20),
                max(25, np.floor(freq_ghz.min()/25)*25 + 25),
                30
            ]
            print(f"  调整后边界: {freq_bins}")
        
        # 4. 创建频率分组
        try:
            freq_groups = pd.cut(freq_ghz, bins=freq_bins, labels=freq_labels, 
                                include_lowest=True)
        except ValueError as e:
            print(f"❌ 分组错误: {e}")
            print(f"  频率范围: {freq_ghz.min():.2f}-{freq_ghz.max():.2f}")
            print(f"  分组边界: {freq_bins}")
            return None
        
        # 5. 计算每个频率组的误差指标
        group_metrics = {}
        for group in freq_labels:
            mask = freq_groups == group
            if np.sum(mask) > 10:  # 确保统计可靠性
                group_pred = predictions[mask]
                group_true = targets[mask]
                
                # 计算基础指标
                mae = mean_absolute_error(group_true, group_pred)
                rmse = np.sqrt(mean_squared_error(group_true, group_pred))
                r2 = r2_score(group_true, group_pred)
                
                # 计算绝对误差统计
                abs_errors = np.abs(group_true - group_pred)
                
                # 计算dB误差（如果适用）
                db_errors = []
                for i in range(0, group_true.shape[1], 2):
                    if i+1 < group_true.shape[1]:  # 确保有实部和虚部
                        true_complex = group_true[:, i] + 1j * group_true[:, i+1]
                        pred_complex = group_pred[:, i] + 1j * group_pred[:, i+1]
                        error = np.abs(true_complex-pred_complex)
                        # true_db = 20 * np.log10(np.abs(true_complex) + 1e-10)
                        # pred_db = 20 * np.log10(np.abs(pred_complex) + 1e-10)
                        db_errors.extend(self._linear_to_db(error))
                
                group_metrics[group] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'count': np.sum(mask),
                    'abs_error_mean': np.mean(abs_errors),
                    'abs_error_median': np.median(abs_errors),
                    'abs_error_95th': np.percentile(abs_errors, 95),
                    'db_error_mean': np.mean(db_errors) if db_errors else None,
                    'db_error_median': np.median(db_errors) if db_errors else None,
                    'db_error_95th': np.percentile(db_errors, 95) if db_errors else None,
                    'freq_range': f"{freq_ghz[mask].min():.1f}-{freq_ghz[mask].max():.1f} GHz"
                }
        
        # 6. 创建图形 - 调整为3x2布局
        fig, axes = plt.subplots(3, 2, figsize=(18, 20))
        axes = axes.flatten()
        
        # 确保有足够的组
        valid_groups = [g for g in freq_labels if g in group_metrics]
        if not valid_groups:
            print("❌ 没有有效的频率组!")
            return None
        
        colors = sns.color_palette("husl", len(valid_groups))
        
        # 子图1: 各频率组的样本数量分布
        ax1 = axes[0]
        counts = [group_metrics[g]['count'] for g in valid_groups]
        total_samples = sum(counts)
        
        # 使用条形图显示样本数量
        bars1 = ax1.bar(valid_groups, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Frequency Range', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_title('Sample Distribution by Frequency', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数量标签和百分比
        for i, (bar, count) in enumerate(zip(bars1, counts)):
            height = bar.get_height()
            percentage = (count / total_samples) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 子图2: 各频率组的R²分数
        ax2 = axes[1]
        r2_values = [group_metrics[g]['R2'] for g in valid_groups]
        
        # 创建水平条形图，方便查看
        y_pos = np.arange(len(valid_groups))
        bars2 = ax2.barh(y_pos, r2_values, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(valid_groups, fontsize=10)
        ax2.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax2.set_title('R² Score by Frequency Range', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, 1)  # R²通常在0-1之间
        
        # 添加R²值标签
        for bar, r2 in zip(bars2, r2_values):
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{r2:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 子图3: 各频率组的MAE和RMSE对比
        ax3 = axes[2]
        mae_values = [group_metrics[g]['MAE'] for g in valid_groups]
        rmse_values = [group_metrics[g]['RMSE'] for g in valid_groups]
        
        x = np.arange(len(valid_groups))
        width = 0.35
        
        bars_mae = ax3.bar(x - width/2, mae_values, width, label='MAE', 
                        color='skyblue', edgecolor='black', linewidth=1.2)
        bars_rmse = ax3.bar(x + width/2, rmse_values, width, label='RMSE', 
                            color='lightcoral', edgecolor='black', linewidth=1.2)
        
        ax3.set_xlabel('Frequency Range', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Error Value', fontsize=12, fontweight='bold')
        ax3.set_title('MAE and RMSE by Frequency Range', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(valid_groups, rotation=45, fontsize=10)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # 添加误差值标签
        for bars in [bars_mae, bars_rmse]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 子图4: 绝对误差统计
        ax4 = axes[3]
        abs_means = [group_metrics[g]['abs_error_mean'] for g in valid_groups]
        abs_medians = [group_metrics[g]['abs_error_median'] for g in valid_groups]
        abs_95th = [group_metrics[g]['abs_error_95th'] for g in valid_groups]
        
        x = np.arange(len(valid_groups))
        width = 0.25
        
        ax4.bar(x - width, abs_means, width, label='Mean', color='lightgreen', 
                edgecolor='black', linewidth=1.2)
        ax4.bar(x, abs_medians, width, label='Median', color='gold', 
                edgecolor='black', linewidth=1.2)
        ax4.bar(x + width, abs_95th, width, label='95th Percentile', color='lightcoral', 
                edgecolor='black', linewidth=1.2)
        
        ax4.set_xlabel('Frequency Range', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
        ax4.set_title('Absolute Error Statistics by Frequency', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(valid_groups, rotation=45, fontsize=10)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 子图5: 整体预测误差随频率变化（散点图）
        ax5 = axes[4]
        sample_mae = np.mean(np.abs(predictions - targets), axis=1)
        
        # 使用渐变色表示误差大小
        scatter = ax5.scatter(freq_ghz, sample_mae, alpha=0.6, s=40, 
                            c=sample_mae, cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # 添加移动平均线
        if len(freq_ghz) > 20:
            # 按频率排序
            sort_idx = np.argsort(freq_ghz)
            freq_sorted = freq_ghz[sort_idx]
            mae_sorted = sample_mae[sort_idx]
            
            # 计算移动平均
            window_size = max(1, len(freq_ghz) // 50)  # 更小的窗口，适应更长的频率范围
            mae_moving_avg = np.convolve(mae_sorted, np.ones(window_size)/window_size, mode='valid')
            freq_moving_avg = freq_sorted[window_size-1:]
            
            ax5.plot(freq_moving_avg, mae_moving_avg, 'r-', linewidth=2.5, label=f'{window_size}-point Moving Avg')
            ax5.legend(loc='upper right')
        
        ax5.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Sample MAE', fontsize=12, fontweight='bold')
        ax5.set_title('Prediction Error vs Frequency', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.set_xlim(0, 30)  # 设置x轴范围为0-30 GHz
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('MAE', fontsize=11, fontweight='bold')
        
        # 子图6: dB误差统计（如果可用）
        ax6 = axes[5]
        db_means = [group_metrics[g]['db_error_mean'] for g in valid_groups 
                    if group_metrics[g]['db_error_mean'] is not None]
        db_medians = [group_metrics[g]['db_error_median'] for g in valid_groups 
                    if group_metrics[g]['db_error_median'] is not None]
        
        if db_means:
            valid_groups_db = [g for g in valid_groups if group_metrics[g]['db_error_mean'] is not None]
            x = np.arange(len(valid_groups_db))
            width = 0.35
            
            ax6.bar(x - width/2, db_means, width, label='Mean dB Error', 
                    color='cornflowerblue', edgecolor='black', linewidth=1.2)
            ax6.bar(x + width/2, db_medians, width, label='Median dB Error', 
                    color='salmon', edgecolor='black', linewidth=1.2)
            
            ax6.set_xlabel('Frequency Range', fontsize=12, fontweight='bold')
            ax6.set_ylabel('dB Error', fontsize=12, fontweight='bold')
            ax6.set_title('dB Error Statistics by Frequency', fontsize=14, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(valid_groups_db, rotation=45, fontsize=10)
            ax6.legend(loc='upper right')
            ax6.grid(True, alpha=0.3, linestyle='--')
            
            # 添加dB误差值标签
            for i, (mean_val, median_val) in enumerate(zip(db_means, db_medians)):
                ax6.text(i - width/2, mean_val + 0.05, f'{mean_val:.2f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax6.text(i + width/2, median_val + 0.05, f'{median_val:.2f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No dB Error Data Available', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            ax6.axis('off')
        
        # 设置总标题
        plt.suptitle('Frequency-based Performance Analysis (0-30 GHz)\n'
                    f'Total Samples: {total_samples}', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 7. 打印详细的统计信息
        print("\n" + "="*60)
        print("频率分组性能统计 (0-30 GHz):")
        print("="*60)
        
        for group in valid_groups:
            metrics = group_metrics[group]
            print(f"\n{group} ({metrics['freq_range']}):")
            print(f"  样本数: {metrics['count']} ({metrics['count']/total_samples*100:.1f}%)")
            print(f"  R²: {metrics['R2']:.4f}")
            print(f"  MAE: {metrics['MAE']:.6f}")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  绝对误差均值: {metrics['abs_error_mean']:.6f}")
            print(f"  绝对误差中位数: {metrics['abs_error_median']:.6f}")
            print(f"  绝对误差95%分位数: {metrics['abs_error_95th']:.6f}")
            
            if metrics['db_error_mean'] is not None:
                print(f"  dB误差均值: {metrics['db_error_mean']:.2f} dB")
                print(f"  dB误差中位数: {metrics['db_error_median']:.2f} dB")
                print(f"  dB误差95%分位数: {metrics['db_error_95th']:.2f} dB")
        
        # 8. 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ 频率分析图已保存到: {save_path}")
        
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