# models/model_manager.py (部分更新)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np

from utils import save_checkpoint, find_latest_checkpoint, load_model
from utils import print_training_progress
from config import TRAIN_CONFIG, DEVICE, OPTIMIZER_CONFIG, SCHEDULER_CONFIG, MODEL_CONFIG

class ModelManager:
    def __init__(self, model, checkpoint_dir=None):
        self.model = model
        self.checkpoint_dir = checkpoint_dir or TRAIN_CONFIG['checkpoint_dir']
        self.device = DEVICE
        self.best_val_loss = float('inf')
        
        # 创建检查点目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def setup_training(self, learning_rate=None, weight_decay=None):
        """设置训练组件 - 使用新的优化器和调度器"""
        learning_rate = learning_rate or OPTIMIZER_CONFIG['lr']
        weight_decay = weight_decay or OPTIMIZER_CONFIG['weight_decay']
        
        self.criterion = nn.MSELoss()
        
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
        
        print(f"使用优化器: {type(self.optimizer).__name__}")
        print(f"使用调度器: ReduceLROnPlateau")
        print(f"初始学习率: {learning_rate}")
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            
            # 梯度裁剪（可选，防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train_model(self, train_loader, val_loader, epochs=None, resume=True, 
                   print_every=20, save_every=50):
        """训练模型的主函数 - 添加进度打印"""
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
                
                # 更新最佳验证损失
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                
                # 学习率调度（基于验证损失）
                self.scheduler.step(avg_val_loss)
                
                # 保存最佳模型
                if avg_val_loss == self.best_val_loss:
                    best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                
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
    
    def evaluate_model(self,preprocessor, data_loader, y_scaler, output_names,dataset_name="dataset@"):
        """评估模型性能"""
        device = next(self.model.parameters()).device
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = self.model(X_batch)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # 合并所有批次
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 反标准化
        predictions_original = preprocessor.inverse_transform_y(all_predictions)
        targets_original = preprocessor.inverse_transform_y(all_targets)
        # predictions_original = y_scaler.inverse_transform(all_predictions)
        # targets_original = y_scaler.inverse_transform(all_targets)
        metrics = self._calculate_comprehensive_metrics(predictions_original, targets_original, output_names, dataset_name)
        
        return predictions_original, targets_original, metrics
        # 计算每个输出指标的评估指标
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        print("\n" + "="*80)
        print("模型性能评估报告")
        print("="*80)
        
        metrics_summary = {}
        
        for i, output_name in enumerate(output_names):
            pred_i = predictions_original[:, i]
            target_i = targets_original[:, i]
            
            r2 = r2_score(target_i, pred_i)
            mae = mean_absolute_error(target_i, pred_i)
            rmse = np.sqrt(mean_squared_error(target_i, pred_i))
            mape = np.mean(np.abs((target_i - pred_i) / (np.abs(target_i) + 1e-8))) * 100
            
            metrics_summary[output_name] = {
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            
            print(f"{output_name}:")
            print(f"  R² (决定系数): {r2:.6f}")
            print(f"  MAE (平均绝对误差): {mae:.2e}")
            print(f"  RMSE (均方根误差): {rmse:.2e}")
            print(f"  MAPE (平均绝对百分比误差): {mape:.2f}%")
            print()
        
        return predictions_original, targets_original, metrics_summary
    def _calculate_comprehensive_metrics(self, predictions, targets, output_names, dataset_name):
        """计算全面的评估指标"""
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
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
                'Mean_Relative_Error': np.mean(np.abs((pred_i - target_i) / (np.abs(target_i) + 1e-12))) * 100
            }
        
        return metrics
    
    def evaluate_single_sample_corrected(self, X_scaler, y_scalers, input_features, true_values=None, output_names=None):
        """修正后的单个样本评估函数 - 正确处理不同的逆变换"""
        self.model.eval()
        
        # 使用您的归一化方法处理输入
        input_normalized = X_scaler.transform([input_features])
        input_tensor = torch.FloatTensor(input_normalized).to(self.device)
        
        with torch.no_grad():
            prediction_normalized = self.model(input_tensor)
        
        # 将预测结果转换为numpy数组
        prediction_normalized_np = prediction_normalized.cpu().numpy()
        
        # 使用您的逆变换方法处理输出
        prediction_original = self._inverse_transform_y_single(y_scalers, prediction_normalized_np, output_names)
        
        # 如果有真实值，也进行逆变换
        if true_values is not None:
            true_values_normalized = np.array([true_values])  # 转换为2D数组
            true_values_original = self._inverse_transform_y_single(y_scalers, true_values_normalized, output_names)
            true_values_original = true_values_original[0]  # 转换回1D数组
        else:
            true_values_original = None
        
        return prediction_original[0], true_values_original
    
    def _inverse_transform_y_single(self, y_scalers, y_normalized, output_names):
        """单个样本的y逆变换 - 处理不同的变换方法"""
        y_original = np.zeros_like(y_normalized)
        
        for i, target in enumerate(output_names):
            if target in y_scalers:
                transform_info = y_scalers[target]
                transform_type = transform_info[0]
                scaler = transform_info[1]
                target_data = y_normalized[:, i].reshape(-1, 1)
                
                if transform_type == 'log+standard':
                    # 逆标准化 -> 逆对数
                    log_data = scaler.inverse_transform(target_data)
                    original_data = 10 ** log_data
                elif transform_type == 'yeo-johnson':
                    original_data = scaler.inverse_transform(target_data)
                else:  # standard
                    original_data = scaler.inverse_transform(target_data)
                
                y_original[:, i] = original_data.flatten()
        
        return y_original
    
    def create_precision_summary(self, train_metrics, test_metrics, single_sample_metrics=None):
        """创建精度总结报告"""
        print("\n" + "="*80)
        print("神经网络模型精度评估总结")
        print("="*80)
        
        # 训练集指标
        print(f"\n📊 {train_metrics['dataset']}评估结果:")
        print(f"整体指标 - R²: {train_metrics['overall']['R2']:.6f}, "
              f"MAE: {train_metrics['overall']['MAE']:.6f}, "
              f"RMSE: {train_metrics['overall']['RMSE']:.6f}")
        
        for output_name, metrics in train_metrics['per_output'].items():
            print(f"  {output_name}: R²={metrics['R2']:.4f}, "
                  f"MAE={metrics['MAE']:.2e}, "
                  f"相对误差={metrics['Mean_Relative_Error']:.2f}%")
        
        # 测试集指标
        print(f"\n📊 {test_metrics['dataset']}评估结果:")
        print(f"整体指标 - R²: {test_metrics['overall']['R2']:.6f}, "
              f"MAE: {test_metrics['overall']['MAE']:.6f}, "
              f"RMSE: {test_metrics['overall']['RMSE']:.6f}")
        
        for output_name, metrics in test_metrics['per_output'].items():
            print(f"  {output_name}: R²={metrics['R2']:.4f}, "
                  f"MAE={metrics['MAE']:.2e}, "
                  f"相对误差={metrics['Mean_Relative_Error']:.2f}%")
        
        # 单个样本预测结果
        if single_sample_metrics:
            print(f"\n🎯 单个样本预测结果:")
            for output_name, (true_val, pred_val, error_pct) in single_sample_metrics.items():
                print(f"  {output_name}: 真实值={true_val:.6e}, "
                      f"预测值={pred_val:.6e}, 误差={error_pct:.2f}%")
        
        # 性能对比
        print(f"\n📈 性能对比:")
        print(f"训练集 vs 测试集 R²差异: {train_metrics['overall']['R2'] - test_metrics['overall']['R2']:.6f}")
        print(f"训练集 vs 测试集 MAE比率: {test_metrics['overall']['MAE']/train_metrics['overall']['MAE']:.4f}")
    def save_final_model(self, filepath, X_scaler, y_scaler, train_losses, val_losses):
        """保存最终模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'X_scaler': X_scaler,
            'y_scaler': y_scaler,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'input_size': MODEL_CONFIG['input_size'],
                'output_size': MODEL_CONFIG['output_size'],
                'hidden_dims': MODEL_CONFIG['hidden_dims'],
                'dropout_rates': MODEL_CONFIG['dropout_rates'],
                'use_batchnorm': MODEL_CONFIG['use_batchnorm']
            }
        }, filepath)
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