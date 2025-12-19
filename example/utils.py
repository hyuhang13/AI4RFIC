# utils/checkpoint.py
import torch
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.font_manager as fm

from matplotlib import rcParams

def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, filepath):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath} (Epoch {epoch})")
    print(filepath)
    # loaded_checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
    # saved_state = loaded_checkpoint['model_state_dict']
    # print("has been saved")
    # # for name, tensor in saved_state.items():
    # #     print(f"  {name}: {tensor.shape}")

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点"""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None
    
    # 按epoch编号排序
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

def load_model(checkpoint_path, model, optimizer=None):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('train_losses', []), checkpoint.get('val_losses', [])
def print_training_progress(epoch, train_loss, val_loss, lr, best_val_loss):
    """打印训练进度"""
    print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | LR: {lr:.6f} | "
          f"Best Val: {best_val_loss:.6f}")
class TrainingVisualizer:
    def __init__(self):
        self.fig = None
        self.axes = None
        
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'] 
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
    def setup_plots(self):
        """设置绘图区域"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Neural Network Training Monitoring and Performance Evaluation', 
                         fontsize=16, fontweight='bold')
        return self.fig, self.axes
    
    def plot_training_curves(self, train_losses, val_losses, axes=None):
        """绘制训练和验证损失曲线"""
        if axes is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = axes
            
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7, linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7, linewidth=2)
        
        # 标记最佳验证损失点
        best_epoch = np.argmin(val_losses)
        best_loss = val_losses[best_epoch]
        ax.scatter(best_epoch + 1, best_loss, color='red', s=100, zorder=5, 
                  label=f'Best Validation (epoch {best_epoch + 1})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 添加文本信息
        textstr = f'Final Train Loss: {train_losses[-1]:.6f}\nFinal Val Loss: {val_losses[-1]:.6f}\nBest Val Loss: {best_loss:.6f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax
    
    def plot_prediction_vs_actual(self,preprocessor, model, data_loader, y_scaler, output_names, axes=None):
        """绘制预测值 vs 真实值散点图"""
        device = next(model.parameters()).device
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # 合并所有批次
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 反标准化
        predictions_original = all_predictions
        targets_original = all_targets
        # predictions_original = preprocessor.inverse_transform_y(all_predictions)
        # targets_original = preprocessor.inverse_transform_y(all_targets)
        # predictions_original = y_scaler.inverse_transform(all_predictions)
        # targets_original = y_scaler.inverse_transform(all_targets)
        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Predicted vs Actual Values', fontsize=16)
            
            axes_flat = axes.flatten()
        else:

            axes_flat = axes
        

        for i, (ax, output_name) in enumerate(zip(axes_flat[:len(output_names)], output_names)):
        
            pred_i = predictions_original[:, i]
            target_i = targets_original[:, i]
            
            # 计算评估指标
            r2 = r2_score(target_i, pred_i)
            mae = mean_absolute_error(target_i, pred_i)
            rmse = np.sqrt(mean_squared_error(target_i, pred_i))
            
            # 绘制散点图
            ax.scatter(target_i, pred_i, alpha=0.6, s=30)
            
            # 绘制完美预测线
            min_val = min(target_i.min(), pred_i.min())
            max_val = max(target_i.max(), pred_i.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel(f'Actual {output_name}')
            ax.set_ylabel(f'Predicted {output_name}')
            ax.set_title(f'{output_name}\nR² = {r2:.4f}, MAE = {mae:.2e}, RMSE = {rmse:.2e}')
            ax.grid(True, alpha=0.3)
            
            # 添加文本信息
            error_percentage = mae / (np.abs(target_i).mean() + 1e-8) * 100
            ax.text(0.05, 0.95, f'Mean Error: {error_percentage:.2f}%', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(len(output_names), len(axes_flat)):
            axes.flatten()[i].set_visible(False)
        
        return predictions_original, targets_original
    
    def plot_error_distribution(self, predictions, targets, output_names, axes=None):
        """绘制误差分布"""
        errors = predictions - targets
        
        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (ax, output_name) in enumerate(zip(axes.flatten()[:len(output_names)], output_names)):
            error_i = errors[:, i]
            
            # 计算统计信息
            mean_error = error_i.mean()
            std_error = error_i.std()
            max_abs_error = np.abs(error_i).max()
            
            # 绘制误差分布直方图
            ax.hist(error_i, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_error:.2e}')
            ax.axvline(mean_error + std_error, color='orange', linestyle='--', linewidth=1)
            ax.axvline(mean_error - std_error, color='orange', linestyle='--', linewidth=1)
            
            ax.set_xlabel(f'{output_name} Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{output_name} Error Distribution\nMean: {mean_error:.2e}, Std: {std_error:.2e}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(output_names), len(axes.flatten())):
            axes.flatten()[i].set_visible(False)
    
    def plot_residuals(self, predictions, targets, output_names, axes=None):
        """绘制残差图"""
        errors = predictions - targets
        
        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (ax, output_name) in enumerate(zip(axes.flatten()[:len(output_names)], output_names)):
            target_i = targets[:, i]
            error_i = errors[:, i]
            
            ax.scatter(target_i, error_i, alpha=0.6, s=30)
            ax.axhline(y=0, color='red', linestyle='-', linewidth=2)
            
            # 添加趋势线
            if len(target_i) > 1:
                z = np.polyfit(target_i, error_i, 1)
                p = np.poly1d(z)
                ax.plot(target_i, p(target_i), "r--", alpha=0.8, linewidth=1)
            
            ax.set_xlabel(f'Actual {output_name}')
            ax.set_ylabel(f'{output_name} Residual')
            ax.set_title(f'{output_name} Residual Plot')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(output_names), len(axes.flatten())):
            axes.flatten()[i].set_visible(False)
    
    def create_comprehensive_report(self, preprocessor,model, train_loader, val_loader, y_scaler, 
                                  output_names, train_losses, val_losses, save_path=None):
        """创建综合训练报告"""
        fig, axes = self.setup_plots()
        
        # 1. 训练曲线
        self.plot_training_curves(train_losses, val_losses, axes[0, 0])
        
        # 2. 验证集预测 vs 真实值
        predictions, targets = self.plot_prediction_vs_actual(
            preprocessor,
            model, val_loader, y_scaler, output_names, 
            axes=[axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
        )
        
        # 3. 误差分布 (使用新的图形)
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
        fig2.suptitle('Error Analysis', fontsize=16, fontweight='bold')
        self.plot_error_distribution(predictions, targets, output_names, axes2)
        
        # 4. 残差图 (使用新的图形)
        fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
        fig3.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
        self.plot_residuals(predictions, targets, output_names, axes3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(f"{save_path}_training_curves.png", dpi=300, bbox_inches='tight')
            fig2.savefig(f"{save_path}_error_distribution.png", dpi=300, bbox_inches='tight')
            fig3.savefig(f"{save_path}_residuals.png", dpi=300, bbox_inches='tight')
            print(f"Training report saved to: {save_path}_*.png")
        
        plt.show()
        
        return predictions, targets   

def print_training_progress(epoch, train_loss, val_loss, learning_rate, best_val_loss=None):
    """打印训练进度"""
    if best_val_loss is not None and val_loss < best_val_loss:
        improvement = f" ↓{best_val_loss - val_loss:.2e}" if best_val_loss != float('inf') else ""
    else:
        improvement = ""
    
    print(f'Epoch {epoch:4d}: '
          f'Train Loss = {train_loss:.6f}, '
          f'Val Loss = {val_loss:.6f}{improvement}, '
          f'LR = {learning_rate:.2e}')