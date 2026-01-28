# s_param_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os

class SParamVisualizer:
    """
    S参数可视化器：绘制模型预测与真实S参数的对比曲线
    """
    
    def __init__(self, model_manager, matrix: np.ndarray, device=None):
        """
        初始化可视化器
        
        Args:
            model_manager: 模型管理器
            matrix: 固定的19x19二进制矩阵
            device: 计算设备
        """
        self.model_manager = model_manager
        self.matrix = matrix
        self.device = device or model_manager.device
        
        # S参数名称
        self.s_param_names = ['S11', 'S21', 'S12', 'S22']
        self.s_component_names = ['real', 'imag', 'mag', 'phase']
        
    def load_ground_truth(self, filepath: str) -> pd.DataFrame:
        """
        加载真实数据TXT文件
        
        Args:
            filepath: TXT文件路径
            
        Returns:
            DataFrame包含频率和S参数
        """
        print(f"加载真实数据: {filepath}")
        
        try:
            # 读取CSV格式数据
            df = pd.read_csv(filepath)
            print(f"数据加载成功，形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            # 检查必要的列
            required_cols = ['Freq(Hz)', 'S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                            'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
            
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"缺少必要列: {col}")
            
            # 提取频率数据（Hz）
            frequencies_hz = df['Freq(Hz)'].values
            
            # 提取S参数
            ground_truth = {
                'frequencies_hz': frequencies_hz,
                'frequencies_ghz': frequencies_hz / 1e9,  # 转换为GHz
                'S11_real': df['S11_real'].values,
                'S11_imag': df['S11_imag'].values,
                'S21_real': df['S21_real'].values,
                'S21_imag': df['S21_imag'].values,
                'S12_real': df['S12_real'].values,
                'S12_imag': df['S12_imag'].values,
                'S22_real': df['S22_real'].values,
                'S22_imag': df['S22_imag'].values,
            }
            
            # 如果有幅度和相位数据，也提取
            if 'S11_mag' in df.columns:
                ground_truth.update({
                    'S11_mag': df['S11_mag'].values,
                    'S11_phase': df['S11_phase'].values,
                    'S21_mag': df['S21_mag'].values,
                    'S21_phase': df['S21_phase'].values,
                    'S12_mag': df['S12_mag'].values,
                    'S12_phase': df['S12_phase'].values,
                    'S22_mag': df['S22_mag'].values,
                    'S22_phase': df['S22_phase'].values,
                })
            
            # 打印数据统计
            print(f"频率范围: {frequencies_hz.min():.2e} Hz 到 {frequencies_hz.max():.2e} Hz")
            print(f"频率范围: {frequencies_hz.min()/1e9:.2f} GHz 到 {frequencies_hz.max()/1e9:.2f} GHz")
            print(f"数据点数: {len(frequencies_hz)}")
            
            return ground_truth
            
        except Exception as e:
            print(f"加载真实数据失败: {e}")
            raise
    
    def predict_s_params_for_frequencies(self, frequencies_hz: np.ndarray) -> Dict[str, np.ndarray]:
        """
        对给定频率数组进行预测
        
        Args:
            frequencies_hz: 频率数组（Hz）
            
        Returns:
            预测的S参数字典
        """
        print(f"开始预测 {len(frequencies_hz)} 个频率点的S参数...")
        
        # 初始化存储数组
        predictions = {
            'S11_real': np.zeros(len(frequencies_hz)),
            'S11_imag': np.zeros(len(frequencies_hz)),
            'S21_real': np.zeros(len(frequencies_hz)),
            'S21_imag': np.zeros(len(frequencies_hz)),
            'S12_real': np.zeros(len(frequencies_hz)),
            'S12_imag': np.zeros(len(frequencies_hz)),
            'S22_real': np.zeros(len(frequencies_hz)),
            'S22_imag': np.zeros(len(frequencies_hz)),
        }
        
        # 逐个频率进行预测
        for i, freq_hz in enumerate(frequencies_hz):
            if i % 100 == 0:  # 每100个点打印一次进度
                print(f"  预测进度: {i+1}/{len(frequencies_hz)}")
            
            # 将频率归一化（根据你的模型要求）
            freq_norm = (freq_hz - 100000000) / 2.99e10
            
            # 确保矩阵是连续的
            matrix = np.ascontiguousarray(self.matrix)
            
            # 预测
            s_params = self.model_manager.predict_single_sample(matrix, freq_norm)
            
            # 存储结果
            predictions['S11_real'][i] = s_params['S11_real']
            predictions['S11_imag'][i] = s_params['S11_imag']
            predictions['S21_real'][i] = s_params['S21_real']
            predictions['S21_imag'][i] = s_params['S21_imag']
            predictions['S12_real'][i] = s_params['S12_real']
            predictions['S12_imag'][i] = s_params['S12_imag']
            predictions['S22_real'][i] = s_params['S22_real']
            predictions['S22_imag'][i] = s_params['S22_imag']
        print(predictions)
        print("预测完成!")
        return predictions
    
    def calculate_magnitude_and_phase(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        计算幅度和相位
        
        Args:
            predictions: 预测的实部和虚部
            
        Returns:
            包含幅度和相位的字典
        """
        print("计算幅度和相位...")
        
        results = predictions.copy()
        
        for s_name in self.s_param_names:
            real_key = f'{s_name}_real'
            imag_key = f'{s_name}_imag'
            
            if real_key in predictions and imag_key in predictions:
                real = predictions[real_key]
                imag = predictions[imag_key]
                
                # 计算幅度
                mag = np.sqrt(real**2 + imag**2)
                mag_key = f'{s_name}_mag'
                results[mag_key] = mag
                
                # 计算相位（度）
                phase = np.degrees(np.arctan2(imag, real))
                phase_key = f'{s_name}_phase'
                results[phase_key] = phase
                
                print(f"  {s_name}: 幅度范围 [{mag.min():.4f}, {mag.max():.4f}], "
                      f"相位范围 [{phase.min():.1f}°, {phase.max():.1f}°]")
        
        return results
    
    def calculate_errors(self, predictions: Dict[str, np.ndarray], 
                        ground_truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        计算预测误差
        
        Args:
            predictions: 预测结果
            ground_truth: 真实数据
            
        Returns:
            误差统计字典
        """
        print("\n计算预测误差...")
        
        errors = {}
        
        for s_name in self.s_param_names:
            for component in ['real', 'imag', 'mag', 'phase']:
                pred_key = f'{s_name}_{component}'
                truth_key = f'{s_name}_{component}'
                
                if pred_key in predictions and truth_key in ground_truth:
                    pred = predictions[pred_key]
                    truth = ground_truth[truth_key]
                    
                    # 绝对误差
                    abs_error = np.abs(pred - truth)
                    
                    # 相对误差（对于非零值）
                    if component in ['real', 'imag', 'mag']:
                        mask = np.abs(truth) > 1e-12  # 避免除以0
                        rel_error = np.zeros_like(abs_error)
                        rel_error[mask] = abs_error[mask] / np.abs(truth[mask]) * 100
                    else:
                        rel_error = abs_error  # 相位直接使用绝对误差
                    
                    errors[f'{s_name}_{component}_abs'] = abs_error
                    errors[f'{s_name}_{component}_rel'] = rel_error
                    
                    # 打印统计
                    print(f"  {pred_key}:")
                    print(f"    绝对误差 - 均值: {abs_error.mean():.6f}, 最大: {abs_error.max():.6f}")
                    if component in ['real', 'imag', 'mag']:
                        print(f"    相对误差 - 均值: {rel_error[mask].mean():.2f}%, 最大: {rel_error[mask].max():.2f}%")
        
        return errors
    
    def plot_s_param_comparison(self, ground_truth: Dict[str, np.ndarray], 
                              predictions: Dict[str, np.ndarray],
                              save_dir: str = 's_param_plots'):
        """
        绘制S参数对比图
        
        Args:
            ground_truth: 真实数据
            predictions: 预测数据
            save_dir: 保存目录
        """
        print(f"\n开始绘制S参数对比图，保存到: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        
        frequencies_ghz = ground_truth.get('frequencies_ghz', 
                                          ground_truth['frequencies_hz'] / 1e9)
        
        # 1. 绘制实部和虚部对比（8个子图）
        fig_real_imag, axes = plt.subplots(4, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        plot_idx = 0
        for s_name in self.s_param_names:
            for component in ['real', 'imag']:
                truth_key = f'{s_name}_{component}'
                pred_key = f'{s_name}_{component}'
                
                if truth_key in ground_truth and pred_key in predictions:
                    ax = axes[plot_idx]
                    
                    # 绘制曲线
                    ax.plot(frequencies_ghz, ground_truth[truth_key], 
                           'b-', linewidth=2, label='Ground Truth', alpha=0.8)
                    ax.plot(frequencies_ghz, predictions[pred_key], 
                           'r--', linewidth=2, label='Prediction', alpha=0.8)
                    
                    # 计算误差
                    error = predictions[pred_key] - ground_truth[truth_key]
                    rmse = np.sqrt(np.mean(error**2))
                    mae = np.mean(np.abs(error))
                    
                    # 设置图形属性
                    ax.set_xlabel('Frequency (GHz)', fontsize=11)
                    ax.set_ylabel(f'{s_name} {component}', fontsize=11)
                    ax.set_title(f'{s_name} {component} Comparison\nRMSE: {rmse:.4f}, MAE: {mae:.4f}', 
                                fontsize=12, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
        
        # 调整布局
        plt.suptitle('S-Parameter Real/Imaginary Component Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, 's_param_real_imag_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"实部/虚部对比图已保存: {save_path}")
        
        # 2. 绘制幅度和相位对比（8个子图）
        if all(f'{s_name}_mag' in ground_truth for s_name in self.s_param_names):
            fig_mag_phase, axes = plt.subplots(4, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            plot_idx = 0
            for s_name in self.s_param_names:
                # 幅度
                ax_mag = axes[plot_idx]
                ax_mag.plot(frequencies_ghz, ground_truth[f'{s_name}_mag'], 
                           'b-', linewidth=2, label='Ground Truth', alpha=0.8)
                ax_mag.plot(frequencies_ghz, predictions[f'{s_name}_mag'], 
                           'r--', linewidth=2, label='Prediction', alpha=0.8)
                
                ax_mag.set_xlabel('Frequency (GHz)', fontsize=11)
                ax_mag.set_ylabel(f'|{s_name}|', fontsize=11)
                ax_mag.set_title(f'{s_name} Magnitude Comparison', fontsize=12, fontweight='bold')
                ax_mag.legend(fontsize=10)
                ax_mag.grid(True, alpha=0.3)
                
                # 相位
                ax_phase = axes[plot_idx + 1]
                ax_phase.plot(frequencies_ghz, ground_truth[f'{s_name}_phase'], 
                             'b-', linewidth=2, label='Ground Truth', alpha=0.8)
                ax_phase.plot(frequencies_ghz, predictions[f'{s_name}_phase'], 
                             'r--', linewidth=2, label='Prediction', alpha=0.8)
                
                ax_phase.set_xlabel('Frequency (GHz)', fontsize=11)
                ax_phase.set_ylabel(f'∠{s_name} (°)', fontsize=11)
                ax_phase.set_title(f'{s_name} Phase Comparison', fontsize=12, fontweight='bold')
                ax_phase.legend(fontsize=10)
                ax_phase.grid(True, alpha=0.3)
                
                plot_idx += 2
            
            plt.suptitle('S-Parameter Magnitude/Phase Comparison', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, 's_param_mag_phase_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"幅度/相位对比图已保存: {save_path}")
        
        # 3. 绘制dB对比图（幅度以dB表示）
        fig_db, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, s_name in enumerate(self.s_param_names):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # 计算dB值（20*log10(magnitude)）
            truth_mag = ground_truth.get(f'{s_name}_mag', 
                                        np.sqrt(ground_truth[f'{s_name}_real']**2 + 
                                                ground_truth[f'{s_name}_imag']**2))
            pred_mag = predictions.get(f'{s_name}_mag', 
                                      np.sqrt(predictions[f'{s_name}_real']**2 + 
                                              predictions[f'{s_name}_imag']**2))
            
            truth_db = 10 * np.log10(np.maximum(truth_mag, 1e-12))
            pred_db = 10 * np.log10(np.maximum(pred_mag, 1e-12))
            
            # 绘制dB曲线
            ax.plot(frequencies_ghz, truth_db, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
            ax.plot(frequencies_ghz, pred_db, 'r--', linewidth=2, label='Prediction', alpha=0.8)
            
            # 计算dB误差
            db_error = pred_db - truth_db
            db_rmse = np.sqrt(np.mean(db_error**2))
            db_mae = np.mean(np.abs(db_error))
            
            ax.set_xlabel('Frequency (GHz)', fontsize=11)
            ax.set_ylabel(f'{s_name} (dB)', fontsize=11)
            ax.set_title(f'{s_name} dB Comparison\nRMSE: {db_rmse:.2f} dB, MAE: {db_mae:.2f} dB', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('S-Parameter dB Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 's_param_db_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"dB对比图已保存: {save_path}")
        
        return fig_real_imag, fig_mag_phase if 'fig_mag_phase' in locals() else None, fig_db
    
    def plot_error_analysis(self, errors: Dict[str, np.ndarray], 
                          frequencies_ghz: np.ndarray,
                          save_dir: str = 's_param_plots'):
        """
        绘制误差分析图
        
        Args:
            errors: 误差字典
            frequencies_ghz: 频率数组（GHz）
            save_dir: 保存目录
        """
        print("\n绘制误差分析图...")
        
        # 1. 绘制绝对误差随频率变化
        fig_abs_error, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, s_name in enumerate(self.s_param_names):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # 收集该S参数的所有绝对误差
            abs_errors = []
            for component in ['real', 'imag', 'mag']:
                error_key = f'{s_name}_{component}_abs'
                if error_key in errors:
                    abs_errors.append(errors[error_key])
            
            if abs_errors:
                # 计算平均绝对误差
                mean_abs_error = np.mean(abs_errors, axis=0)
                
                ax.plot(frequencies_ghz, mean_abs_error, 'r-', linewidth=1.5, alpha=0.8)
                ax.fill_between(frequencies_ghz, 0, mean_abs_error, 
                               alpha=0.3, color='red')
                
                ax.set_xlabel('Frequency (GHz)', fontsize=11)
                ax.set_ylabel('Absolute Error', fontsize=11)
                ax.set_title(f'{s_name} Absolute Error vs Frequency', 
                            fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息
                stats_text = f'Mean: {mean_abs_error.mean():.4f}\nMax: {mean_abs_error.max():.4f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Absolute Error vs Frequency', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'absolute_error_vs_frequency.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"绝对误差图已保存: {save_path}")
        
        # 2. 绘制相对误差分布（直方图）
        fig_rel_error, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, s_name in enumerate(self.s_param_names):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # 收集该S参数的所有相对误差（只考虑实部、虚部、幅度）
            rel_errors = []
            for component in ['real', 'imag', 'mag']:
                error_key = f'{s_name}_{component}_rel'
                if error_key in errors:
                    error_data = errors[error_key]
                    # 过滤掉无效值（比如除0得到的无穷大）
                    valid_mask = np.isfinite(error_data) & (error_data < 1000)  # 过滤异常值
                    if np.sum(valid_mask) > 0:
                        rel_errors.extend(error_data[valid_mask])
            
            if rel_errors:
                rel_errors = np.array(rel_errors)
                
                # 绘制直方图
                n, bins, patches = ax.hist(rel_errors, bins=50, alpha=0.7, 
                                          color='skyblue', edgecolor='black', density=True)
                
                ax.set_xlabel('Relative Error (%)', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                ax.set_title(f'{s_name} Relative Error Distribution', 
                            fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息
                stats_text = f'Mean: {rel_errors.mean():.2f}%\nStd: {rel_errors.std():.2f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Relative Error Distribution', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'relative_error_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相对误差分布图已保存: {save_path}")
        
        return fig_abs_error, fig_rel_error
    
    def create_summary_report(self, ground_truth: Dict[str, np.ndarray],
                            predictions: Dict[str, np.ndarray],
                            errors: Dict[str, np.ndarray],
                            save_dir: str = 's_param_plots'):
        """
        创建评估总结报告
        
        Args:
            ground_truth: 真实数据
            predictions: 预测数据
            errors: 误差数据
            save_dir: 保存目录
        """
        print("\n创建评估总结报告...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("S参数模型预测评估报告")
        report_lines.append("="*80)
        report_lines.append(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 基本统计
        frequencies_ghz = ground_truth.get('frequencies_ghz', 
                                          ground_truth['frequencies_hz'] / 1e9)
        report_lines.append(f"\n频率范围: {frequencies_ghz.min():.2f} GHz 到 {frequencies_ghz.max():.2f} GHz")
        report_lines.append(f"频率点数: {len(frequencies_ghz)}")
        
        # 各S参数的误差统计
        for s_name in self.s_param_names:
            report_lines.append(f"\n{s_name} 误差统计:")
            
            for component in ['real', 'imag', 'mag']:
                abs_key = f'{s_name}_{component}_abs'
                rel_key = f'{s_name}_{component}_rel'
                
                if abs_key in errors:
                    abs_error = errors[abs_key]
                    abs_mean = abs_error.mean()
                    abs_max = abs_error.max()
                    
                    report_lines.append(f"  {component}绝对误差: 均值={abs_mean:.6f}, 最大={abs_max:.6f}")
                
                if rel_key in errors:
                    rel_error = errors[rel_key]
                    valid_mask = np.isfinite(rel_error) & (rel_error < 1000)
                    if np.sum(valid_mask) > 0:
                        rel_error_valid = rel_error[valid_mask]
                        rel_mean = rel_error_valid.mean()
                        rel_max = rel_error_valid.max()
                        
                        report_lines.append(f"  {component}相对误差: 均值={rel_mean:.2f}%, 最大={rel_max:.2f}%")
        
        # dB误差统计
        report_lines.append("\n\nS参数dB误差统计:")
        for s_name in self.s_param_names:
            # 计算dB值
            truth_mag = ground_truth.get(f'{s_name}_mag', 
                                        np.sqrt(ground_truth[f'{s_name}_real']**2 + 
                                                ground_truth[f'{s_name}_imag']**2))
            pred_mag = predictions.get(f'{s_name}_mag', 
                                      np.sqrt(predictions[f'{s_name}_real']**2 + 
                                              predictions[f'{s_name}_imag']**2))
            
            truth_db = 20 * np.log10(np.maximum(truth_mag, 1e-12))
            pred_db = 20 * np.log10(np.maximum(pred_mag, 1e-12))
            
            db_error = pred_db - truth_db
            db_rmse = np.sqrt(np.mean(db_error**2))
            db_mae = np.mean(np.abs(db_error))
            
            report_lines.append(f"  {s_name}: RMSE={db_rmse:.2f} dB, MAE={db_mae:.2f} dB")
        
        # 总体性能
        report_lines.append("\n\n总体性能评估:")
        
        # 计算整体绝对误差
        all_abs_errors = []
        for s_name in self.s_param_names:
            for component in ['real', 'imag']:
                error_key = f'{s_name}_{component}_abs'
                if error_key in errors:
                    all_abs_errors.extend(errors[error_key])
        
        if all_abs_errors:
            overall_abs_mean = np.mean(all_abs_errors)
            overall_abs_max = np.max(all_abs_errors)
            report_lines.append(f"  整体绝对误差: 均值={overall_abs_mean:.6f}, 最大={overall_abs_max:.6f}")
        
        # 保存报告
        report_path = os.path.join(save_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"评估报告已保存: {report_path}")
        
        # 打印报告
        print('\n'.join(report_lines))
        
        return report_path
    
    def visualize_all(self, ground_truth_file: str, 
                     save_dir: str = 's_param_evaluation'):
        """
        完整的可视化流程
        
        Args:
            ground_truth_file: 真实数据文件路径
            save_dir: 保存目录
            
        Returns:
            所有图形和数据的字典
        """
        print("\n" + "="*80)
        print("开始完整的S参数模型评估")
        print("="*80)
        
        # 1. 加载真实数据
        ground_truth = self.load_ground_truth(ground_truth_file)
        
        # 2. 使用相同的频率点进行预测
        predictions = self.predict_s_params_for_frequencies(ground_truth['frequencies_hz'])
        
        # 3. 计算幅度和相位
        predictions = self.calculate_magnitude_and_phase(predictions)
        
        # 4. 计算误差
        errors = self.calculate_errors(predictions, ground_truth)
        
        # 5. 绘制对比图
        frequencies_ghz = ground_truth.get('frequencies_ghz', 
                                          ground_truth['frequencies_hz'] / 1e9)
        
        fig_real_imag, fig_mag_phase, fig_db = self.plot_s_param_comparison(
            ground_truth, predictions, save_dir)
        
        # 6. 绘制误差分析图
        fig_abs_error, fig_rel_error = self.plot_error_analysis(
            errors, frequencies_ghz, save_dir)
        
        # 7. 创建总结报告
        report_path = self.create_summary_report(
            ground_truth, predictions, errors, save_dir)
        
        print("\n" + "="*80)
        print("评估完成!")
        print(f"所有结果已保存到: {save_dir}")
        print("="*80)
        
        # 返回所有结果
        results = {
            'ground_truth': ground_truth,
            'predictions': predictions,
            'errors': errors,
            'figures': {
                'real_imag_comparison': fig_real_imag,
                'mag_phase_comparison': fig_mag_phase,
                'db_comparison': fig_db,
                'abs_error': fig_abs_error,
                'rel_error': fig_rel_error,
            },
            'report_path': report_path
        }
        
        return results


# # 使用示例函数
# def visualize_s_param_performance(model_manager, matrix, ground_truth_file, 
#                                  output_dir='s_param_evaluation'):
#     """
#     方便的使用函数
    
#     Args:
#         model_manager: 模型管理器
#         matrix: 19x19矩阵
#         ground_truth_file: 真实数据文件路径
#         output_dir: 输出目录
        
#     Returns:
#         可视化结果
#     """
#     # 创建可视化器
#     visualizer = SParamVisualizer(model_manager, matrix)
    
#     # 执行完整可视化
#     results = visualizer.visualize_all(ground_truth_file, output_dir)
    
#     return results


# 快速使用示例
# if __name__ == "__main__":
#     # 示例用法
#     print("S参数可视化模块示例")
    
    # 这里需要你提供模型管理器和矩阵
    # 示例：
    # from model import CnnNet
    # from train import ModelManager
    # import torch
    
    # # 加载模型
    # model = CnnNet()
    # model.load_state_dict(torch.load('best_model.pth'))
    # model_manager = ModelManager(model)
    
    # # 定义矩阵（示例：随机矩阵）
    # test_matrix = np.random.randint(0, 2, (19, 19)).astype(np.float32)
    
    # # 真实数据文件
    # ground_truth_file = 'ground_truth_data.txt'
    
    # # 运行可视化
    # results = visualize_s_param_performance(
    #     model_manager, test_matrix, ground_truth_file, 
    #     output_dir='s_param_evaluation_results'
    # )
    
    # print("请参考模块文档使用visualize_s_param_performance函数")