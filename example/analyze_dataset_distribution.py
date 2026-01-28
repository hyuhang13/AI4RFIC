# analyze_dataset_distribution.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

def load_dataset_info(dataset_dir):
    """加载数据集信息"""
    dataset_dir = Path(dataset_dir)
    
    # 加载配置
    config_path = dataset_dir / 'dataset_config.pkl'
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # 加载元数据
    metadata_path = dataset_dir / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # 加载各数据集
    datasets = {}
    for split in ['train', 'val', 'test']:
        dataset_path = dataset_dir / f'{split}_dataset.pkl'
        with open(dataset_path, 'rb') as f:
            datasets[split] = pickle.load(f)
    
    return config, metadata, datasets

def analyze_frequency_distribution(datasets, bins=None, freq_min=8, freq_max=12):
    """
    分析频率分布
    
    Args:
        datasets: 包含train/val/test的字典
        bins: 频率分箱边界
        freq_min: 最小频率(GHz)
        freq_max: 最大频率(GHz)
    """
    if bins is None:
        bins = np.linspace(freq_min, freq_max, 13)  # 8-12GHz，每0.5GHz一个bin
    
    results = {}
    
    for split, data in datasets.items():
        frequencies = data['frequencies'].flatten()
        
        # 统计信息
        results[split] = {
            'frequencies': frequencies,
            'count': len(frequencies),
            'min': frequencies.min(),
            'max': frequencies.max(),
            'mean': frequencies.mean(),
            'std': frequencies.std(),
            'percentiles': {
                '25%': np.percentile(frequencies, 25),
                '50%': np.percentile(frequencies, 50),
                '75%': np.percentile(frequencies, 75)
            }
        }
    
    # 频率分箱统计
    freq_distribution = {}
    for split, data in datasets.items():
        frequencies = data['frequencies'].flatten()
        hist, bin_edges = np.histogram(frequencies, bins=bins)
        
        freq_distribution[split] = {
            'hist': hist,
            'bin_edges': bin_edges,
            'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2,
            'relative_freq': hist / hist.sum() * 100  # 百分比
        }
    
    return results, freq_distribution, bins

def print_distribution_summary(results, freq_distribution):
    """打印分布摘要"""
    print("=" * 80)
    print("数据集频率分布分析")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}数据集:")
        print(f"  样本总数: {results[split]['count']:,}")
        print(f"  频率范围: {results[split]['min']:.4f} - {results[split]['max']:.4f}")
        print(f"  平均值: {results[split]['mean']:.4f}")
        print(f"  标准差: {results[split]['std']:.4f}")
        print(f"  25%分位数: {results[split]['percentiles']['25%']:.4f}")
        print(f"  中位数: {results[split]['percentiles']['50%']:.4f}")
        print(f"  75%分位数: {results[split]['percentiles']['75%']:.4f}")
    
    # 打印频率分箱统计
    print("\n" + "-" * 80)
    print("频率分箱统计 (百分比):")
    print("-" * 80)
    
    # 获取所有分箱标签
    bin_labels = []
    for i in range(len(freq_distribution['train']['bin_centers'])):
        bin_start = freq_distribution['train']['bin_edges'][i]
        bin_end = freq_distribution['train']['bin_edges'][i+1]
        bin_labels.append(f"{bin_start:.1f}-{bin_end:.1f}")
    
    # 创建表格
    print(f"{'频率范围(GHz)':<15} {'训练集(%)':<12} {'验证集(%)':<12} {'测试集(%)':<12} {'总计(%)':<12}")
    print("-" * 60)
    
    for i, label in enumerate(bin_labels):
        train_pct = freq_distribution['train']['relative_freq'][i]
        val_pct = freq_distribution['val']['relative_freq'][i]
        test_pct = freq_distribution['test']['relative_freq'][i]
        total_pct = (train_pct + val_pct + test_pct) / 3
        
        print(f"{label:<15} {train_pct:>10.2f} {val_pct:>11.2f} {test_pct:>11.2f} {total_pct:>11.2f}")

def plot_frequency_distribution(results, freq_distribution, bins, save_dir=None):
    """绘制频率分布图"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 频率直方图对比
    ax1 = plt.subplot(2, 3, 1)
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    
    for split in ['train', 'val', 'test']:
        ax1.hist(results[split]['frequencies'], bins=50, alpha=0.6, 
                label=f'{split} ({results[split]["count"]:,})',
                color=colors[split], edgecolor='black')
    
    ax1.set_xlabel('freq', fontsize=12)
    ax1.set_ylabel('sample size', fontsize=12)
    ax1.set_title('frequency distribution histogram', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 分箱对比图
    ax2 = plt.subplot(2, 3, 2)
    x = np.arange(len(freq_distribution['train']['bin_centers']))
    width = 0.25
    
    for i, split in enumerate(['train', 'val', 'test']):
        ax2.bar(x + i*width, freq_distribution[split]['hist'], width,
               label=f'{split}', alpha=0.8, color=colors[split])
    
    ax2.set_xlabel('freq range (GHz)', fontsize=12)
    ax2.set_ylabel('sample size', fontsize=12)
    ax2.set_title('Frequency binning comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f"{bins[i]:.1f}-{bins[i+1]:.1f}" 
                        for i in range(len(bins)-1)], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 百分比对比图
    ax3 = plt.subplot(2, 3, 3)
    for split in ['train', 'val', 'test']:
        ax3.plot(freq_distribution[split]['bin_centers'], 
                freq_distribution[split]['relative_freq'],
                marker='o', linewidth=2, markersize=6, label=split)
    
    ax3.set_xlabel('freq center (GHz)', fontsize=12)
    ax3.set_ylabel('percent (%)', fontsize=12)
    ax3.set_title('freq percent comprasion', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱线图对比
    ax4 = plt.subplot(2, 3, 4)
    data_to_plot = [results[split]['frequencies'] for split in ['train', 'val', 'test']]
    box = ax4.boxplot(data_to_plot, labels=['train', 'val', 'test'],
                     patch_artist=True, showmeans=True)
    
    # 设置颜色
    colors_list = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box['boxes'], colors_list):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('freq', fontsize=12)
    ax4.set_title('Frequency distribution box plot', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. 累积分布函数
    ax5 = plt.subplot(2, 3, 5)
    for split in ['train', 'val', 'test']:
        sorted_freq = np.sort(results[split]['frequencies'])
        y = np.arange(1, len(sorted_freq) + 1) / len(sorted_freq)
        ax5.plot(sorted_freq, y * 100, linewidth=2, label=split)
    
    ax5.set_xlabel('freq', fontsize=12)
    ax5.set_ylabel('cumulative percentage (%)', fontsize=12)
    ax5.set_title(' (CDF)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 数据集大小饼图
    ax6 = plt.subplot(2, 3, 6)
    sizes = [results[split]['count'] for split in ['train', 'val', 'test']]
    labels = [f'train\n{sizes[0]:,}', f'val\n{sizes[1]:,}', f'test\n{sizes[2]:,}']
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax6.set_title('The proportion of dataset division', fontsize=14, fontweight='bold')
    
    plt.suptitle('Data Set Frequency Distribution Analysis Report', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / 'frequency_distribution_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存到: {save_path}")
    
    # plt.show()
    return fig

def calculate_distribution_imbalance(freq_distribution):
    """计算分布不平衡度"""
    imbalances = {}
    
    for split in ['train', 'val', 'test']:
        hist = freq_distribution[split]['hist']
        
        # 计算变异系数
        cv = hist.std() / hist.mean() if hist.mean() > 0 else 0
        
        # 计算最大最小值比
        max_min_ratio = hist.max() / hist.min() if hist.min() > 0 else float('inf')
        
        # 计算基尼系数
        sorted_hist = np.sort(hist)
        n = len(sorted_hist)
        cumsum = np.cumsum(sorted_hist)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        imbalances[split] = {
            'coefficient_of_variation': cv,
            'max_min_ratio': max_min_ratio,
            'gini_coefficient': gini,
            'is_balanced': cv < 0.5  # 变异系数小于0.5认为相对平衡
        }
    
    return imbalances

def analyze_sample_indices(datasets):
    """分析样本索引的分布（检查是否有数据泄露）"""
    print("\n" + "=" * 80)
    print("样本索引分析 (检查数据泄露)")
    print("=" * 80)
    
    # 检查是否有重复的索引
    all_indices = []
    for split, data in datasets.items():
        if 'indices' in data and data['indices'] is not None:
            indices = data['indices'].flatten()
            print(f"{split.upper()}数据集索引范围: {indices.min():.0f} - {indices.max():.0f}")
            all_indices.append(set(indices))
        else:
            print(f"{split.upper()}数据集: 无索引信息")
            all_indices.append(set())
    
    # 检查交集
    train_set, val_set, test_set = all_indices
    
    if train_set and val_set:
        train_val_overlap = train_set.intersection(val_set)
        print(f"训练集-验证集重叠样本数: {len(train_val_overlap)}")
    
    if train_set and test_set:
        train_test_overlap = train_set.intersection(test_set)
        print(f"训练集-测试集重叠样本数: {len(train_test_overlap)}")
    
    if val_set and test_set:
        val_test_overlap = val_set.intersection(test_set)
        print(f"验证集-测试集重叠样本数: {len(val_test_overlap)}")

def analyze_s_parameters(datasets):
    """分析S参数的统计特征"""
    print("\n" + "=" * 80)
    print("S参数统计分析")
    print("=" * 80)
    
    s_param_names = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag',
                    'S12_real', 'S12_imag', 'S22_real', 'S22_imag']
    
    for split, data in datasets.items():
        print(f"\n{split.upper()}数据集 S参数统计:")
        s_params = data['s_params']
        
        for i, name in enumerate(s_param_names):
            values = s_params[:, i]
            print(f"  {name}: 均值={values.mean():.4f}, 标准差={values.std():.4f}, "
                  f"范围=[{values.min():.4f}, {values.max():.4f}]")

def analyze_dataset_quality(dataset_dir, output_dir=None, freq_min=8, freq_max=12):
    """
    完整的数据集质量分析
    
    Args:
        dataset_dir: 数据集目录
        output_dir: 输出目录（用于保存图表）
        freq_min: 最小频率(GHz)
        freq_max: 最大频率(GHz)
    """
    print("开始分析数据集分布...")
    
    # 1. 加载数据
    config, metadata, datasets = load_dataset_info(dataset_dir)
    print(f"数据集目录: {dataset_dir}")
    print(f"创建时间: {metadata.get('created_time', '未知')}")
    print(f"总样本数: {metadata['dataset_sizes']['train'] + metadata['dataset_sizes']['val'] + metadata['dataset_sizes']['test']}")
    
    # 2. 分析频率分布
    results, freq_distribution, bins = analyze_frequency_distribution(
        datasets, freq_min=freq_min, freq_max=freq_max
    )
    
    # 3. 打印摘要
    print_distribution_summary(results, freq_distribution)
    
    # 4. 计算不平衡度
    imbalances = calculate_distribution_imbalance(freq_distribution)
    print("\n" + "=" * 80)
    print("分布不平衡度分析")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}数据集:")
        print(f"  变异系数: {imbalances[split]['coefficient_of_variation']:.3f}")
        print(f"  最大最小比: {imbalances[split]['max_min_ratio']:.2f}")
        print(f"  基尼系数: {imbalances[split]['gini_coefficient']:.3f}")
        print(f"  是否平衡: {'是' if imbalances[split]['is_balanced'] else '否'}")
    
    # 5. 检查数据泄露
    analyze_sample_indices(datasets)
    
    # 6. 分析S参数
    analyze_s_parameters(datasets)
    
    # 7. 绘制图表
    fig = plot_frequency_distribution(results, freq_distribution, bins, output_dir)
    
    # 8. 给出建议
    print("\n" + "=" * 80)
    print("分析结论与建议")
    print("=" * 80)
    
    # 检查训练集、验证集、测试集的频率分布是否相似
    train_freq_mean = results['train']['mean']
    val_freq_mean = results['val']['mean']
    test_freq_mean = results['test']['mean']
    
    mean_diff = max(abs(train_freq_mean - val_freq_mean),
                   abs(train_freq_mean - test_freq_mean),
                   abs(val_freq_mean - test_freq_mean))
    
    if mean_diff > 0.1:  # 如果均值差异超过0.1GHz
        print("⚠️  警告: 不同数据集的频率均值差异较大，可能导致泛化性能下降")
        print(f"   训练集均值: {train_freq_mean:.3f} GHz")
        print(f"   验证集均值: {val_freq_mean:.3f} GHz")
        print(f"   测试集均值: {test_freq_mean:.3f} GHz")
        print("   建议: 使用分层抽样或交叉验证来确保分布一致")
    else:
        print("✅  良好: 不同数据集的频率分布基本一致")
    
    # 检查数据量
    train_size = results['train']['count']
    val_size = results['val']['count']
    test_size = results['test']['count']
    
    total_size = train_size + val_size + test_size
    train_ratio = train_size / total_size
    val_ratio = val_size / total_size
    test_ratio = test_size / total_size
    
    print(f"\n📊 数据集划分比例:")
    print(f"   训练集: {train_ratio:.1%} ({train_size:,} 样本)")
    print(f"   验证集: {val_ratio:.1%} ({val_size:,} 样本)")
    print(f"   测试集: {test_ratio:.1%} ({test_size:,} 样本)")
    
    if train_ratio < 0.6:
        print("⚠️  警告: 训练集比例较低，可能导致模型欠拟合")
    if val_ratio < 0.1:
        print("⚠️  警告: 验证集比例较低，可能导致超参数调优不准确")
    if test_ratio < 0.1:
        print("⚠️  警告: 测试集比例较低，可能导致评估结果不稳定")
    
    # 检查频率覆盖范围
    all_frequencies = np.concatenate([results[split]['frequencies'] for split in ['train', 'val', 'test']])
    freq_coverage = (all_frequencies.max() - all_frequencies.min())
    
    if freq_coverage < (freq_max - freq_min) * 0.8:
        print(f"\n⚠️  警告: 频率覆盖范围不足")
        print(f"   实际覆盖: {all_frequencies.min():.1f}-{all_frequencies.max():.1f} GHz")
        print(f"   期望覆盖: {freq_min}-{freq_max} GHz")
    
    return {
        'results': results,
        'freq_distribution': freq_distribution,
        'imbalances': imbalances,
        'figure': fig
    }

def save_analysis_report(analysis_results, output_path):
    """保存分析报告到文本文件"""
    results = analysis_results['results']
    imbalances = analysis_results['imbalances']
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("数据集分布分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for split in ['train', 'val', 'test']:
        report_lines.append(f"\n{split.upper()}数据集:")
        report_lines.append(f"  样本数: {results[split]['count']:,}")
        report_lines.append(f"  频率范围: {results[split]['min']:.4f} - {results[split]['max']:.4f} GHz")
        report_lines.append(f"  均值: {results[split]['mean']:.4f} GHz")
        report_lines.append(f"  标准差: {results[split]['std']:.4f} GHz")
        report_lines.append(f"  变异系数: {imbalances[split]['coefficient_of_variation']:.3f}")
        report_lines.append(f"  是否平衡: {'是' if imbalances[split]['is_balanced'] else '否'}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"分析报告已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    DATASET_DIR = "/root/aicp-data/saved_datasets/v1"  # 修改为你的数据集路径
    OUTPUT_DIR = "analysis_results"  # 分析结果保存目录
    
    # 扩展用户目录
    DATASET_DIR = os.path.expanduser(DATASET_DIR)
    
    # 运行分析
    try:
        analysis_results = analyze_dataset_quality(
            dataset_dir=DATASET_DIR,
            output_dir=OUTPUT_DIR,
            freq_min=0,  # 根据你的数据调整
            freq_max=1   # 根据你的数据调整
        )
        
        # 保存详细报告
        save_analysis_report(analysis_results, os.path.join(OUTPUT_DIR, "distribution_analysis_report.txt"))
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()