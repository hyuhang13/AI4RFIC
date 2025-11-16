# quick_test_fixed.py
import torch
import numpy as np
import pandas as pd
from data_preprocessor import DataPreprocessor
from model import InductorNet
from config import DATA_CONFIG

def quick_test_fixed():
    """修复后的快速测试 - 确保输入正确归一化"""
    print("=== 神经网络快速测试 (修复版) ===")
    
    # 1. 首先加载数据预处理器来获取归一化参数
    print("加载数据预处理器...")
    try:
        preprocessor = DataPreprocessor()
        X, y = preprocessor.load_and_preprocess_data("Diff_SQ_XFAB_MJ_All_copy.csv")
        print("✅ 数据预处理器加载成功")
        
        # 显示数据范围信息
        print("\n数据范围信息:")
        X_original = preprocessor.X_scaler.inverse_transform(X[:1])  # 反归一化一个样本来看原始范围
        y_original = preprocessor.inverse_transform_y(y[:1])
        print("输入特征原始范围示例:")
        # for i, feature in enumerate(DATA_CONFIG['input_features']):
        #     print(f"  {feature}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}] (标准化后)")
        #     print(f"            [{X_original[0, i]:.2f}] (原始值示例)")
        print("输出目标原始范围示例:")
        for i, output_name in enumerate(DATA_CONFIG['output_targets']):
            print(f"  {output_name}: [{y[:, i].min():.2f}, {y[:, i].max():.2f}] (标准化后)")
            print(f"            [{y_original[0, i]:.2f}] (原始值示例)")
    except Exception as e:
        print(f"❌ 加载数据预处理器失败: {e}")
        return
    
    # 2. 创建或加载模型
    try:
        # 尝试加载训练好的模型
        checkpoint = torch.load("checkpoints/inductor_checkpoints/checkpoint_epoch_499.pth")
        model = InductorNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 已加载训练好的模型")
    except Exception as e:
        # 使用随机初始化模型
        print(f"######: {e}")
        model = InductorNet()
        print("⚠️ 使用随机初始化模型（未训练）")
    
    model.eval()
    
    # 3. 创建测试输入（使用原始尺度参数）
    test_inputs_original = [
        # [Line_Width, Turns, Line_space, Y_Dimension, X_Dimension, freq] - 原始尺度
        [8.0, 2.0, 150.0, 150.0, 5.0],   # 典型参数
        [5.0, 2.0, 100.0, 100.0, 2.0],   # 小电感
        [12.0, 3.0,200.0, 200.0, 10.0], # 大电感
    ]
    
    print(f"\n测试输入参数 (原始尺度):")
    for i, input_vec in enumerate(test_inputs_original):
        print(f"样本 {i+1}: {input_vec}")
    
    # 4. 进行预测
    print("\n" + "="*50)
    print("预测结果")
    print("="*50)
    
    for i, input_original in enumerate(test_inputs_original):
        print(f"\n--- 样本 {i+1} ---")
        
        # 重要步骤：将原始输入归一化
        input_normalized = preprocessor.X_scaler.transform([input_original])
        
        print("输入参数 (原始尺度):")
        for j, feature in enumerate(DATA_CONFIG['input_features']):
            print(f"  {feature}: {input_original[j]}")
        
        print("输入参数 (归一化后):")
        for j, feature in enumerate(DATA_CONFIG['input_features']):
            print(f"  {feature}: {input_normalized[0, j]:.6f}")
        
        # 转换为张量并进行预测
        input_tensor = torch.FloatTensor(input_normalized)
        
        with torch.no_grad():
            output_normalized = model(input_tensor)
        
        print("神经网络输出 (归一化尺度):")
        outputs = ['Ldiff', 'Qdiff', 'Leff', 'Q', 'Reff']
        for j, output_name in enumerate(outputs):
            print(f"  {output_name}: {output_normalized[0, j]:.6f}")
        
        # 将输出反归一化到原始尺度
        try:
            output_original = preprocessor.inverse_transform_y(
                output_normalized.numpy()
            )
            
            print("神经网络输出 (原始尺度):")
            for j, output_name in enumerate(outputs):
                print(f"  {output_name}: {output_original[0, j]:.6e}")
                
        except Exception as e:
            print(f"反归一化输出时出错: {e}")
    
    # 5. 验证归一化-反归一化过程
    print("\n" + "="*50)
    print("归一化-反归一化验证")
    print("="*50)
    
    # 测试一个已知样本
    sample_idx = 900
    original_sample = X[sample_idx:sample_idx+1]
    original_target = y[sample_idx:sample_idx+1]
    
    # 反归一化查看原始值
    input_original_verify = preprocessor.inverse_transform_y(original_sample)
    target_original_verify = preprocessor.inverse_transform_y(original_target)
    
    print("验证样本 (索引 0):")
    print("输入原始值:")
    for j, feature in enumerate(DATA_CONFIG['input_features']):
        print(f"  {feature}: {input_original_verify[0, j]:.6f}")
    
    print("目标原始值:")
    for j, output_name in enumerate(DATA_CONFIG['output_targets']):
        print(f"  {output_name}: {target_original_verify[0, j]:.6e}")
    
    # 重新归一化并预测
    input_renormalized = preprocessor.X_scaler.transform(input_original_verify)
    input_tensor_verify = torch.FloatTensor(input_renormalized)
    
    with torch.no_grad():
        prediction_normalized = model(input_tensor_verify)
        prediction_original = preprocessor.inverse_transform_y(
            prediction_normalized.numpy()
        )
    
    print("预测结果验证:")
    for j, output_name in enumerate(DATA_CONFIG['output_targets']):
        true_val = target_original_verify[0, j]
        pred_val = prediction_original[0, j]
        error_pct = abs(pred_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-12 else float('inf')
        print(f"  {output_name}: 真实={true_val:.6e}, 预测={pred_val:.6e}, 误差={error_pct:.2f}%")

def test_normalization_process():
    """专门测试归一化过程"""
    print("\n" + "="*50)
    print("归一化过程测试")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.load_and_preprocess_data("Diff_SQ_XFAB_MJ_All_copy.csv")
    
    # 测试归一化-反归一化的往返一致性
    test_samples = [0, 10, 50]  # 测试几个样本
    
    print("测试归一化-反归一化往返一致性:")
    for idx in test_samples:
        original_X = X[idx:idx+1]
        original_y = y[idx:idx+1]
        
        # 反归一化
        X_denorm = preprocessor.X_scaler.inverse_transform(original_X)
        y_denorm = preprocessor.inverse_transform_y(original_y)
        
        # 重新归一化
        X_renorm = preprocessor.X_scaler.transform(X_denorm)
        y_renorm = preprocessor.inverse_transform_y(y_denorm)
        
        # 检查是否一致
        X_diff = np.abs(original_X - X_renorm).max()
        y_diff = np.abs(original_y - y_renorm).max()
        
        print(f"样本 {idx}: X误差={X_diff:.6e}, y误差={y_diff:.6e}")
        
        if X_diff > 1e-10 or y_diff > 1e-10:
            print(f"  ⚠️ 归一化过程可能存在精度问题!")
        else:
            print(f"  ✅ 归一化过程正常")

if __name__ == "__main__":
    quick_test_fixed()
    test_normalization_process()
    
    print("\n" + "="*50)
    print("测试完成总结")
    print("="*50)
    print("关键要点:")
    print("1. 输入数据必须使用X_scaler进行归一化")
    print("2. 输出数据必须使用y_scaler进行反归一化") 
    print("3. 归一化-反归一化过程应该保持一致性")
    print("4. 确保使用训练时相同的归一化参数")