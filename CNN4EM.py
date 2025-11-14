import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import os
# ==================== 1. 数据预处理 ====================

class InductorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(file_path):
    """
    加载和预处理电感数据
    """
    # 读取Excel文件
    df = pd.read_csv(file_path)
    print("原始列名:", df.columns.tolist())
    print("数据形状:", df.shape)
    # 提取输入特征 (6个参数)
    #线宽，匝数，线间距，纵向直径，横向直径，频率点
    input_features = ['Line_Width', 'Turns', 'Line_space', 'Y_Dimension', 'X_Dimension', 'freq']
    # 注意：freq列名可能需要根据实际数据调整，如 'freq (GHz)' 等
    
    # 提取输出目标 (5个性能指标)
    #差分电感值，差分品质因数，有效电感，有效电阻
    output_targets = ['Ldiff', 'Qdiff', 'Leff', 'Q', 'Reff']
    # 数据清洗和转换
    print("数据清洗前:")
    for col in input_features + output_targets:
        print(f"{col}: 数据类型={df[col].dtype}, 示例={df[col].iloc[:3].tolist()}")
    # 专门处理频率列 - 移除"GHz"单位并转换为数值
    if df['freq'].dtype == 'object':
        # 移除"GHz"单位
        df['freq'] = df['freq'].str.replace('GHz', '', regex=False).str.strip()
        
        # 转换为数值
        df['freq'] = pd.to_numeric(df['freq'], errors='coerce')
        print(f"频率列转换后，缺失值数量: {df['freq'].isna().sum()}")
    
    # 处理其他列，确保它们都是数值类型
    for col in input_features + output_targets:
        if df[col].dtype == 'object':
            # 尝试转换为数值
            df[col] = pd.to_numeric(df[col], errors='coerce')
            na_count = df[col].isna().sum()
            if na_count > 0:
                print(f"列 {col} 转换后有 {na_count} 个缺失值")
                print("数据清洗后:")
    for col in input_features + output_targets:
        print(f"{col}: 数据类型={df[col].dtype}, 示例={df[col].iloc[:3].tolist()}")
    X = df[input_features].values
    y = df[output_targets].values
    
    # 数据标准化
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_normalized = X_scaler.fit_transform(X)
    y_normalized = y_scaler.fit_transform(y)
    
    return X_normalized, y_normalized, X_scaler, y_scaler, input_features, output_targets

# ==================== 2. 神经网络模型 ====================

class InductorNet(nn.Module):
    def __init__(self, input_size=6, output_size=5, hidden_layers=[128, 256, 128, 64]):
        super(InductorNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
def train_model_with_resume(model, train_loader, val_loader, epochs=500, learning_rate=0.001, checkpoint_path=None):
    """
    训练神经网络模型，支持从检查点恢复
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    start_epoch = 0
    train_losses = []
    val_losses = []
    
    # 如果提供了检查点路径，加载模型状态
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"从检查点恢复训练: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_loss']
        val_losses = checkpoint['val_loss']
        print(f"从 epoch {start_epoch} 继续训练")
    
    try:
        for epoch in range(start_epoch, epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
                
            # 定期保存检查点
            if epoch % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                }, f'checkpoint_epoch_{epoch}.pth')
    
    except KeyboardInterrupt:
        print("\n训练被用户中断!")
        print(f"训练在 epoch {epoch} 停止")
        
        # 保存中断时的模型状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            'val_loss': val_losses,
        }, 'interrupted_training.pth')
        print("模型状态已保存到 'interrupted_training.pth'")
    
    return train_losses, val_losses

# ==================== 3. 遗传算法优化 ====================

class GeneticAlgorithm:
    def __init__(self, model, X_scaler, y_scaler, input_features, output_targets):
        self.model = model
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.input_features = input_features
        self.output_targets = output_targets
        
        # 定义参数范围（基于你的数据规格）
        self.param_ranges = {
            'Line_Width': (2, 10),      # 2-10 μm stride 1
            'Turns': (1, 3),            # 1-3 (整数)stride 1
            'Line_space': (3, 3),       # 固定3μm
            'Y_Dimension': (100, 200),  # 100-200 μm    stride 20
            'X_Dimension': (100, 200),  # 100-200 μm    stride 20
            'freq': (2, 20)             # 2-20 GHz      stride 0.1
        }
    
    def create_individual(self):
        """创建一个随机个体"""
        individual = []
        for param in self.input_features:
            min_val, max_val = self.param_ranges[param]
            if param == 'Turns':
                # Turns是整数
                individual.append(random.randint(min_val, max_val))
            else:
                individual.append(random.uniform(min_val, max_val))
        return individual
    
    def create_population(self, population_size):
        """创建初始种群"""
        return [self.create_individual() for _ in range(population_size)]
    
    def predict_performance(self, individual):
        """使用神经网络预测个体性能"""
        self.model.eval()
        with torch.no_grad():
            # 标准化输入
            individual_normalized = self.X_scaler.transform([individual])
            individual_tensor = torch.FloatTensor(individual_normalized)
            
            # 预测
            prediction_normalized = self.model(individual_tensor)
            prediction = self.y_scaler.inverse_transform(prediction_normalized.numpy())
            
            return prediction[0]  # [Ldiff, Qdiff, Leff, Q, Reff]
    
    def fitness_function(self, individual, target_freq, target_Leff, target_Q=None):
        """
        适应度函数
        individual: [Line_Width, Turns, Line_space, Y_Dimension, X_Dimension, freq]
        """
        # 设置目标频率
        individual_with_freq = individual.copy()
        individual_with_freq[self.input_features.index('freq')] = target_freq#复制个体并替换概率
        
        # 预测性能
        prediction = self.predict_performance(individual_with_freq)
        Leff_pred, Q_pred = prediction[2], prediction[3]  # Leff和Q
        
        # 计算适应度（误差越小，适应度越高）
        Leff_error = abs(Leff_pred - target_Leff) / target_Leff
        
        if target_Q is not None:
            Q_error = abs(Q_pred - target_Q) / target_Q
            total_error = 0.7 * Leff_error + 0.3 * Q_error  # 加权误差
        else:
            total_error = Leff_error
        
        # 适应度是误差的倒数（误差越小，适应度越高）
        fitness = 1.0 / (1.0 + total_error)
        
        return fitness, prediction
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    def mutate(self, individual, mutation_rate=0.1):
        """变异操作"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                param_name = self.input_features[i]
                min_val, max_val = self.param_ranges[param_name]
                if param_name == 'Turns':
                    mutated[i] = random.randint(min_val, max_val)
                else:
                    mutated[i] = random.uniform(min_val, max_val)
        return mutated
    
    def optimize(self, target_freq, target_Leff, target_Q=None, 
                 population_size=50, generations=100, 
                 mutation_rate=0.1, elite_size=5):
        """
        遗传算法优化主函数
        """
        # 创建初始种群
        population = self.create_population(population_size)
        
        best_individual = None
        best_fitness = -float('inf')
        best_prediction = None
        
        for generation in range(generations):
            # 评估种群中每个个体的适应度
            fitness_scores = []
            predictions = []
            
            for individual in population:
                fitness, prediction = self.fitness_function(
                    individual, target_freq, target_Leff, target_Q
                )
                fitness_scores.append(fitness)
                predictions.append(prediction)
            
            # 选择最佳个体
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_individual = population[best_idx]
                best_prediction = predictions[best_idx]
            
            # 选择精英个体直接进入下一代
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            new_population = [population[i] for i in elite_indices]
            
            # 生成新一代
            while len(new_population) < population_size:
                # 锦标赛选择
                tournament_size = 3
                tournament_indices = random.sample(range(len(population)), tournament_size)
                print("tournment1indices:"+tournament_indices)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                print("tournment1fitness:"+tournament_fitness)
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                print("parent1:"+parent1_idx)
                tournament_indices = random.sample(range(len(population)), tournament_size)
                print("tournment2indices:"+tournament_indices)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                print("tournment2fitness:"+tournament_fitness)
                parent2_idx = tournament_indices[np.argmax(tournament_fitness)]
                print("parent2:"+parent2_idx)
                # 交叉和变异
                child = self.crossover(population[parent1_idx], population[parent2_idx])
                child = self.mutate(child, mutation_rate)
                new_population.append(child)
            
            population = new_population
            
            if generation % 20 == 0:
                Leff_pred = best_prediction[2]
                print(f'Generation {generation}: Best Fitness = {best_fitness:.4f}, Leff = {Leff_pred:.2e}')
        
        return best_individual, best_prediction, best_fitness

# ==================== 4. 主程序 ====================

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 1. 加载和预处理数据
    print("Loading and preprocessing data...")
    file_path = "Diff_SQ_XFAB_MJ_All_copy.csv"  # 替换为你的Excel文件路径
    X, y, X_scaler, y_scaler, input_features, output_targets = load_and_preprocess_data(file_path)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_dataset = InductorDataset(X_train, y_train)
    test_dataset = InductorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. 构建和训练神经网络
    print("Building and training neural network...")
    model = InductorNet(input_size=6, output_size=5)
    
    # 检查是否有中断的训练可以恢复
    checkpoint_path = None
    if os.path.exists('interrupted_training.pth'):
        response = input("发现中断的训练记录，是否恢复训练? (y/n): ")
        if response.lower() == 'y':
            checkpoint_path = 'interrupted_training.pth'
    
    # 训练模型
    train_losses, val_losses = train_model_with_resume(
        model, train_loader, test_loader, 
        epochs=1000, learning_rate=0.001,
        checkpoint_path=checkpoint_path
    )
    
    # 3. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'input_features': input_features,
        'output_targets': output_targets
    }, 'inductor_model.pth')
    
    print("Model training completed and saved!")
    
    # 4. 使用遗传算法进行逆向设计 
    print("\nStarting genetic algorithm optimization...")
    
    # 创建遗传算法优化器
    ga = GeneticAlgorithm(model, X_scaler, y_scaler, input_features, output_targets)
    
    # 设置设计目标：在10GHz时Leff=5nH
    target_freq = 4.0  # GHz
    target_Leff = 2e-9  # 5nH
    
    # 运行优化
    best_params, best_performance, best_fitness = ga.optimize(
        target_freq=target_freq,
        target_Leff=target_Leff,
        population_size=50,
        generations=100,
        mutation_rate=0.1
    )
    
    # 输出优化结果
    print("\n=== Optimization Results ===")
    print(f"Target: Leff = {target_Leff:.2e} at {target_freq} GHz")
    print(f"Best Fitness: {best_fitness:.4f}")
    print("\nBest Parameters:")
    for i, param_name in enumerate(input_features):
        print(f"  {param_name}: {best_params[i]}")
    
    print("\nPredicted Performance:")
    for i, target_name in enumerate(output_targets):
        print(f"  {target_name}: {best_performance[i]:.2e}")
    
    # 验证在目标频率下的性能
    verification_freq = target_freq
    verification_params = best_params.copy()
    verification_params[input_features.index('freq')] = verification_freq
    
    final_prediction = ga.predict_performance(verification_params)
    print(f"\nVerification at {verification_freq} GHz:")
    print(f"  Leff: {final_prediction[2]:.2e} (target: {target_Leff:.2e})")
    print(f"  Error: {abs(final_prediction[2] - target_Leff) / target_Leff * 100:.2f}%")

if __name__ == "__main__":
    main()