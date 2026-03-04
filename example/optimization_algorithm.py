# optimization/genetic_algorithm.py
import os
import numpy as np
import random
from config import GA_CONFIG, DATA_CONFIG
import torch
from typing import List, Tuple, Dict, Optional, Callable
from config import DEVICE,GA_CONFIG
from tqdm import tqdm
import matplotlib.pyplot as plt
from s_param_visualizer import SParamVisualizer
from collections import deque
class GeneticAlgorithm:
    """
    遗传算法优化器：优化19×19二进制矩阵结构以获得目标S参数
    """
    def __init__(self, model_manager, matrix_shape = (19,19),
                 target_real_csv=None, target_imag_csv=None,
                 freq_sweep_bool = 1,device = None):
        self.model_manager = model_manager
        self.height, self.width = matrix_shape
        self.total_pixels = self.height * self.width
        self.density = 0.15 #random.uniform(0.3, 0.7) #暂未使用该超参数
        self.device = DEVICE
        self.target_real_csv = target_real_csv
        self.target_imag_csv = target_imag_csv
        # S参数名称
        self.s_param_names = ['S11', 'S21', 'S12', 'S22']
        self.s_component_names = ['real', 'imag', 'mag', 'phase']
        # 遗传算法参数
        self.target_freq = None
        self.population_size = GA_CONFIG['population_size']
        self.generations = GA_CONFIG['generations']
        self.mutation_rate = GA_CONFIG['mutation_rate']  # 变异率
        self.crossover_rate = GA_CONFIG['crossover_rate']  # 交叉率
        self.elite_size = GA_CONFIG['elite_size']  # 精英个体数量
        self.tournament_size = GA_CONFIG['tournament_size']
        self.w1 = GA_CONFIG['fitness_weight']['w1']
        self.w2 = GA_CONFIG['fitness_weight']['w2']
        self.w3 = GA_CONFIG['fitness_weight']['w3']
        self.freq_sweep_bool = freq_sweep_bool
        print(f"矩阵遗传算法初始化完成")
        print(f"矩阵形状: {matrix_shape}")
        print(f"总像素数: {self.total_pixels}")
        print(f"使用设备: {self.device}")

    def apply_symmetry(self, matrix: np.ndarray) -> np.ndarray:
        pass  
    def ensure_double_port(self,matrix):
        matrix[9][0] = 1
        matrix[9][18] = 1
        return matrix
    # def create_random_matrix(self, method='random', density=0.5) -> np.ndarray:
    #     """
    #     创建随机二进制矩阵
        
    #     Args:
    #         method: 创建方法 ('random', 'sparse', 'dense', 'pattern')
    #         density: 1的密度 (仅用于random方法)
        
    #     Returns:
    #         19×19的二进制矩阵
    #     """
    #     if method == 'random':
    #         # 完全随机
    #         # matrix = np.random.rand(self.height, self.width)
    #         # matrix = (matrix < density).astype(np.float32)
    #         matrix = np.random.randint(
    #             low=0,          
    #             high=2,         
    #             size=(self.height, self.width),  
    #             dtype=np.int32
    #         )
    #     elif method == 'sparse':
    #         pass
    #     if not self.check_connectivity(matrix):
    #         # 如果生成的矩阵不连通，递归重新生成
    #         return self.create_random_matrix(method, density)
    #     matrix = self.ensure_double_port(matrix)
        
    #     return np.ascontiguousarray(matrix)  # 确保返回连续数组
    def create_random_matrix(self, method='random', density=0.15) -> np.ndarray:
        """
        创建随机二进制矩阵（自带 4连通骨架保底，完美控制密度，防止递归爆栈）
        """
        # 放弃危险的递归，改用 for 循环。最多尝试 100 次
        max_attempts = 100
        
        for attempt in range(max_attempts):
            if method == 'random':
                # 1. 【修复密度问题】
                # 使用 np.random.rand 生成 0~1 的浮点数，然后与 density 比较
                # 这样就能完美控制金属生成的初始概率（比如 density=0.15 就是 15% 面积是金属）
                matrix = (np.random.rand(self.height, self.width) < density).astype(np.int32)
                
                # 2. 【核心保底】：强制铺设一条严格 4 连通的高速公路
                current_r = 9
                for c in range(self.width):
                    matrix[current_r][c] = 1
                    # 随机上下游走（增加结构的随机性，避免全是一条直线）
                    if random.random() < 0.3 and current_r > 0:
                        current_r -= 1
                    elif random.random() < 0.3 and current_r < self.height - 1:
                        current_r += 1
                    # 确保垂直方向上也连通（满足4连通规则的拐角）
                    matrix[current_r][c] = 1 
                    
                # 将尾端强制连接到右侧的标准输出端口 [9][18]
                if current_r != 9:
                    step = 1 if current_r < 9 else -1
                    for r in range(current_r, 9, step):
                        matrix[r][18] = 1
                        
            elif method == 'sparse':
                matrix = np.zeros((self.height, self.width), dtype=np.int32)
            
            # 3. 【修复顺序 Bug】：必须在检查连通性前，给左右端口加上金属！
            matrix = self.ensure_double_port(matrix)
            
            # 4. 检查连通性。由于有了上面的“铺路”操作，这一步几乎 100% 会瞬间返回 True
            if self.check_connectivity(matrix):
                return np.ascontiguousarray(matrix)
                
        # 5. 终极防崩溃保底
        print("警告: 随机生成连通矩阵异常，返回保底直连结构")
        matrix = np.zeros((self.height, self.width), dtype=np.int32)
        matrix[9, :] = 1
        return np.ascontiguousarray(matrix)
    def create_individual(self):
        """创建一个随机个体"""
        individual = self.create_random_matrix('random', self.density)
        return individual
    
    def create_population(self, population_size: int = None) -> List[np.ndarray]:
        """创建初始种群"""
        population_size = self.population_size
        population = []
        print(f"创建初始种群 (大小: {population_size})...")
        
        for i in tqdm(range(population_size), desc="创建种群"):
            individual = self.create_individual()
            population.append(individual)
            
        return population
    
    def matrix_to_tensor(self, matrix: np.ndarray) -> torch.Tensor:
        """将numpy矩阵转换为模型输入张量"""
        # 添加批次维度和通道维度
        tensor = torch.FloatTensor(matrix).unsqueeze(0).unsqueeze(0)  # (1, 1, 19, 19)
        return tensor.to(self.device)
    
    def predict_s_params(self, matrix: np.ndarray, frequency: float) -> Dict[str, float]:
        """预测矩阵在指定频率下的S参数"""
        # 归一化频率 (0-30GHz -> 0-1)
        # print(frequency)
        frequency_norm = (frequency - 100000000) / 2.99e10
        # print(frequency_norm)
        matrix = self.ensure_contiguous(matrix)
        # 使用模型管理器进行预测
        s_params = self.model_manager.predict_single_sample(matrix, frequency_norm)
        # print(s_params)
        return s_params
    
    def calculate_magnitude_db(self, real: float, imag: float) -> float:
        """计算S参数的幅度(dB)"""
        magnitude = np.sqrt(real**2 + imag**2)
        if magnitude < 1e-12:  # 避免log(0)
            return -100.0  # 非常小的dB值
        return 20 * np.log10(magnitude)
    
    def calculate_phase_degrees(self, real: float, imag: float) -> float:
        """计算S参数的相位(度)"""
        return np.degrees(np.arctan2(imag, real))
    
    # def predict_performance(self, individual):
    #     """使用神经网络预测个体性能"""
    #     self.model.eval()
    #     with torch.no_grad():
    #         individual_normalized = self.X_scaler.transform([individual])
    #         individual_tensor = torch.FloatTensor(individual_normalized)
    #         device = next(self.model.parameters()).device
    #         individual_tensor = individual_tensor.to(device)
            
    #         prediction_normalized = self.model(individual_tensor)
        
    #         prediction_cpu = prediction_normalized.cpu()
    #         prediction = self.y_scaler.inverse_transform(prediction_cpu.numpy())
    #         return prediction[0]
    def print_model_first_10_params(self,model):
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
    def check_connectivity(self, matrix: np.ndarray) -> bool:
        """
        使用广度优先搜索(BFS)检查射频端口1 [9][0] 到 端口2 [9][18] 是否物理连通。
        【修正】：严格遵循共边导通的物理规则，采用4连通检查（仅上下左右）。
        """
        start_node = (9, 0)
        end_node = (9, 18)
        
        # 如果起点或终点连金属都没有，直接判定断路
        if matrix[start_node] != 1 or matrix[end_node] != 1:
            return False
            
        visited = set()
        from collections import deque
        queue = deque([start_node])
        visited.add(start_node)
        
        # 【关键修改】：严格4连通，去除了所有对角线方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                      
        while queue:
            current_r, current_c = queue.popleft()
            
            # 如果到达了目标端口，说明连通
            if (current_r, current_c) == end_node:
                return True
                
            for dr, dc in directions:
                next_r, next_c = current_r + dr, current_c + dc
                
                # 检查边界
                if 0 <= next_r < self.height and 0 <= next_c < self.width:
                    # 如果下一个像素是金属(1) 且 尚未被访问过
                    if matrix[next_r, next_c] == 1 and (next_r, next_c) not in visited:
                        visited.add((next_r, next_c))
                        queue.append((next_r, next_c))
                        
        return False
    def fitness_function(self, matrix: np.ndarray, frequency, 
                        target_params: Dict) -> Tuple[float, Dict]:
        """
        计算适应度，支持单频点和多频点。
        多频点时，frequency应为列表，target_params中应包含'gamma_opt_list'（长度相同）。
        """
        # 预测S参数
        # 判断是否为多频点模式
        multi_freq_mode = (isinstance(frequency, (list, tuple, np.ndarray)) and 
                        'gamma_opt_list' in target_params and
                        len(frequency) == len(target_params['gamma_opt_list']))
        if multi_freq_mode and self.freq_sweep_bool:
            freqs = frequency
            gamma_opt_list = target_params['gamma_opt_list']
            
            total_cost = 0.0
            freq_predictions = []  # 保存每个频点的预测结果
            
            for f, gamma_opt in zip(freqs, gamma_opt_list):
                # 预测该频率下的S参数
                s_params = self.predict_s_params(matrix, f)
                s21 = s_params['S21_real'] + 1j * s_params['S21_imag']
                s22 = s_params['S22_real'] + 1j * s_params['S22_imag']
                IL_passive = 1 - abs(s21)
                cost = self.w1 * abs(s22 - gamma_opt) + self.w2 * IL_passive  # w3 暂为0
                total_cost += cost
                freq_predictions.append({
                    'freq': f,
                    's_params': s_params,
                    'cost': cost
                })
            
            avg_cost = total_cost / len(freqs)
            fitness = -avg_cost
            
            # 端口惩罚（输入输出端口必须为金属）
            if matrix[9][0] != 1 or matrix[9][18] != 1:
                fitness -= 50
            # 2. 【新增】物理连通性惩罚
            if not self.check_connectivity(matrix):
                fitness -= 1000  # 给予极大的惩罚，淘汰所有断路的矩阵
            prediction_info = {
                'matrix_density': np.mean(matrix),
                'multi_freq_predictions': freq_predictions,  # 包含所有频点的详细信息
                'avg_cost': avg_cost
            }
            return fitness, prediction_info
        else:
            # ---------- 原有单频点代码（保持不变） ----------
            s_params = self.predict_s_params(matrix, frequency)
            # 计算代价值（使用target_params中的gamma_opt）
            gamma_opt = target_params.get('gamma_opt', {}).get('origin_value', None)
            if gamma_opt is None:
                # 兼容旧调用方式
                gamma_opt = target_params.get('gamma_opt_list', [0])[0]
            s21_predict = s_params['S21_real'] + 1j * s_params['S21_imag']
            s22_predict = s_params['S22_real'] + 1j * s_params['S22_imag']
            IL_passive = 1 - abs(s21_predict)
            individual_cost = self.w1 * abs(s22_predict - gamma_opt) + self.w2 * IL_passive + self.w3 * 0
            individual_fitness = -individual_cost
            freq_predictions.append({
                    'freq': frequency,
                    's_params': s_params,
                    'cost': individual_cost
                })
            if matrix[9][0] != 1 and matrix[9][18] != 1:
                individual_fitness -= 50
            # 2. 【新增】物理连通性惩罚
            if not self.check_connectivity(matrix):
                fitness -= 1000  # 给予极大的惩罚，淘汰所有断路的矩阵
            prediction_info = {
                'matrix_density': np.mean(matrix),
                'multi_freq_predictions': freq_predictions,  # 包含所有频点的详细信息
                'avg_cost': individual_cost
            }
            return individual_fitness, prediction_info
    def ensure_contiguous(self, matrix):
        """确保矩阵是连续数组"""
        if isinstance(matrix, np.ndarray):
            return np.ascontiguousarray(matrix)
        elif isinstance(matrix, torch.Tensor):
            return matrix.contiguous()
        return matrix
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """交叉操作：创建子代矩阵"""
        parent1 = self.ensure_contiguous(parent1)
        parent2 = self.ensure_contiguous(parent2)
        child = np.zeros_like(parent1)
        
        # 随机选择交叉方式
        crossover_type = random.choice(['uniform', 'single_point', 'two_point', 'block'])
        
        if crossover_type == 'uniform':
            # 均匀交叉：每个像素随机从父代选择
            for i in range(self.height):
                for j in range(self.width):
                    if random.random() < 0.5:
                        child[i, j] = parent1[i, j]
                    else:
                        child[i, j] = parent2[i, j]
                        
        elif crossover_type == 'single_point':
            # 单点交叉：选择一行或一列作为交叉点
            if random.random() < 0.5:
                # 水平交叉
                crossover_point = random.randint(1, self.height-2)
                child[:crossover_point, :] = parent1[:crossover_point, :]
                child[crossover_point:, :] = parent2[crossover_point:, :]
            else:
                # 垂直交叉
                crossover_point = random.randint(1, self.width-2)
                child[:, :crossover_point] = parent1[:, :crossover_point]
                child[:, crossover_point:] = parent2[:, crossover_point:]
                
        elif crossover_type == 'two_point':
            # 两点交叉
            point1 = random.randint(1, self.total_pixels // 3)
            point2 = random.randint(2 * self.total_pixels // 3, self.total_pixels - 1)
            
            # 展平矩阵
            flat1 = parent1.flatten()
            flat2 = parent2.flatten()
            flat_child = np.zeros_like(flat1)
            
            flat_child[:point1] = flat1[:point1]
            flat_child[point1:point2] = flat2[point1:point2]
            flat_child[point2:] = flat1[point2:]
            
            child = flat_child.reshape(self.height, self.width)
            
        elif crossover_type == 'block':
            # 块交叉：交换一个子块
            block_size = random.randint(3, 7)
            h_start = random.randint(0, self.height - block_size)
            w_start = random.randint(0, self.width - block_size)
            
            child = parent1.copy()
            child[h_start:h_start+block_size, w_start:w_start+block_size] = \
                parent2[h_start:h_start+block_size, w_start:w_start+block_size]
        
        # 二值化
        child = (child > 0.5).astype(np.float32)
        child = self.ensure_double_port(child)
        return np.ascontiguousarray(child)  # 确保返回连续数组
    
    def mutate(self, matrix: np.ndarray, mutation_rate: float = None) -> np.ndarray:
        """变异操作"""
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
            
        mutated = matrix.copy()
        
        # 随机翻转像素
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < mutation_rate:
                    mutated[i, j] = 1.0 - mutated[i, j]  # 翻转
        
        # 额外的变异操作（以较低概率）
        if random.random() < 0.1:  # 30%概率进行额外变异
            mutation_type = random.choice(['invert', 'shift', 'rotate', 'noise'])
            
            if mutation_type == 'invert':
                # 整体反转
                mutated = 1.0 - mutated
                
            elif mutation_type == 'shift':
                # 平移
                shift_h = random.randint(-2, 2)
                shift_w = random.randint(-2, 2)
                temp = np.zeros_like(mutated)
                
                for i in range(self.height):
                    for j in range(self.width):
                        new_i, new_j = i + shift_h, j + shift_w
                        if 0 <= new_i < self.height and 0 <= new_j < self.width:
                            temp[new_i, new_j] = mutated[i, j]
                
                mutated = temp
                
            elif mutation_type == 'rotate':
                # 90度旋转
                k = random.choice([1, 2, 3])  # 1:90°, 2:180°, 3:270°
                mutated = np.rot90(mutated, k)
                
            elif mutation_type == 'noise':
                # 添加随机噪声块
                block_size = random.randint(2, 5)
                h_start = random.randint(0, self.height - block_size-1)
                w_start = random.randint(0, self.width - block_size-1)
                
                noise = np.random.rand(block_size, block_size) > 0.5
                mutated[h_start:h_start+block_size, w_start:w_start+block_size] = \
                    noise.astype(np.float32)
        matrix = self.ensure_double_port(mutated)
        return matrix
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[float], 
                           tournament_size: int = 256) -> int:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx
  
    def optimize(self, target_freq, target_params: Dict[str, Dict],
                population_size: int = None, generations: int = None,
                mutation_rate: float = None, elite_size: int = None,
                verbose: bool = True) -> Tuple[np.ndarray, Dict, float]:
        """
        遗传算法优化主函数
        
        Args:
            target_freq: 目标频率 (GHz)
            target_params: 目标参数配置
            population_size: 种群大小
            generations: 进化代数
            mutation_rate: 变异率
            elite_size: 精英个体数量
            verbose: 是否显示详细信息
        
        Returns:
            best_matrix: 最佳矩阵
            best_info: 最佳个体的信息
            best_fitness: 最佳适应度
        """
        # 设置参数
        
        population_size = population_size or self.population_size
        generations = generations or self.generations
        mutation_rate = mutation_rate or self.mutation_rate
        crossover_rate = self.crossover_rate
        elite_size = elite_size or self.elite_size
        tournament_size = self.tournament_size
        print(f"\n{'='*60}")
        print("启动遗传算法优化")
        print(f"{'='*60}")
        print(f"目标频率: {target_freq} GHz")
        print(f"目标参数: {target_params}")
        print(f"种群大小: {population_size}")
        print(f"进化代数: {generations}")
        print(f"变异率: {mutation_rate}")
        print(f"交叉率: {crossover_rate}")
        print(f"精英数量: {elite_size}")
        print(f"tournament_size: {tournament_size}")
        print(f"{'='*60}")
        self.target_freq = target_freq
        # 创建初始种群
        population = self.create_population(population_size)

        # 初始化记录
        best_matrix = None
        best_fitness = -float('inf')
        best_info = None
        
        fitness_history = []
        best_fitness_history = []
        
        # 进化循环
        for generation in tqdm(range(generations), desc="遗传演进", disable=not verbose):
            current_mutation_rate = self.mutation_rate * (1.0 - generation / generations)
            # 计算适应度
            fitness_scores = []
            predictions_info = []
            
            for matrix in population:
                fitness, info = self.fitness_function(matrix, target_freq, target_params)
                fitness_scores.append(fitness)
                predictions_info.append(info)
            
            # 更新最佳个体
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            print(f"当前代数: {generation}, 当前最佳适应度: {current_best_fitness:.4f}")
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_matrix = population[current_best_idx].copy()
                best_info = predictions_info[current_best_idx]

            # 记录历史
            avg_fitness = np.mean(fitness_scores)
            fitness_history.append(avg_fitness)
            best_fitness_history.append(best_fitness)
            
            # 选择精英
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            new_population = [population[i].copy() for i in elite_indices]
            
            # 生成新一代
            while len(new_population) < population_size:
                # 选择父代
                parent1_idx = self.tournament_selection(population, fitness_scores, tournament_size)
                parent2_idx = self.tournament_selection(population, fitness_scores, tournament_size)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2]).copy()
                
                # 变异
                child = self.mutate(child, mutation_rate=current_mutation_rate)
                
                new_population.append(child)
            
            population = new_population
            
            # 打印进度
            if verbose and (generation % 10 == 0 or generation == generations - 1):
                print(f"代 {generation:3d}: "
                      f"平均适应度 = {avg_fitness:.4f}, "
                      f"最佳适应度 = {best_fitness:.4f}")
                
                # if best_info is not None:
                #     # print(f"      S11 = {best_info['S11_mag_db']:.2f} dB, "
                #     #       f"S21 = {best_info['S21_mag_db']:.2f} dB, "
                #     #       f"密度 = {best_info['matrix_density']:.3f}")
                #     # print(f"s_params ={best_info['s_params']:.6f}")
                #     print(f"s_params =\n")
                #     # print(best_info['multi_freq_predictions'][]['s_params'])
        
        # 输出最终结果
        print(f"\n{'='*60}")
        print("遗传算法优化完成！")
        print(f"{'='*60}")
        print(f"最佳适应度: {best_fitness:.4f}")
        
        # if best_info is not None:
        #     print(f"\n最佳个体的S参数 (在 {target_freq} GHz):")
        #     print(f"  |S11| = {best_info['S11_mag_db']:.2f} dB")
        #     print(f"  |S21| = {best_info['S21_mag_db']:.2f} dB")
        #     print(f"  |S12| = {best_info['S12_mag_db']:.2f} dB")
        #     print(f"  |S22| = {best_info['S22_mag_db']:.2f} dB")
        #     print(f"  S11相位 = {best_info['S11_phase_deg']:.1f}°")
        #     print(f"  S21相位 = {best_info['S21_phase_deg']:.1f}°")
        #     print(f"  矩阵密度 = {best_info['matrix_density']:.3f}")
        
        return best_matrix, best_info, best_fitness, fitness_history, best_fitness_history
    def visualize_results(self, best_matrix: np.ndarray, 
                          best_info,
                         fitness_history: List[float], 
                         best_fitness_history: List[float],
                         port_for_impedance='s22',
                         output_dir:str = '.'):
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 最佳矩阵可视化
        axes[0, 0].imshow(best_matrix, cmap='binary', interpolation='nearest')
        axes[0, 0].set_title(f'optimized matrix (density: {np.mean(best_matrix):.3f})')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 进化过程
        axes[0, 1].plot(fitness_history, label='mean_fitness', alpha=0.7)
        axes[0, 1].plot(best_fitness_history, label='best_fitness', linewidth=2)
        axes[0, 1].set_xlabel('代数')
        axes[0, 1].set_ylabel('fitness')
        axes[0, 1].set_title('evolution_process')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 矩阵统计
        density = np.mean(best_matrix)
        axes[1, 0].bar(['0', '1'], [1-density, density], color=['white', 'black'])
        axes[1, 0].set_title('pixel_distribution')
        axes[1, 0].set_ylabel('protortion')
        axes[1, 0].set_ylim([0, 1])
        
        # 4. 适应度组件（如果可用）
        # 这里需要从best_info中获取，暂时留空或显示其他信息
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, f"best_fitness: {best_fitness_history[-1]:.4f}\n"
                        f"final_density: {density:.3f}\n"
                        f"final_evolution_num: {len(fitness_history)}", 
                        fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'genetic_optimization_results.png'), dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 单独保存最佳矩阵图像
        plt.figure(figsize=(8, 8))
        plt.imshow(best_matrix, cmap='binary', interpolation='nearest')
        plt.title(f'Optimized Binary Matrix (Frequency: {self.target_freq} GHz)')
        plt.colorbar(label='Value (0/1)')
        plt.savefig(os.path.join(output_dir, 'optimized_matrix.png'), dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        # ---------- 阻抗对比图（如果提供了CSV文件） ----------
        if self.target_real_csv and self.target_imag_csv:
            self._plot_impedance_comparison(best_info, port_for_impedance, output_dir=output_dir)

        # # ---------- S21幅度图 ----------
        # self._plot_s21_magnitude(best_info)
    def _plot_impedance_comparison(self, best_info, port='s22', output_dir=None):
        """

        绘制预测阻抗与目标阻抗的实部/虚部对比图
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import interpolate

        # 读取目标阻抗
        df_real = pd.read_csv(self.target_real_csv, delimiter=',', header=0)
        df_imag = pd.read_csv(self.target_imag_csv, delimiter=',', header=0)
        freq_target_hz = df_real.iloc[:, 0].values
        z_real_target = df_real.iloc[:, 1].values
        z_imag_target = df_imag.iloc[:, 1].values

        # 提取优化频点的预测S参数
        preds = best_info['multi_freq_predictions']
        freqs_pred_hz = np.array([p['freq'] for p in preds])
        # 按频率排序（确保插值正确）
        sort_idx = np.argsort(freqs_pred_hz)
        freqs_pred_hz = freqs_pred_hz[sort_idx]
        preds_sorted = [preds[i] for i in sort_idx]

        # 计算预测阻抗（从指定端口的反射系数）
        Z0 = 50.0
        z_real_pred = []
        z_imag_pred = []
        for p in preds_sorted:
            s = p['s_params']
            # 根据 port 参数选择 S 参数
            if port == 's11':
                s_re = s['S11_real']
                s_im = s['S11_imag']
            elif port == 's22':
                s_re = s['S22_real']
                s_im = s['S22_imag']
            else:
                raise ValueError("port must be 's11' or 's22'")
            S = s_re + 1j * s_im
            Z = Z0 * (1 + S) / (1 - S) if abs(1 - S) > 1e-12 else complex(1e6, 0)
            z_real_pred.append(Z.real)
            z_imag_pred.append(-Z.imag)

        z_real_pred = np.array(z_real_pred)
        z_imag_pred = np.array(z_imag_pred)

        # 创建画布
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 绘制目标阻抗（连续曲线）
        freq_target_ghz = freq_target_hz / 1e9
        ax1.plot(freq_target_ghz, z_real_target, 'b-', label='Target Real(Z)', linewidth=2)
        ax2.plot(freq_target_ghz, z_imag_target, 'b-', label='Target Imag(Z)', linewidth=2)

        # 绘制预测阻抗离散点
        freq_pred_ghz = freqs_pred_hz / 1e9
        ax1.scatter(freq_pred_ghz, z_real_pred, c='red', marker='o', label='Predicted (discrete)', zorder=5)
        ax2.scatter(freq_pred_ghz, z_imag_pred, c='red', marker='o', label='Predicted (discrete)', zorder=5)

        # 对预测点进行插值（如果点数足够）
        if len(freq_pred_ghz) >= 3:
            # 实部插值
            f_real = interpolate.interp1d(freq_pred_ghz, z_real_pred, kind='cubic',
                                          fill_value='extrapolate')
            # 虚部插值
            f_imag = interpolate.interp1d(freq_pred_ghz, z_imag_pred, kind='cubic',
                                          fill_value='extrapolate')
            # 生成密集频率用于绘制平滑曲线（在优化频点范围内）
            dense_freq = np.linspace(freq_pred_ghz.min(), freq_pred_ghz.max(), 200)
            ax1.plot(dense_freq, f_real(dense_freq), 'r--', label='Predicted (interpolated)', alpha=0.7)
            ax2.plot(dense_freq, f_imag(dense_freq), 'r--', label='Predicted (interpolated)', alpha=0.7)
        else:
            # 点数少则直接连线
            ax1.plot(freq_pred_ghz, z_real_pred, 'r--', label='Predicted (linear)', alpha=0.7)
            ax2.plot(freq_pred_ghz, z_imag_pred, 'r--', label='Predicted (linear)', alpha=0.7)

        # 设置标签和图例
        ax2.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Real(Z) (Ω)')
        ax2.set_ylabel('Imag(Z) (Ω)')
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax1.set_title(f'Impedance Comparison (using {port})')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'impedance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("阻抗对比图已保存至 impedance_comparison.png")
    def _plot_s21_magnitude(self, best_info, output_dir=None):
        """
        绘制 S21 幅度 (dB) 曲线
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import interpolate

        preds = best_info['multi_freq_predictions']
        freqs_hz = np.array([p['freq'] for p in preds])
        sort_idx = np.argsort(freqs_hz)
        freqs_hz = freqs_hz[sort_idx]
        preds_sorted = [preds[i] for i in sort_idx]

        s21_mag_db = []
        for p in preds_sorted:
            s = p['s_params']
            mag = np.sqrt(s['S21_real']**2 + s['S21_imag']**2)
            mag_db = 20 * np.log10(mag + 1e-12)
            s21_mag_db.append(mag_db)

        freqs_ghz = freqs_hz / 1e9
        plt.figure(figsize=(8, 5))
        # 绘制离散点
        plt.scatter(freqs_ghz, s21_mag_db, c='green', marker='s', label='Predicted (discrete)', zorder=5)
        # 绘制插值曲线
        if len(freqs_ghz) >= 3:
            f = interpolate.interp1d(freqs_ghz, s21_mag_db, kind='cubic', fill_value='extrapolate')
            dense_freq = np.linspace(freqs_ghz.min(), freqs_ghz.max(), 200)
            plt.plot(dense_freq, f(dense_freq), 'g-', label='S21 (interpolated)')
        else:
            plt.plot(freqs_ghz, s21_mag_db, 'g-', label='S21 (linear)')

        plt.xlabel('Frequency (GHz)')
        plt.ylabel('|S21| (dB)')
        plt.title('S21 Magnitude of Optimized Structure')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('s21_magnitude.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("S21幅度图已保存至 s21_magnitude.png")
