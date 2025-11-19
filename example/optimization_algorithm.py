# optimization/genetic_algorithm.py
import numpy as np
import random
from config import GA_CONFIG, DATA_CONFIG
import torch
class GeneticAlgorithm:
    def __init__(self, model, X_scaler, y_scaler):
        self.model = model
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.input_features = DATA_CONFIG['input_features']
        self.output_targets = DATA_CONFIG['output_targets']
        
        # 定义参数范围
        self.param_ranges = {
            'Line_Width': (2, 10),
            'Turns': (1, 3),
            # 'Line_space': (3, 3),
            'Y_Dimension': (100, 200),
            'X_Dimension': (100, 200),
            'freq': (2, 20)
        }
    
    def create_individual(self):
        """创建一个随机个体"""
        individual = []
        for param in self.input_features:
            min_val, max_val = self.param_ranges[param]
            if param == 'Turns':
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
            individual_normalized = self.X_scaler.transform([individual])
            individual_tensor = torch.FloatTensor(individual_normalized)
            device = next(self.model.parameters()).device
            individual_tensor = individual_tensor.to(device)
            
            prediction_normalized = self.model(individual_tensor)
        
            prediction_cpu = prediction_normalized.cpu()
            prediction = self.y_scaler.inverse_transform(prediction_cpu.numpy())
            return prediction[0]
    
    def fitness_function(self, individual, target_freq, target_Leff, target_Q=None):
        """适应度函数"""
        individual_with_freq = individual.copy()
        individual_with_freq[self.input_features.index('freq')] = target_freq
        
        prediction = self.predict_performance(individual_with_freq)
        Leff_pred, Q_pred = prediction[2], prediction[3]
        
        Leff_error = abs(Leff_pred - target_Leff) / target_Leff
        
        if target_Q is not None:
            Q_error = abs(Q_pred - target_Q) / target_Q
            total_error = 0.7 * Leff_error + 0.3 * Q_error
        else:
            total_error = Leff_error
        
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
                 population_size=None, generations=None, 
                 mutation_rate=None, elite_size=None):
        """遗传算法优化主函数"""
        # 使用配置参数
        population_size = population_size or GA_CONFIG['population_size']
        generations = generations or GA_CONFIG['generations']
        mutation_rate = mutation_rate or GA_CONFIG['mutation_rate']
        elite_size = elite_size or GA_CONFIG['elite_size']
        tournament_size = GA_CONFIG['tournament_size']
        
        population = self.create_population(population_size)
        best_individual = None
        best_fitness = -float('inf')
        best_prediction = None
        
        for generation in range(generations):
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
            
            # 选择精英个体
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            new_population = [population[i] for i in elite_indices]
            
            # 生成新一代
            while len(new_population) < population_size:
                # 锦标赛选择
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent2_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                # 交叉和变异
                child = self.crossover(population[parent1_idx], population[parent2_idx])
                child = self.mutate(child, mutation_rate)
                new_population.append(child)
            
            population = new_population
            
            if generation % 20 == 0:
                Leff_pred = best_prediction[2]
                print(f'Generation {generation}: Best Fitness = {best_fitness:.4f}, Leff = {Leff_pred:.2e}')
        
        return best_individual, best_prediction, best_fitness