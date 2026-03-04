# config.py
import torch
# 模型配置
MODEL_CONFIG = {
    'input_size': 5,
    'output_size': 5,
    'hidden_dims': [128, 256, 128, 64],
    'dropout_rates': [0.4,0.3,0.2,0.1],   #每层采用不同的池化率
    'use_batchnorm': True  # 使用BatchNorm
}

# 数据配置
DATA_CONFIG = {
    'input_features': ['Line_Width', 'Turns', 'Y_Dimension', 'X_Dimension', 'freq'],#'Line_space', 
    'output_targets': ['Ldiff', 'Qdiff', 'Leff', 'Q', 'Reff'],
    'test_size': 0.2,
    'random_state': 42
}

# 遗传算法配置
GA_CONFIG = {
            'population_size': 4096,
            'generations': 100,
            'mutation_rate': 0.1,
            'elite_size': 5,
            'tournament_size': 128,
            'crossover_rate' :0.8,
            'fitness_weight': {
                'w1': 1,
                'w2': 1,
                'w3': 0
            }
        }
# 优化目标配置  7	22000000000	-0.0445877	0.0125595	0.914061	-0.241433	0.914061	-0.241433	-0.04327	0.0114308
#-0.0445877	0.0125595
#0.914061	-0.241433
#0.914061	-0.241433
#-0.04327	0.0114308
# -0.0191442,0.053485,0.914802,-0.263989,0.914802,-0.263989,-0.0208617,0.0557113
TARGET_CONFIG = {
    'target_freq': [10e9,10e9,1e9,1],#freq_start, freq_end, freq_step = 1GHz,freq_sweep_bool
    'target_gamma_opt': 1,
    'target_s_params': {
        's11_real': -0.0191442,
        's11_imag': 0.053485,
        's21_real': 0.914802,
        's21_imag': -0.263989,
        's12_real': 0.914802,
        's12_imag': -0.263989,
        's22_real': -0.0208617,
        's22_imag': 0.0557113
    }
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 1024,
    'learning_rate': 0.001,
    'epochs': 400,
    'weight_decay': 0.001,
    'checkpoint_dir': 'checkpoints/inductor_checkpoints',
    'resume_training': True
}
# 优化器配置
OPTIMIZER_CONFIG = {
    'type': 'AdamW',  # 使用AdamW优化器
    'lr': 0.001,
    'weight_decay': 0.01
}

# 调度器配置
SCHEDULER_CONFIG = {
    'type': 'ReduceLROnPlateau',
    'patience': 5,
    'factor': 0.8
}
# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')