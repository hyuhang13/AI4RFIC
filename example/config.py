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
    'tournament_size': 256
}
# 训练配置
TRAIN_CONFIG = {
    'batch_size': 1024,
    'learning_rate': 0.001,
    'epochs': 800,
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
    'patience': 10,
    'factor': 0.95
}
# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')