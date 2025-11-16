# models/inductor_net.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import MODEL_CONFIG, OPTIMIZER_CONFIG, SCHEDULER_CONFIG

class InductorNet(nn.Module):
    def __init__(self, input_size=None, output_size=None, hidden_dims=None, 
                 dropout_rates=None, use_batchnorm=None):
        super(InductorNet, self).__init__()
        
        # 使用配置参数或默认值
        input_size = input_size or MODEL_CONFIG['input_size']
        output_size = output_size or MODEL_CONFIG['output_size']
        # hidden_dims = hidden_dims or MODEL_CONFIG['hidden_dims']
        # dropout_rates = dropout_rates or MODEL_CONFIG['dropout_rates']
        use_batchnorm = use_batchnorm or MODEL_CONFIG['use_batchnorm']
        
        layers = []
        prev_dim = input_size
        init_method='kaiming'
        # 使用更合理的网络结构
        hidden_dims = hidden_dims or [256, 512, 256, 128, 64]
        dropout_rates = dropout_rates or [0.4, 0.4, 0.3, 0.2, 0.1]
        
        self.input_size = input_size
        self.output_size = output_size
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rates[0]),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rates[1]),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rates[2]),
        )
        
        # 为表现好的目标（Ldiff, Leff）设计的输出头
        self.good_targets_head = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rates[3]),
            
            nn.Linear(hidden_dims[3], hidden_dims[4]),
            nn.BatchNorm1d(hidden_dims[4]) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rates[4]),
            
            nn.Linear(hidden_dims[4], 2),  # Ldiff和Leff
        )
        
        # 为表现差的目标（Qdiff, Q, Reff）设计的专门输出头
        self.poor_targets_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 128),
            nn.BatchNorm1d(128) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            
            nn.Linear(64, 3),  # Qdiff, Q, Reff
        )
    
    def forward(self, x):
        # 共享特征提取
        shared_features = self.shared_layers(x)
        
        # 分别预测不同组的目标
        good_targets = self.good_targets_head(shared_features)
        poor_targets = self.poor_targets_head(shared_features)
        
        # 合并输出 [Ldiff, Qdiff, Leff, Q, Reff]
        # 注意：需要确保顺序正确
        output = torch.cat([
            good_targets[:, 0:1],  # Ldiff
            poor_targets[:, 0:1],  # Qdiff
            good_targets[:, 1:2],  # Leff
            poor_targets[:, 1:2],  # Q
            poor_targets[:, 2:3]   # Reff
        ], dim=1)
        
        return output


class ResidualInductorNet(nn.Module):
    """可选：带有残差连接的更复杂网络"""
    def __init__(self, input_size=None, output_size=None, hidden_dims=None):
        super(ResidualInductorNet, self).__init__()
        
        input_size = input_size or MODEL_CONFIG['input_size']
        output_size = output_size or MODEL_CONFIG['output_size']
        hidden_dims = hidden_dims or [256, 512, 256, 128]
        
        self.input_layer = nn.Linear(input_size, hidden_dims[0])
        self.bn_input = nn.BatchNorm1d(hidden_dims[0])
        
        # 残差块
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            res_block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
            )
            self.res_blocks.append(res_block)
            
            # 如果维度不匹配，添加投影层
            if hidden_dims[i] != hidden_dims[i+1]:
                self.res_blocks.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            else:
                self.res_blocks.append(nn.Identity())
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn_input(x)
        x = nn.ReLU()(x)
        
        for i in range(0, len(self.res_blocks), 2):
            res_block = self.res_blocks[i]
            shortcut = self.res_blocks[i+1]
            
            identity = shortcut(x)
            out = res_block(x)
            x = nn.ReLU()(out + identity)
            x = self.dropout(x)
        
        return self.output_layer(x)

# 为了向后兼容，保留原来的类名
