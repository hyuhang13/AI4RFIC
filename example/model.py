# models/inductor_net.py
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
        hidden_dims = hidden_dims or MODEL_CONFIG['hidden_dims']
        dropout_rates = dropout_rates or MODEL_CONFIG['dropout_rates']
        use_batchnorm = use_batchnorm or MODEL_CONFIG['use_batchnorm']
        
        layers = []
        prev_dim = input_size
        
        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # BatchNorm层
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            # 激活函数
            layers.append(nn.ReLU())
            
            # Dropout层（使用对应的dropout率）
            dropout_rate = dropout_rates[i] if i < len(dropout_rates) else dropout_rates[-1]
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

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
