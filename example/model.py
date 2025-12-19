# models/inductor_net.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import MODEL_CONFIG, OPTIMIZER_CONFIG, SCHEDULER_CONFIG
class SingleOutputInductorNet(nn.Module):
    """针对单个输出的神经网络"""
    def __init__(self, input_size, output_size=1, hidden_dims=[128, 64, 32], 
                 dropout_rate=0.2, use_batchnorm=True):
        super().__init__()
        
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CnnNet(nn.Module):
    """
    融合频率信息的CNN模型
    使用双分支结构：图像分支 + 频率分支
    """
    def __init__(self):
        super().__init__()
        
        # ===== 图像分支（处理二进制矩阵）=====
        self.image_branch = nn.Sequential(
            # 第一卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 自适应池化到固定大小
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        # ===== 频率分支（处理频率信息）=====
        self.frequency_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # ===== 特征融合和回归 =====
        # 图像特征维度: 128 * 2 * 2 = 512
        # 频率特征维度: 128
        # 融合后总维度: 512 + 128 = 640
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 8)  # 输出8个S参数
        )
        
    def forward(self, matrix, frequency):
        # 图像特征提取
        image_features = self.image_branch(matrix)
        image_features = image_features.view(image_features.size(0), -1)  # 展平
        
        # 频率特征提取
        freq_features = self.frequency_branch(frequency)
        
        # 特征融合
        combined_features = torch.cat([image_features, freq_features], dim=1)
        
        # 回归预测
        output = self.fusion_layers(combined_features)
        
        return output

class ResidualNet(nn.Module):
    """
    使用注意力机制融合图像和频率信息的高级模型
    """
    
    def __init__(self):
        super(ResidualNet, self).__init__()
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 频率编码器
        self.freq_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制
        self.image_attention = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.Tanh(),
            nn.Linear(256, 256 * 4 * 4),
            nn.Sigmoid()
        )
        
        self.freq_attention = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Sigmoid()
        )
        
        # 融合和回归
        self.fusion = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 8)
        )
        
    def forward(self, matrix, frequency):
        # 编码图像
        image_features = self.image_encoder(matrix)
        image_features_flat = image_features.view(image_features.size(0), -1)
        
        # 编码频率
        freq_features = self.freq_encoder(frequency)
        
        # 应用注意力
        image_att = self.image_attention(image_features_flat)
        attended_image = image_features_flat * image_att
        
        freq_att = self.freq_attention(freq_features)
        attended_freq = freq_features * freq_att
        
        # 融合特征
        combined = torch.cat([attended_image, attended_freq], dim=1)
        
        # 回归
        output = self.fusion(combined)
        
        return output