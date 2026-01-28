# models/inductor_net.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import MODEL_CONFIG, OPTIMIZER_CONFIG, SCHEDULER_CONFIG

class CnnNet(nn.Module):
    """
    融合频率信息的CNN模型
    使用双分支结构：图像分支 + 频率分支
    输入：19×19矩阵 + 频率（归一化到[0,1]）
    输出：8个S参数（2端口，每个复数实部+虚部）
    """
    def __init__(self, num_freq_bins=100):
        super().__init__()
        
        # # ===== 图像分支（处理二进制矩阵）=====
        # self.image_branch = nn.Sequential(
        #     # 第一卷积块
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
            
        #     # 第二卷积块
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
            
        #     # 第三卷积块
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
            
        #     # 自适应池化到固定大小
        #     nn.AdaptiveAvgPool2d((2, 2))
        # )
        
        # # ===== 频率分支（处理频率信息）=====
        # self.frequency_branch = nn.Sequential(
        #     nn.Linear(1, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True)
        # )
        # # ===== 特征融合和回归 =====
        # # 图像特征维度: 128 * 2 * 2 = 512
        # # 频率特征维度: 128
        # # 融合后总维度: 512 + 128 = 640
        
        # self.fusion_layers = nn.Sequential(
        #     nn.Linear(512 + 128, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
            
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
            
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
            
        #     nn.Linear(128, 8)  # 输出8个S参数
        # )
        # ===== 频率条件模块 =====
        # 将连续频率映射到高维嵌入
        self.freq_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ===== 空间特征提取 =====
        # 第一组卷积（大感受野）
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)  # 19×19 -> 9×9
        )
        
        # 第二组卷积（中等感受野）
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)  # 9×9 -> 4×4
        )
        
        # 第三组卷积（小感受野，细节特征）
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ===== 自适应池化 =====
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        # ===== 特征融合模块 =====
        # 空间特征维度: 512 * 2 * 2 = 2048
        # 频率特征维度: 512
        # 总维度: 2048 + 512 = 2560
        
        self.fusion_module = nn.Sequential(
            nn.Linear(2048 + 512, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(128, 8)  # 输出8个参数
        )
        # ===== 输出层 =====
        # 使用Tanh确保输出在[-1,1]范围内
        self.output_activation = nn.Tanh()
        
    def forward(self, matrix, frequency):
        """
        matrix: (batch, 1, 19, 19) 二进制矩阵
        frequency: (batch, 1) 归一化频率 [0,1]
        """
        # 1. 频率条件嵌入
        freq_feat = self.freq_embedding(frequency)  # (batch, 512)
        
        # 2. 空间特征提取
        x = self.conv_block1(matrix)  # (batch, 64, 9, 9)
        x = self.conv_block2(x)       # (batch, 128, 4, 4)
        x = self.conv_block3(x)       # (batch, 512, 4, 4)
        
        # 3. 池化
        x = self.adaptive_pool(x)     # (batch, 512, 2, 2)
        
        # 4. 展平空间特征
        spatial_feat = x.view(x.size(0), -1)  # (batch, 2048)
        
        # 5. 特征融合
        combined = torch.cat([spatial_feat, freq_feat], dim=1)  # (batch, 2560)
        
        # 6. 通过融合模块
        output = self.fusion_module(combined)
        
        # 7. 应用输出激活
        output = self.output_activation(output)
        
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