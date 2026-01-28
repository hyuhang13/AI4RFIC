# dataset_package.py
import torch
from torch.utils.data import Dataset
import numpy as np

class MultiModalDataset(Dataset):
    """多模态数据集（矩阵 + 频率 → S参数）"""
    
    def __init__(self, matrices, frequencies, s_params, indices=None):
        """
        初始化数据集
        
        Args:
            matrices: 矩阵数据 (n_samples, 1, height, width)
            frequencies: 频率数据 (n_samples, 1)
            s_params: S参数数据 (n_samples, 8)
            indices: 原始索引（可选）
        """
        self.matrices = matrices
        self.frequencies = frequencies
        self.s_params = s_params
        self.indices = indices
        
        # 转换为torch tensor
        if isinstance(self.matrices, np.ndarray):
            self.matrices = torch.FloatTensor(self.matrices)
        if isinstance(self.frequencies, np.ndarray):
            self.frequencies = torch.FloatTensor(self.frequencies)
        if isinstance(self.s_params, np.ndarray):
            self.s_params = torch.FloatTensor(self.s_params)
    
    def __len__(self):
        return len(self.matrices)
    
    def __getitem__(self, idx):
        return {
            'matrix': self.matrices[idx],
            'frequency': self.frequencies[idx],
            's_params': self.s_params[idx],
            'index': self.indices[idx] if self.indices is not None else idx
        }
    
    def get_sample_by_index(self, idx):
        """根据原始索引获取样本"""
        if self.indices is not None:
            if idx in self.indices:
                real_idx = np.where(self.indices == idx)[0][0]
                return self[real_idx]
        return None