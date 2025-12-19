# data_loader/dataset.py
import torch
from torch.utils.data import Dataset,DataLoader

# class SplitDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.FloatTensor(X)
#         self.y = torch.FloatTensor(y)
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
class DataLoaderCreator:
    """创建PyTorch DataLoader的类"""
    
    @staticmethod
    def create_tensor_dataset(matrices, frequencies, s_params):
        """创建PyTorch TensorDataset"""
        matrices_tensor = torch.FloatTensor(matrices)
        frequencies_tensor = torch.FloatTensor(frequencies)
        s_params_tensor = torch.FloatTensor(s_params)
        
        # 使用自定义Dataset类包装数据
        class SplitDataset(Dataset):
            def __init__(self, matrices, frequencies, s_params):
                self.matrices = matrices
                self.frequencies = frequencies
                self.s_params = s_params
            
            def __len__(self):
                return len(self.matrices)
            
            def __getitem__(self, idx):
                return {
                    'matrix': self.matrices[idx],
                    'frequency': self.frequencies[idx],
                    's_params': self.s_params[idx]
                }
        
        return SplitDataset(matrices_tensor, frequencies_tensor, s_params_tensor)
    
    @staticmethod
    def create_data_loaders(split_data_dict, batch_size=64, num_workers=4):
        """创建训练、验证、测试DataLoader"""
        
        # 创建数据集
        train_dataset = DataLoaderCreator.create_tensor_dataset(
            split_data_dict['train']['matrices'],
            split_data_dict['train']['frequencies'],
            split_data_dict['train']['s_params']
        )
        
        val_dataset = DataLoaderCreator.create_tensor_dataset(
            split_data_dict['val']['matrices'],
            split_data_dict['val']['frequencies'],
            split_data_dict['val']['s_params']
        )
        
        test_dataset = DataLoaderCreator.create_tensor_dataset(
            split_data_dict['test']['matrices'],
            split_data_dict['test']['frequencies'],
            split_data_dict['test']['s_params']
        )
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
