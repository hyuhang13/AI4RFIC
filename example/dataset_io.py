# utils/dataset_io.py
import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path

class DatasetSaver:
    """保存和加载预处理后的数据集"""
    
    @staticmethod
    def save_dataset(data_loader_dict, dataset_dict, save_dir='saved_datasets'):
        """
        保存数据集和DataLoader配置
        
        Args:
            data_loader_dict: 包含train_loader, val_loader, test_loader的字典
            dataset_dict: 包含train_dataset, val_dataset, test_dataset的字典
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        save_info = {
            'data_loaders': {},
            'datasets': {},
            'metadata': {
                'timestamp': str(np.datetime64('now')),
                'dataset_sizes': {
                    'train': len(dataset_dict['train_dataset']),
                    'val': len(dataset_dict['val_dataset']),
                    'test': len(dataset_dict['test_dataset'])
                }
            }
        }
        
        # 1. 保存数据集对象
        for name, dataset in dataset_dict.items():
            dataset_path = os.path.join(save_dir, f'{name}.pkl')
            
            # 提取数据集的原始数据
            dataset_data = {
                'matrices': dataset.matrices,
                'frequencies': dataset.frequencies,
                's_params': dataset.s_params,
                'indices': getattr(dataset, 'indices', None)
            }
            
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset_data, f)
            
            save_info['datasets'][name] = dataset_path
        
        # 2. 保存DataLoader配置
        for name, loader in data_loader_dict.items():
            loader_config = {
                'batch_size': loader.batch_size,
                'shuffle': loader.sampler is not None and hasattr(loader.sampler, 'shuffle'),
                'num_workers': loader.num_workers,
                'pin_memory': loader.pin_memory,
                'drop_last': loader.drop_last
            }
            
            save_info['data_loaders'][name] = loader_config
        
        # 3. 保存元数据
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(save_info['metadata'], f, indent=2)
        
        # 4. 保存整体配置
        config_path = os.path.join(save_dir, 'dataset_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(save_info, f)
        
        print(f"数据集已保存到: {save_dir}")
        print(f"训练集大小: {save_info['metadata']['dataset_sizes']['train']}")
        print(f"验证集大小: {save_info['metadata']['dataset_sizes']['val']}")
        print(f"测试集大小: {save_info['metadata']['dataset_sizes']['test']}")
        
        return save_dir
    
    @staticmethod
    def load_dataset(load_dir='saved_datasets', dataset_class=None):
        """
        加载保存的数据集
        
        Args:
            load_dir: 加载目录
            dataset_class: 数据集类（需要与保存时相同）
        
        Returns:
            data_loaders, datasets, metadata
        """
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"数据集目录不存在: {load_dir}")
        
        # 1. 加载配置
        config_path = os.path.join(load_dir, 'dataset_config.pkl')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'rb') as f:
            save_info = pickle.load(f)
        print(save_info)
        # 2. 加载数据集
        datasets = {}
        i = 0
        for name, path in save_info['datasets'].items():
            if(i < 5):
                print(name)
                print(path)
                i+=1
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    dataset_data = pickle.load(f)
                
                # 使用自定义数据集类重建
                if dataset_class:
                    dataset = dataset_class(
                        matrices=dataset_data['matrices'],
                        frequencies=dataset_data['frequencies'],
                        s_params=dataset_data['s_params']
                    )
                    if dataset_data['indices'] is not None:
                        dataset.indices = dataset_data['indices']
                else:
                    dataset = dataset_data
                
                datasets[name] = dataset
            else:
                print(f"警告: {path} 不存在")
        # print("datasets的所有键名：", datasets.keys())
        # 3. 重建DataLoader
        data_loaders = {}
        for loader_name, config in save_info['data_loaders'].items():
            # 根据loader_name找到对应的dataset_name
            # 假设loader_name是'train_loader'，对应的dataset_name是'train_dataset'
            dataset_name = loader_name.replace('_loader', '_dataset')
            
            if dataset_name in datasets:
                data_loaders[loader_name] = DataLoader(
                    datasets[dataset_name],
                    batch_size=config['batch_size'],
                    shuffle=config['shuffle'],
                    num_workers=config['num_workers'],
                    pin_memory=config['pin_memory'],
                    drop_last=config['drop_last']
                )
            else:
                print(f"警告: 找不到对应的数据集 {dataset_name} 来创建 {loader_name}")
    
        # print("data_loaders的所有键名：", data_loaders.keys())
        # 4. 加载元数据
        metadata_path = os.path.join(load_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = save_info['metadata']
        
        print(f"数据集已从 {load_dir} 加载")
        print(f"训练集大小: {metadata['dataset_sizes']['train']}")
        print(f"验证集大小: {metadata['dataset_sizes']['val']}")
        print(f"测试集大小: {metadata['dataset_sizes']['test']}")
        print(f"加载的data_loaders键: {list(data_loaders.keys())}")
        print(f"加载的datasets键: {list(datasets.keys())}")
        return data_loaders, datasets, metadata