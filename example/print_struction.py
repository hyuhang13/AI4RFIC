import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os

from config import TRAIN_CONFIG, DATA_CONFIG, DEVICE
from data_preprocessor import DataPreprocessor
from data_loader import InductorDataset
from model import InductorNet,ResidualInductorNet 
from train import ModelManager
from optimization_algorithm import GeneticAlgorithm
from utils import TrainingVisualizer
def quick_shape_check():
    """¿ìËÙ¼ì²éÐÎ×´"""
    checkpoint_path = "checkpoints/inductor_checkpoints/checkpoint_epoch_0.pth"
    
    print("=== NOW model ===")
    
    # µ±Ç°Ä£ÐÍ
    model = InductorNet()
    print("µ±Ç°Ä£ÐÍ²ÎÊý:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # ±£´æµÄÄ£ÐÍ
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        saved_state = checkpoint['model_state_dict']
        print("\nMODEL has been saved:")
        for name, tensor in saved_state.items():
            print(f"  {name}: {tensor.shape}")
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        saved_state = loaded_checkpoint['model_state_dict']
        print("has been saved")
        
        for name, tensor in saved_state.items():
            print(f"  {name}: {tensor.shape}")

        current_state = model.state_dict()
        for key in current_state:
            if key in saved_state:
                status = "?" if current_state[key].shape == saved_state[key].shape else "?"
                print(f"  {status} {key}: µ±Ç°{current_state[key].shape} vs ±£´æ{saved_state[key].shape}")
            else:
                print(f"  ? {key}: µ±Ç°{current_state[key].shape} vs ±£´æ<È±Ê§>")
        
        for key in saved_state:
            if key not in current_state:
                print(f"  ? {key}: µ±Ç°<È±Ê§> vs ±£´æ{saved_state[key].shape}")
                
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    quick_shape_check()