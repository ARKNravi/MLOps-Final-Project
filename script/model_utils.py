import torch
import torch.nn as nn
from torchvision import models

def initialize_model(num_classes):
    """Initialize the ResNet18 model"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model 