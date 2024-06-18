import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import os
image_transforms = { 'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), #随机裁剪到256*256
        transforms.RandomRotation(degrees=15),#随机旋转
        transforms.RandomHorizontalFlip(p=0.5), #依概率水平旋转
        transforms.CenterCrop(size=224),#中心裁剪到224*224符合resnet的输入要求
        transforms.ToTensor(),#填充
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])#转化为tensor，并归一化至[0，-1]
    ]),
    'valid': transforms.Compose([transforms.Resize(size=256),transforms.CenterCrop(size=224),
        transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])}

