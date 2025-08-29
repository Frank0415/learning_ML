import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np

nn.BatchNorm2d()

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.BatchNorm2d(6),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.BatchNorm2d(120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.BatchNorm2d(84),
    nn.ReLU(),
    nn.Linear(84, 10),21
)
