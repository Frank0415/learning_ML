import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np


class ResMod(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        use1x1conv=False,
        stride=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )
        if use1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.conv3 = None

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, X):
        Y = self.relu(self.bn(self.conv1(X)))
        Y = self.bn(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)

        Y += X

        return self.relu(Y)


def resnet_block(input_channels, output_channels, num_residual=2, first_block=False):
    blk = []
    for i in range(num_residual):
        if i == 0 and not first_block:
            blk.append(
                ResMod(input_channels, output_channels, use1x1conv=True, stride=2)
            )
        else:
            blk.append(ResMod(output_channels, output_channels))

    return blk


b1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

b2 = nn.Sequential(
    *resnet_block(64, 64, 2, True),
    *resnet_block(
        64,
        128,
    ),
    *resnet_block(128, 256),
    *resnet_block(256, 512),
)

net = nn.Sequential(
    b1,
    b2,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(in_features=512, out_features=10),
)

x = torch.randn(size=(1, 1, 96, 96))

for layer in net:
    x = layer(x)
    print(layer.__class__.__name__, "output shape:\t", x.shape)
