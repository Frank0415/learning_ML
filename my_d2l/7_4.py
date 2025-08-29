import torch
import torch.nn as nn
import torch.nn.functional as F


class inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(inception, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
        self.conv21 = nn.Conv2d(
            in_channels=in_channels, out_channels=c2[0], kernel_size=1
        )
        self.conv22 = nn.Conv2d(
            in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1, stride=1
        )
        self.conv31 = nn.Conv2d(
            in_channels=in_channels, out_channels=c3[0], kernel_size=1
        )
        self.conv32 = nn.Conv2d(
            in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2, stride=1
        )
        self.conv41 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.conv1(x))
        p2 = F.relu(self.conv22(F.relu(self.conv21(x))))
        p3 = F.relu(self.conv32(F.relu(self.conv31(x))))
        p4 = F.relu(self.conv42(self.conv41(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


if __name__ == "__main__":
    layers = []
    # b1
    layers.extend(
        [
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        ]
    )
    # b2
    layers.extend(
        [
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
    )
    # b3
    layers.extend(
        [
            inception(192, 64, (96, 128), (16, 32), 32),
            inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
    )
    # b4
    layers.extend(
        [
            inception(480, 192, (96, 208), (16, 48), 64),
            inception(512, 160, (112, 224), (24, 64), 64),
            inception(512, 128, (128, 256), (24, 64), 64),
            inception(512, 112, (144, 288), (32, 64), 64),
            inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
    )
    # b5
    layers.extend(
        [
            inception(832, 256, (160, 320), (32, 128), 128),
            inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10),
        ]
    )

    net = nn.Sequential(*layers)

    X = torch.randn(size=(1, 1, 96, 96))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, "output shape:\t", X.shape)
