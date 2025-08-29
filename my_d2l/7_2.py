import torch
import torch.nn as nn


def vgg_block(num_convs, input_channels, output_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                padding=1,
                kernel_size=3,
            )
        )
        layers.append(nn.ReLU())
        input_channels = output_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg_structure(conv_arch):
    conv_struct = []
    in_channel = 1
    for conv, out_channel in conv_arch:
        conv_struct.append(vgg_block(conv, in_channel, out_channel))
        in_channel = out_channel

    return nn.Sequential(
        *conv_struct,
        nn.Flatten(),
        nn.Linear(out_channel * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 10),
    )


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

if __name__ == "__main__":
    X = torch.randn(size=(1, 1, 224, 224))
    net = vgg_structure(conv_arch=conv_arch)
    init_weights(net)
    print(X.shape)
    for blk in net:
        X = blk(X)
        print(X.shape)
