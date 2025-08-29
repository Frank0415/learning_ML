import torch
import numpy
import torch.nn as nn
import torchvision
from torchvision import transforms

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

batch_size = 256


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_and_evaluate(batch_size=256, lr=0.1, num_epochs=5):   
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    net.apply(init_weights)
    
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4)

    def accuracy(hat_y, real_y):
        if len(hat_y.shape) > 1:
            hat_y = hat_y.argmax(axis=1)
        return (hat_y == real_y).sum().item()

    for ep in range(num_epochs):
        net.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = net(x_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            metric = Accumulator(2)
            for x_batch, y_batch in train_loader:
                y_hat = net(x_batch)
                metric.add(accuracy(y_hat, y_batch), y_batch.numel())
            print(f"epoch {ep + 1}, accuracy {metric[0] / metric[1]:.4f}")
    with torch.no_grad():
        total_acc = 0
        total_num = 0
        for x_batch, y_batch in test_loader:
            y_hat = net(x_batch)
            total_acc += accuracy(y_hat, y_batch)
            total_num += y_batch.shape[0]
        print(f"test accuracy {total_acc / total_num:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
