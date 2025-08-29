import torch.nn as nn
import torch
import numpy as np
import torch.utils.data as data

import matplotlib.pyplot as plt

# Simulate NASDAQ and Dow Jones trends
np.random.seed(42)
x = np.arange(1000)
a = 0.8 * x + 500 * np.sin(0.05 * x) + np.random.normal(0, 200, 1000)
b = 0.4 * x + 300 * np.sin(0.1 * x + 0.2) + np.random.normal(0, 150, 1000)


def plot_indices(vec1, vec2):
    plt.figure(figsize=(12, 6))
    plt.plot(vec1, color="blue", label="real")
    plt.plot(vec2, color="orange", label="Train")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Index Value")
    plt.title("NASDAQ vs Dow Jones Trends")
    plt.show()


class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, window_size=4):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size].unsqueeze(0)
        return X, y


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    init_weights(net)
    return net


loss = nn.MSELoss(reduction="none")


def train(net, iter, loss, epoch=30, lr=0.0005):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for i in range(epoch):
        for X, y in iter:
            trainer.zero_grad()
            loss_diff = loss(net(X), y)
            loss_diff.sum().backward()
            trainer.step()
        print(f"epoch {i + 1}, loss {loss_diff.mean().item():f}")
    return net


def test(net, iter, loss):
    for X, y in iter:
        loss_diff = loss(net(X), y)
        print(loss_diff.sum().mean().item())
    plot_indices(net(X).detach().numpy(), y.numpy())


if __name__ == "__main__":
    net = get_net()
    dataset = TimeSeriesDataset(a[:600])
    train_iter = data.DataLoader(dataset, batch_size=32, shuffle=True)
    train(net, train_iter, loss)
    test_dataset = TimeSeriesDataset(a[600:])
    test_iter = data.DataLoader(test_dataset, batch_size=400, shuffle=False)
    test(net, test_iter, loss)
