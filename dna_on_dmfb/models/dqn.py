import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_channels=10, num_actions=5):
        super(DQN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.Sigmoid(),
        )
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean())
        return qvals


if __name__ == "__main__":
    net = DQN()
    print(net)
    x = torch.randn(8, 10, 15, 15)
    y = net(x)
    print(y.shape)
