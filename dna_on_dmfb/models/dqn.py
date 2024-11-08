import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 卷积层1：输入通道10，输出通道32
        self.conv1 = nn.Conv2d(
            in_channels=10, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积层2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 卷积层3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 卷积层4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 卷积层5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 5)  # 输出5个动作的Q值

    def forward(self, x):
        # 卷积层和激活函数
        x = F.relu(self.bn1(self.conv1(x)))  # 输出尺寸：32 x 15 x 15
        x = F.relu(self.bn2(self.conv2(x)))  # 输出尺寸：64 x 15 x 15
        x = F.relu(self.bn3(self.conv3(x)))  # 输出尺寸：128 x 15 x 15
        x = F.relu(self.bn4(self.conv4(x)))  # 输出尺寸：256 x 15 x 15
        x = F.relu(self.bn5(self.conv5(x)))  # 输出尺寸：256 x 15 x 15

        # 池化层
        x = self.pool(x)  # 输出尺寸：256 x 7 x 7

        # 展平
        x = x.view(-1, 256 * 7 * 7)

        # 全连接层和激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 输出层
        q_values = self.out(x)
        return q_values


if __name__ == "__main__":
    net = DQN()
    print(net)
    x = torch.randn(8, 10, 15, 15)
    y = net(x)
    print(y.shape)
