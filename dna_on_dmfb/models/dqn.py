import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 卷积层序列
        self.conv_layers = nn.Sequential(
            # 卷积层1: kernel_size=1
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 卷积层2: kernel_size=1
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 卷积层3: kernel_size=1
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 池化层1
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积层4: kernel_size=3
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 卷积层5: kernel_size=3
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 卷积层6: kernel_size=3
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 卷积层7: kernel_size=3
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 池化层2
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层序列
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # 输出5个动作的Q值
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    net = DQN()
    print(net)
    x = torch.randn(8, 10, 15, 15)
    y = net(x)
    print(y.shape)
