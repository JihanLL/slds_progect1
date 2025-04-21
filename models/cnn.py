import torch.nn as nn


class CNN(nn.module):
    def __init(self, conv1_c, conv2_c, in_dim, out_dim, **kwargs):
        super().__init()
        self.conv1 = nn.Conv2d(in_dim, conv1_c)
        self.conv2 = nn.Conv2d(in_channels=conv1_c, out_channels=conv2_c)
        self.head = nn.Linear(out_dim, out_dim)
        self.bn = nn.BatchNorm2d()
        self.act = nn.GELU()

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.head(x)

