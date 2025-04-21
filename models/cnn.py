import torch.nn as nn

class CNN(nn.module):
    """
    input: 1*28*28
    """
    def __init(self, conv1_c, conv2_c, in_c=1, out_dim=10):
        super().__init()
        self.conv1 = nn.Conv2d(in_c, conv1_c,kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels=conv1_c, out_channels=conv2_c,kernel_size=3)
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

