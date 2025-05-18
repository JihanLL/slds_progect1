import torch.nn as nn

class MLP(nn.Module):
    """
    A simple MLP model for 1x28x28 input.
    """

    def __init__(
        self, in_c=1, out_dim=10, hidden1=128, hidden2=128, input_size=28
    ):  # Corrected: __init__ method name
        super().__init__()  # Corrected: Call super().__init__()

        self.flatten = nn.Flatten()
        self.blk = nn.Sequential(
            nn.Linear(in_c * input_size * input_size, hidden1),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden1, hidden2),
            nn.Dropout(p=0.35),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.35),  # 在全连接层前添加Dropout
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.blk(x)
        x = self.head(x)
        return x
