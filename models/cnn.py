import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    A simple CNN model for 1x28x28 input.
    """

    def __init__(
        self, in_c=1, conv1_c=20, conv2_c=15, out_dim=10, input_size=28
    ):  # Corrected: __init__ method name
        super().__init__()  # Corrected: Call super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=in_c, out_channels=conv1_c, kernel_size=7, padding=0
            ),  # 28x28 -> 22x22
            nn.BatchNorm2d(conv1_c),
            nn.GELU(),
            nn.Dropout2d(p=0.25),  # 添加Dropout，使用Dropout2d因为输入是2D的
            # nn.MaxPool2d(kernel_size=2, stride=2), # Optional: Add pooling (e.g., 22x22 -> 11x11)
            # Block 2
            nn.Conv2d(
                in_channels=conv1_c, out_channels=conv2_c, kernel_size=3, padding=0
            ),  # 22x22 -> 20x20 (or 11x11 -> 9x9 if pooling)
            nn.BatchNorm2d(conv2_c),
            nn.GELU(),
            nn.Dropout2d(p=0.25),  # 添加Dropout
            # nn.MaxPool2d(kernel_size=2, stride=2), # Optional: Add pooling (e.g., 20x20 -> 10x10)
            # Add more Conv/BN/Act/Pool blocks here to increase depth
        )

        # Calculate the flattened size dynamically
        # Create a dummy input with the expected size
        with torch.no_grad():  # No need to track gradients for this calculation
            dummy_input = torch.zeros(1, in_c, input_size, input_size)
            dummy_output = self.features(dummy_input)
            # calculate automatically the size of the output after the feature extraction and
            self._flattened_size = dummy_output.numel()  # Get total number of elements

        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),  # 在全连接层前添加Dropout
            nn.Linear(self._flattened_size, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)  # Flatten the output of features
        x = self.head(x)
        return x  # Added return statement
