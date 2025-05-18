import torch
import torch.nn as nn
import torch.nn.functional as F


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
            # nn.Dropout2d(p=0.25),  # 添加Dropout
            # nn.MaxPool2d(kernel_size=2, stride=2), # Optional: Add pooling (e.g., 20x20 -> 10x10)
            # Add more Conv/BN/Act/Pool blocks here to increase depth
            # nn.Conv2d(
            #     in_channels=conv2_c, out_channels=10, kernel_size=3, padding=0
            # ),  # 22x22 -> 20x20 (or 11x11 -> 9x9 if pooling)
            # nn.BatchNorm2d(10),
            # nn.GELU(),
            # #nn.Dropout2d(p=0.25),
            # nn.Conv2d(
            #     in_channels=10, out_channels=8, kernel_size=3, padding=0
            # ),  # 22x22 -> 20x20 (or 11x11 -> 9x9 if pooling)
            # nn.BatchNorm2d(8),
            # nn.GELU(),
            # #nn.Dropout2d(p=0.25),
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
            nn.Dropout(p=0.35),  # 在全连接层前添加Dropout
            nn.Linear(self._flattened_size, out_dim),
        )

    def forward_features(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.flatten(x)  # Flatten the output of features
        x = self.head(x)
        return x


class CNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 9 * 9)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNv3(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()

        # Feature extractor part (convolutional layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),  # 'same' padding
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
            ),  # 'same' padding
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),  # 'valid' padding
            nn.Dropout(p=0.25),
            # Block 2
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
            ),  # 'same' padding
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
            ),  # 'same' padding
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),  # 'valid' padding
            nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),  # Flatten all dimensions except batch
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512),  # BatchNorm1d for dense layers
            nn.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class CNNv4(nn.Module):
    def __init__(
        self, block=CNNBlock, block_config=[2, 2, 2], base_channels=16, num_classes=10
    ):
        super().__init__()
        self.in_channels = base_channels

        self.conv1 = nn.Conv2d(1, base_channels, 3, 1, 1)  # 1 input channel for MNIST
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.build_layer(block, base_channels, block_config[0], stride=1)
        self.layer2 = self.build_layer(
            block, base_channels * 2, block_config[1], stride=2
        )
        self.layer3 = self.build_layer(
            block, base_channels * 4, block_config[2], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def build_layer(self, block, out_channels, blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
