import torch
from torchvision import datasets
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    Resize,
    Normalize,
    RandomAffine,
    RandomRotation,
)
import random

"""
    该类用于从 MNIST 数据集中随机选择指定数量的训练和测试样本。
    后续: 添加不同数据集的选项
    
"""


class PartOfData:
    train_transform = Compose(
        [
            RandomAffine(degrees=0, translate=(0.1, 0.1)),
            RandomRotation((-10, 10)),
            # Resize((48, 48)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = Compose(
        [
            # Resize((48, 48)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize((0.1307,), (0.3081,)),
        ]
    )
    _full_training_data = datasets.MNIST(
        root="data",
        train=True,
        download=False,
        transform=train_transform,  # Apply the transformation
    )

    _full_test_data = datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=test_transform,  # Apply the transformation
    )

    def __init__(
        self,
        full=True,
        num_training_samples=1000,
        num_test_samples=100,
        random_seed=321,
    ):  # 添加 random_seed 参数
        # 设置随机种子以保证可复现性（如果提供了）
        if full:
            self.training_data = self._full_training_data
            self.test_data = self._full_test_data
            return

        if random_seed is not None:
            random.seed(random_seed)

        # --- 获取训练数据的数量 ---
        num_total_train = len(self._full_training_data)
        if num_training_samples > num_total_train:
            print(
                f"Warning: Requested {num_training_samples} training samples, but only {num_total_train} available. Using all."
            )
            num_training_samples = num_total_train
        # 生成所有训练数据的索引
        all_train_indices = list(range(num_total_train))
        # 从所有索引中随机抽取指定数量的索引
        train_indices = random.sample(all_train_indices, num_training_samples)
        self.training_data = torch.utils.data.Subset(
            self._full_training_data, train_indices
        )
        print(f"Randomly selected {len(self.training_data)} samples for training.")

        # --- 获取测试数据的数量 ---
        num_total_test = len(self._full_test_data)
        if num_test_samples > num_total_test:
            print(
                f"Warning: Requested {num_test_samples} test samples, but only {num_total_test} available. Using all."
            )
            num_test_samples = num_total_test
        # 生成所有测试数据的索引
        all_test_indices = list(range(num_total_test))
        # 从所有索引中随机抽取指定数量的索引
        test_indices = random.sample(all_test_indices, num_test_samples)
        self.test_data = torch.utils.data.Subset(self._full_test_data, test_indices)
        print(f"Randomly selected {len(self.test_data)} samples for testing.")

    def get_training_data(self):
        return self.training_data

    def get_testing_data(self):
        return self.test_data
