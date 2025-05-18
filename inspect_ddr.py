import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset.build_ddr import build_DDR_dataset

dataset = build_DDR_dataset(
    data_root_dir="data/DDR",
    is_train=True,
    transform=None,
)

for i in range(10):
    image, label = dataset[i]
    if image is not None:
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.savefig(f"analysis/ddr/sample_{i}_label_{label}.png")
    else:
        print(f"Sample {i} is None.")

print("Sample images saved in 'analysis' directory.")

