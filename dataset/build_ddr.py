import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import cv2
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    Resize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    RandomAutocontrast,
    ColorJitter,
)


class DDRDataset(Dataset):
    """
    Custom Dataset class for the DDR dataset.
    """

    def __init__(self, csv_file_path, img_dir, transform=None):
        """
        Args:
            csv_file_path (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Read the csv file
        # The CSV file is expected to have two columns: image_filename and label
        try:
            self.img_labels = pd.read_csv(
                csv_file_path, header=None, names=["image", "label"]
            )
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file_path}")
            self.img_labels = pd.DataFrame(
                columns=["image", "label"]
            )  # Empty dataframe
        except Exception as e:
            print(f"Error reading CSV {csv_file_path}: {e}")
            self.img_labels = pd.DataFrame(columns=["image", "label"])

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where label is the DR grade.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image file name and label
        try:
            img_name = self.img_labels.iloc[idx, 0]
            label = int(self.img_labels.iloc[idx, 1])
        except IndexError:
            print(
                f"Error: Index {idx} out of bounds for img_labels with length {len(self.img_labels)}."
            )
            # This can happen if the CSV was not loaded correctly or is empty.
            return None, None  # Or raise an error

        # Construct the full image path
        img_path = os.path.join(self.img_dir, img_name)

        try:
            # Load the image
            image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            # You might want to return a placeholder or raise an error
            return None, None
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label


def build_DDR_dataset(data_root_dir, is_train=True, transform=None):
    """
    Builds and returns the DDR dataset for training or validation/testing.

    Args:
        data_root_dir (string): The root directory of the DDR dataset (e.g., './DDR/').
                                This directory should contain 'train/' and 'test/'
                                subdirectories if using the split data structure.
        is_train (bool): If True, loads the training dataset. Otherwise, loads
                         the validation/test dataset.
        transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
        DDRDataset: An instance of the DDRDataset.
    """
    if is_train:
        split_name = "train"
        csv_filename = "train_grading.csv"
    else:
        split_name = "test"  # Assuming 'test' directory for validation data
        csv_filename = "test_grading.csv"

    # Path to the CSV file for the specified split
    csv_path = os.path.join(data_root_dir, split_name, csv_filename)
    # Path to the images directory for the specified split
    images_base_path = os.path.join(data_root_dir, split_name, "images")

    image_size = 224
    t = []
    t.append(Resize((image_size, image_size), interpolation=Image.BICUBIC))
    # t.append(RandomCrop(image_size))
    t.append(RandomHorizontalFlip(p=0.5))
    t.append(RandomRotation(degrees=15))
    # t.append(ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0))
    t.append(RandomAutocontrast(p=0.99))

    if transform:
        t.append(transform)
    t.append(ToImage())
    t.append(ToDtype(torch.float32, scale=True))
    # Add contrast enhancement to transformation pipeline
    train_transform = Compose(t)

    t_test = []
    t_test.append(Resize((image_size, image_size), interpolation=Image.BICUBIC))
    t_test.append(RandomAutocontrast(p=0.99))
    t_test.append(ToImage())
    t_test.append(ToDtype(torch.float32, scale=True))

    test_transform = Compose(t_test)

    if not os.path.exists(csv_path):
        # Fallback for the original structure if train/test subdirs are not found (e.g. for initial testing)
        # This part is more for the self-contained example below.
        # In a real scenario, you'd expect the split directories to exist.
        print(f"Warning: Split-specific CSV not found at {csv_path}.")
        print(
            f"Attempting to load from original structure: {data_root_dir}/DDR_grading.csv"
        )
        csv_path = os.path.join(data_root_dir, "DDR_grading.csv")
        images_base_path = os.path.join(data_root_dir, "DR_grading", "DR_grading")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV file not found at {csv_path} (and fallback also failed). Please check the path."
            )

    if not os.path.isdir(images_base_path):
        raise NotADirectoryError(
            f"Image directory not found at {images_base_path}. Please check the path."
        )

    dataset = DDRDataset(
        csv_file_path=csv_path,
        img_dir=images_base_path,
        transform=train_transform if is_train else test_transform,
    )
    return dataset
