import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For a progress bar


def create_directory_if_not_exists(path):
    """Helper function to create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


def split_ddr_dataset(data_root_dir, test_split_ratio=0.2, random_state=42):
    """
    Splits the DDR dataset into training and validation sets.

    Args:
        data_root_dir (str): The root directory of the DDR dataset (e.g., './DDR/').
                             This directory must contain 'DDR_grading.csv' and
                             the 'DR_grading/DR_grading/' subdirectory with images.
        test_split_ratio (float): Proportion of the dataset to include in the test/validation split.
        random_state (int): Seed for random number generator for reproducible splits.
    """
    print(f"Starting dataset split process for directory: {data_root_dir}")
    print(f"Test split ratio: {test_split_ratio}, Random state: {random_state}\n")

    # --- 1. Define Paths ---
    original_csv_path = os.path.join(data_root_dir, "DR_grading.csv")
    original_images_dir = os.path.join(data_root_dir, "DR_grading", "DR_grading")

    train_output_dir = os.path.join(data_root_dir, "train")
    test_output_dir = os.path.join(
        data_root_dir, "test"
    )  # 'test' is used for validation as per user request

    train_images_output_dir = os.path.join(train_output_dir, "images")
    test_images_output_dir = os.path.join(test_output_dir, "images")

    train_csv_output_path = os.path.join(train_output_dir, "train_grading.csv")
    test_csv_output_path = os.path.join(test_output_dir, "test_grading.csv")

    # --- 2. Validate Original Data Paths ---
    if not os.path.exists(original_csv_path):
        print(f"Error: Original CSV file not found at {original_csv_path}")
        return
    if not os.path.isdir(original_images_dir):
        print(f"Error: Original images directory not found at {original_images_dir}")
        return

    print("Original data paths validated.")

    # --- 3. Create Output Directories ---
    print("\nCreating output directories...")
    create_directory_if_not_exists(train_output_dir)
    create_directory_if_not_exists(test_output_dir)
    create_directory_if_not_exists(train_images_output_dir)
    create_directory_if_not_exists(test_images_output_dir)

    # --- 4. Load and Split the CSV Data ---
    print("\nLoading and splitting CSV data...")
    try:
        df = pd.read_csv(original_csv_path, header=None, names=["image", "label"])
    except Exception as e:
        print(f"Error reading CSV file {original_csv_path}: {e}")
        return

    if df.empty:
        print("Error: The CSV file is empty.")
        return

    print(f"Total images found in CSV: {len(df)}")

    # Stratify by label to ensure similar class distribution in train and test sets
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_split_ratio,
            random_state=random_state,
            stratify=df["label"],  # Stratify by the 'label' column
        )
    except ValueError as e:
        print(
            f"Warning: Could not stratify data (e.g., some classes might have too few samples for stratification): {e}"
        )
        print("Proceeding with a non-stratified split.")
        train_df, test_df = train_test_split(
            df, test_size=test_split_ratio, random_state=random_state
        )

    print(f"Training samples: {len(train_df)}, Validation samples: {len(test_df)}")

    # --- 5. Copy Images and Save New CSVs ---
    def process_split(df_split, output_img_dir, output_csv_path, split_name):
        print(f"\nProcessing {split_name} set ({len(df_split)} images)...")
        if df_split.empty:
            print(f"No images to process for {split_name} set.")
            # Create an empty CSV if the split is empty
            pd.DataFrame(columns=["image", "label"]).to_csv(
                output_csv_path, header=False, index=False
            )
            print(f"Empty {split_name} CSV created at: {output_csv_path}")
            return

        # Copy images
        print(f"Copying {split_name} images to {output_img_dir}...")
        copied_count = 0
        not_found_count = 0
        for _, row in tqdm(
            df_split.iterrows(),
            total=len(df_split),
            desc=f"Copying {split_name} images",
        ):
            img_filename = row["image"]
            src_img_path = os.path.join(original_images_dir, img_filename)
            dest_img_path = os.path.join(output_img_dir, img_filename)

            if os.path.exists(src_img_path):
                try:
                    shutil.copy2(
                        src_img_path, dest_img_path
                    )  # copy2 preserves metadata
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {src_img_path} to {dest_img_path}: {e}")
            else:
                print(f"Warning: Source image not found: {src_img_path}. Skipping.")
                not_found_count += 1

        print(
            f"Finished copying {split_name} images. Copied: {copied_count}, Not found: {not_found_count}"
        )

        # Save the new CSV for the split
        # The CSV should contain only the filename, not the full path,
        # as it's relative to the new images directory.
        df_split.to_csv(output_csv_path, header=False, index=False)
        print(f"{split_name} CSV saved to: {output_csv_path}")

    # Process training set
    process_split(train_df, train_images_output_dir, train_csv_output_path, "training")

    # Process testing/validation set
    process_split(test_df, test_images_output_dir, test_csv_output_path, "validation")

    print("\nDataset splitting process completed.")
    print(
        f"Training data: {train_images_output_dir} (images), {train_csv_output_path} (labels)"
    )
    print(
        f"Validation data: {test_images_output_dir} (images), {test_csv_output_path} (labels)"
    )


if __name__ == "__main__":
    data_root_dir = "data/DDR"
    split_ddr_dataset(data_root_dir, test_split_ratio=0.2, random_state=44)