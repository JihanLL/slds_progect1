import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from models.cnn import CNNBlock, CNNv4
from utils import count_parameters


# --- DDP Setup and Cleanup ---
def setup_ddp(rank, world_size):
    """Initializes the DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12375"  # Ensure this port is free
    # Initialize the process group
    # Use 'nccl' for NVIDIA GPUs, 'gloo' for CPUs or other GPUs
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)  # Crucial for DDP with GPUs


def cleanup_ddp():
    """Destroys the DDP process group."""
    dist.destroy_process_group()


# --- DDP-aware Training and Testing Epoch Functions ---
def train_epoch_ddp(model, device, train_loader, optimizer, criterion, epoch, rank):
    """Trains the model for one epoch with DDP."""
    model.train()
    train_loader.sampler.set_epoch(epoch)  # Important for shuffling with DDP

    local_running_loss = 0.0
    num_batches_local = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(
            output, target
        )  # Already averaged over batch by CrossEntropyLoss default
        loss.backward()
        optimizer.step()
        local_running_loss += loss.item()
        num_batches_local += 1

    avg_local_train_loss = (
        local_running_loss / num_batches_local if num_batches_local > 0 else 0.0
    )

    # Average this loss across all ranks
    loss_tensor = torch.tensor(avg_local_train_loss, device=device)
    dist.all_reduce(
        loss_tensor, op=dist.ReduceOp.AVG
    )  # Get the global average training loss

    return loss_tensor.item()


def test_epoch_ddp(model, device, test_loader, criterion, world_size, rank):
    """Evaluates the model on the test set with DDP."""
    model.eval()
    # test_loader.sampler.set_epoch(0) # Not strictly needed for test sampler if shuffle=False

    local_sum_test_loss = 0.0  # Sum of losses for this rank's data
    local_correct = 0
    local_total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)  # Mean loss for this batch

            local_sum_test_loss += loss.item() * data.size(
                0
            )  # Accumulate sum of losses
            pred = output.argmax(dim=1, keepdim=True)
            local_correct += pred.eq(target.view_as(pred)).sum().item()
            local_total_samples += data.size(0)

    # Tensors for all_reduce
    total_loss_tensor = torch.tensor(local_sum_test_loss, device=device)
    total_correct_tensor = torch.tensor(
        local_correct, dtype=torch.float32, device=device
    )
    total_samples_tensor = torch.tensor(
        local_total_samples, dtype=torch.float32, device=device
    )

    # Sum up across all processes
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    avg_test_loss = 0
    accuracy = 0
    # Rank 0 (or any rank, as they are synced) calculates final metrics
    if total_samples_tensor.item() > 0:
        avg_test_loss = (
            total_loss_tensor.item() / total_samples_tensor.item()
        )  # Global average loss per sample
        accuracy = (
            100.0 * total_correct_tensor.item() / total_samples_tensor.item()
        )  # Global accuracy

    return avg_test_loss, accuracy


# --- Main Worker Function for DDP ---
def main_worker(rank, world_size, config, managed_results_list):
    """Main function executed by each DDP process."""
    if rank == 0:
        print(f"DDP: Initializing process group for rank {rank}/{world_size}...")
    setup_ddp(rank, world_size)

    # Set seed for this process for reproducible data loading/augmentation
    # Model weights are synchronized by DDP automatically.
    torch.manual_seed(config["seed"])  # Keep same base seed for model init
    np.random.seed(config["seed"])

    # Determine device for this rank
    if (
        torch.cuda.is_available() and world_size > 0
    ):  # world_size > 0 for safety with cuda.device_count()
        device = rank  # Each process gets one GPU
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Rank {rank} using device: {device}")

    # Data Loading with DistributedSampler
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config["mnist_mean"], config["mnist_std"]),
        ]
    )

    # Rank 0 downloads dataset, others wait
    if rank == 0:
        print("Rank 0: Downloading MNIST dataset...")
        torchvision.datasets.MNIST(
            root="./data", train=True, download=False, transform=transform
        )
        torchvision.datasets.MNIST(
            root="./data", train=False, download=False, transform=transform
        )
        print("Rank 0: Download complete.")
    dist.barrier()  # All processes wait here until rank 0 finishes downloading

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform
    )

    # DistributedSampler ensures each process gets a different part of the data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config["seed"],
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )  # No shuffle for test

    # Effective batch size per GPU. Global batch size will be batch_size * world_size.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=config.get("num_data_workers", 24),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"] * 2,
        sampler=test_sampler,
        num_workers=config.get("num_data_workers", 24),
        pin_memory=True,
    )

    # Experiment Loop for different base_channels
    for base_ch in config["base_channels_configs"]:
        if rank == 0:
            print(f"\n--- Rank 0: Training model with base_channels = {base_ch} ---")

        # Instantiate model and move to device *before* DDP wrapping
        model = CNNv4(
            block=CNNBlock,
            block_config=config["block_config_fixed"],
            base_channels=base_ch,
            num_classes=config["num_classes"],
        ).to(device)

        # Wrap model with DDP
        if torch.cuda.is_available() and world_size > 0:
            model = DDP(
                model,
                device_ids=[device],
                output_device=device,
                find_unused_parameters=False,
            )
        else:  # CPU DDP
            model = DDP(model, find_unused_parameters=False)

        # Count parameters (from the original model, not the DDP wrapper)
        num_params = count_parameters(model.module)
        if rank == 0:
            print(f"Rank 0: Number of parameters: {num_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.CrossEntropyLoss().to(device)  # Loss function also on device

        epoch_train_losses = []
        epoch_test_losses = []
        epoch_test_accuracies = []

        start_time = time.time()
        for epoch in range(1, config["num_epochs"] + 1):
            train_loss = train_epoch_ddp(
                model, device, train_loader, optimizer, criterion, epoch, rank
            )
            test_loss, test_accuracy = test_epoch_ddp(
                model, device, test_loader, criterion, world_size, rank
            )

            if rank == 0:  # Only rank 0 logs and stores epoch-level results
                epoch_train_losses.append(train_loss)
                epoch_test_losses.append(test_loss)
                epoch_test_accuracies.append(test_accuracy)
                print(
                    f"Rank 0: Epoch {epoch}/{config['num_epochs']} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%"
                )

        end_time = time.time()
        if rank == 0:
            print(
                f"Rank 0: Training for base_channels={base_ch} took {end_time - start_time:.2f} seconds."
            )
            # Rank 0 appends the final results for this configuration to the shared list
            managed_results_list.append(
                {
                    "base_channels": base_ch,
                    "num_params": num_params,
                    "final_train_loss": epoch_train_losses[-1]
                    if epoch_train_losses
                    else float("nan"),
                    "final_test_loss": epoch_test_losses[-1]
                    if epoch_test_losses
                    else float("nan"),
                    "final_test_accuracy": epoch_test_accuracies[-1]
                    if epoch_test_accuracies
                    else float("nan"),
                }
            )
        if rank == 0:
            # Save the model's state dict (weights)
            save_path = f"checkpoints/double_descent/model_base_ch_{base_ch}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # For DDP models, we need to save the module not the DDP wrapper
            torch.save(model.module.state_dict(), save_path)
            print(f"Rank 0: Model with base_channels={base_ch} saved to {save_path}")

        dist.barrier()  # Ensure all ranks complete this configuration before rank 0 might proceed (e.g. to next config or plotting)

    cleanup_ddp()
    if rank == 0:
        print(f"Rank {rank} finished and cleaned up DDP.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    CONFIG = {
        "base_channels_configs": [
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            24,
            28,
            32,
            64,
            128,
            256,
            512
        ],
        "block_config_fixed": [2, 2, 2],
        "num_classes": 10,
        "num_epochs": 15,
        "batch_size": 128,
        "learning_rate": 0.001,
        "mnist_mean": (0.1307,),
        "mnist_std": (0.3081,),
        "seed": 42,
        "num_data_workers": 4,
    }

    # Determine world size (number of processes)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        world_size = torch.cuda.device_count()
        print(f"Main: Using {world_size} CUDA GPU(s) for DDP.")
    else:
        # Fallback to 1 CPU process if no GPUs, or use more for CPU DDP testing
        # world_size = mp.cpu_count() # For multi-process CPU DDP
        world_size = 1
        print(
            f"Main: No CUDA GPUs found or DDP disabled for GPUs. Using {world_size} CPU process(es)."
        )

    # Use a multiprocessing Manager list to collect results from rank 0 process
    manager = mp.Manager()
    managed_results_list = manager.list()

    if (
        world_size > 1
    ):  # or (world_size > 0 and not torch.cuda.is_available()): # For multi-process CPU DDP
        print(f"Main: Spawning {world_size} DDP processes...")
        mp.spawn(
            main_worker,
            args=(world_size, CONFIG, managed_results_list),
            nprocs=world_size,
            join=True,
        )
    elif world_size == 1:  # Single process (GPU or CPU)
        print(
            "Main: Running in single-process mode (no DDP spawn, but DDP logic will run with world_size=1)."
        )
        # Directly call main_worker for the single process case
        # This will still initialize a DDP group of size 1.
        main_worker(0, 1, CONFIG, managed_results_list)
    else:
        print(
            "Main: World size is 0, cannot run. Check CUDA availability or CPU count."
        )

    # Convert managed list to regular list for plotting (only in the main process after spawn completes)
    results = list(managed_results_list)

    # --- Plotting Results (only if results were collected) ---
    if not results:
        print("Main: No results collected. Plotting will be skipped.")
    else:
        print("Main: Plotting results...")
        results_sorted = sorted(results, key=lambda x: x["num_params"])

        param_counts = [r["num_params"] for r in results_sorted]
        train_losses = [r["final_train_loss"] for r in results_sorted]
        test_losses = [r["final_test_loss"] for r in results_sorted]
        test_accuracies = [r["final_test_accuracy"] for r in results_sorted]

        plt.figure(figsize=(18, 6))

        # Plot 1: Training and Test Loss vs. Number of Parameters
        plt.subplot(1, 3, 1)
        plt.plot(
            param_counts,
            train_losses,
            marker="o",
            linestyle="-",
            label="Final Training Loss",
        )
        plt.plot(
            param_counts,
            test_losses,
            marker="x",
            linestyle="-",
            label="Final Test Loss",
        )
        plt.xscale("log")
        plt.xlabel("Number of Parameters (log scale)")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss vs. Model Complexity (Parameters)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # Plot 2: Test Accuracy vs. Number of Parameters
        plt.subplot(1, 3, 2)
        plt.plot(
            param_counts,
            test_accuracies,
            marker="s",
            linestyle="-",
            color="green",
            label="Final Test Accuracy",
        )
        plt.xscale("log")
        plt.xlabel("Number of Parameters (log scale)")
        plt.ylabel("Test Accuracy (%)")
        plt.title("Test Accuracy vs. Model Complexity")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # Plot 3: Test Error Rate vs. Number of Parameters
        plt.subplot(1, 3, 3)
        # Filter out potential NaN values if any configuration failed or didn't produce accuracy
        valid_indices = [
            i
            for i, acc in enumerate(test_accuracies)
            if isinstance(acc, (int, float)) and not np.isnan(acc)
        ]
        if valid_indices:
            valid_param_counts = [param_counts[i] for i in valid_indices]
            test_error = [1 - (test_accuracies[i] / 100.0) for i in valid_indices]
            plt.plot(
                valid_param_counts,
                test_error,
                marker="d",
                linestyle="-",
                color="red",
                label="Final Test Error Rate",
            )
        else:
            plt.text(
                0.5,
                0.5,
                "No valid accuracy data for error rate plot",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        plt.xscale("log")
        plt.xlabel("Number of Parameters (log scale)")
        plt.ylabel("Test Error Rate (1 - Accuracy)")
        plt.title("Test Error Rate vs. Model Complexity (DDP)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        plt.suptitle(
            "Double Descent Phenomenon Exploration with CNNv4 on MNIST",
            fontsize=16,
            y=1.02,
        )

        plot_filename = "results/double_descent_ddp_plot.png"
        plt.savefig(plot_filename)
        print(f"Main: Plot saved as {plot_filename}")
        # plt.show() # plt.show() can be problematic in non-interactive DDP. Saving is safer.

        print("\nMain: Experiment Complete. Check the plots.")
        print("Main: Summary of results:")
        for r in results_sorted:
            print(
                f"  BaseChannels: {r['base_channels']}, Params: {r['num_params']:,}, TrainLoss: {r['final_train_loss']:.4f}, TestLoss: {r['final_test_loss']:.4f}, TestAcc: {r['final_test_accuracy']:.2f}%"
            )
