import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

import numpy as np
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
# import mlflow

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    fvcore = None

from dataset.balance_data import PartOfData
from dataset.build_ddr import build_DDR_dataset
import argparse
from engine import (
    train_loop,
    test_loop,
    plot_metrics,
)
import os
import json
from datetime import datetime
import torch.backends.cudnn as cudnn

from models.cnn import CNN, CNNv2, CNNv3, CNNv4, CNNBlock
from models.mlp import MLP
from sklearn.neighbors import KNeighborsClassifier


def get_args_parser():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--seed", type=int, default=43, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="Dataset to use",
    )
    parser.add_argument(
        "--ddr_root_dir",
        type=str,
        default="",
        help="Root directory for DDR dataset",
    )
    parser.add_argument(
        "--full_dataset", action="store_true", help="Use full dataset or not"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--L1_parameter", type=float, default=1e-5, help="L1 regularization parameter"
    )
    parser.add_argument(
        "--L2_parameter", type=float, default=1e-5, help="L2 regularization parameter"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for the optimizer"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training and testing",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--milestones",
        type=list,
        default=[10, 20, 30],
        help="Milestones for learning rate scheduler",
    )
    parser.add_argument("--model", type=str, default="CNNv4", help="Model to use")
    parser.add_argument(
        "--flopanalysis", action="store_true", help="Enable FLOP analysis"
    )
    parser.add_argument(
        "--log_wrong_type", type=bool, default=False, help="Log wrong type predictions"
    )
    parser.add_argument(
        "--dist-eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        # default=-1, # Avoid default that conflicts with None check if not set by env
        help="Local rank for distributed training. Automatically set by torch.distributed.run.",
    )
    return parser


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Important for multi-GPU
    # Keep cudnn settings if you need strict reproducibility
    cudnn.deterministic = True
    cudnn.benchmark = False  # Setting benchmark to False can impact performance but aids reproducibility


def setup_ddp(local_rank_arg):
    # This function is called when distributed training is active.
    # local_rank_arg here MUST be an integer.
    if not isinstance(local_rank_arg, int):
        raise TypeError(
            f"local_rank_arg must be an int, got {type(local_rank_arg)} with value {local_rank_arg}"
        )

    torch.cuda.set_device(local_rank_arg)
    # init_method='env://' will look for MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE in environment variables
    # These are set by torch.distributed.run
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()  # Global rank
    print(f"DDP: Rank {rank}/{world_size} initialized. Using GPU: {local_rank_arg}.")
    return world_size, rank


def main(args):
    local_rank_from_env = os.environ.get("LOCAL_RANK")
    is_distributed = local_rank_from_env is not None

    world_size = 1
    global_rank = 0  # This is the global rank of the process
    current_gpu_index = 0  # This is the local rank (GPU index for the current process)

    if is_distributed:
        args.local_rank = int(
            local_rank_from_env
        )  # Store the integer local rank in args
        current_gpu_index = args.local_rank

        # Call setup_ddp, which handles torch.cuda.set_device and dist.init_process_group
        world_size, global_rank = setup_ddp(args.local_rank)
    else:
        # Not distributed or LOCAL_RANK env var not set
        print("Running in non-distributed mode.")
        # If user manually provides a local_rank (e.g. for single GPU selection without DDP)
        if args.local_rank is not None and args.local_rank >= 0:
            current_gpu_index = args.local_rank
        # Otherwise, current_gpu_index remains 0 for default device selection.

    if global_rank == 0 or not is_distributed:
        print(f"Script arguments: {args}")

    # Set seed after determining global_rank for process-specific seeding
    set_seed(args.seed + global_rank)

    # Determine device
    device = None
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{current_gpu_index}")
    elif (
        torch.backends.mps.is_available() and not is_distributed
    ):  # MPS typically for single-device non-DDP
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(
        f"Process Global Rank: {global_rank}, World Size: {world_size}. Using device: {device}"
    )

    # data
    if args.dataset == "MINIST":
        data = PartOfData(args.full_dataset)
        train_dataset = data.get_training_data()
        test_dataset = data.get_testing_data()
    elif args.dataset == "DDR":
        train_dataset = build_DDR_dataset(
            data_root_dir=args.ddr_root_dir,
            is_train=True,
            transform=None,
        )
        test_dataset = build_DDR_dataset(
            data_root_dir=args.ddr_root_dir,
            is_train=False,
            transform=None,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_sampler = None
    test_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True
        )
        if args.dist_eval:  # Check if distributed evaluation is enabled
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=world_size, rank=global_rank, shuffle=False
            )
        else:  # If not distributed eval, only rank 0 might evaluate or all evaluate on full data (less common)
            test_sampler = None  # Or handle as per your non-distributed eval strategy

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(
            train_sampler is None
        ),  # Shuffle only if no sampler (i.e., not distributed)
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,  # Pin memory if using CUDA
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(1.2 * args.batch_size),
        sampler=test_sampler,  # Will be None if not distributed eval or not distributed at all
        shuffle=False,  # Test dataloader is rarely shuffled
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize the model
    in_c = 1 if args.dataset == "MNIST" else 3
    n_classes = 10 if args.dataset == "MNIST" else 5
    if args.model == "CNNv1":
        model = CNN(in_c=in_c, conv1_c=20, conv2_c=15, out_dim=n_classes).to(device)
    elif args.model == "CNNv2":
        model = CNNv2(in_c=in_c, num_classes=n_classes).to(device)
    elif args.model == "CNNv3":
        model = CNNv3(input_channels=in_c, num_classes=n_classes).to(device)
    elif args.model == "CNNv4":
        block_config = [6, 8, 4]
        model = CNNv4(in_c=in_c, num_classes=n_classes, block_config=block_config).to(
            device
        )
    elif args.model == "MLP":
        model = MLP(in_c=in_c, out_dim=n_classes).to(device)
    elif args.model == "KNN":
        model = KNeighborsClassifier(n_neighbors=3)

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if is_distributed:
        # args.local_rank is the integer GPU index for this process
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.flopanalysis and (
        global_rank == 0 or not is_distributed and FlopCountAnalysis
    ):
        try:
            model_to_analyze = model.module if is_distributed else model
            dummy_input = torch.randn(
                args.batch_size if args.batch_size > 1 else 8, 1, 28, 28
            ).to(device)
            flops_analysis = FlopCountAnalysis(model_to_analyze, dummy_input)
            print(f"FLOPs: {flops_analysis.total() / 1e6:.2f} M")
        except Exception as e:
            print(f"FLOP analysis failed: {e}")

    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),  # DDP model's parameters are fine
            lr=args.learning_rate,
            weight_decay=args.L2_parameter,
            momentum=args.momentum,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.L2_parameter
        )
    # scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    scheduler = ExponentialLR(
        optimizer, gamma=0.9
    )  # Exponential decay for learning rate
    epochs = args.epochs
    L1_parameter = args.L1_parameter

    # --- Training loop ---
    (
        train_losses,
        train_recalls,
        train_precisions,
        train_f1_scores,
        training_accuracy,
        learning_rates,
    ) = [], [], [], [], [], []
    test_losses, test_accuracies, test_recalls, test_precisions, test_f1_scores = (
        [],
        [],
        [],
        [],
        [],
    )

    start_time = time.time()
    for t in range(epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(t)  # Important for shuffling in DDP

        epoch_log_prefix = f"Epoch {t + 1}/{epochs}"
        if is_distributed:
            epoch_log_prefix = f"[Rank {global_rank}] " + epoch_log_prefix
        # print(epoch_log_prefix)

        epoch_training_start_time = time.time()
        # Pass the DDP model (or regular model if not distributed)
        train_loss, train_accuracy, train_recall, train_precision, train_f1, last_lr = (
            train_loop(
                train_dataloader,
                model,  # This is DDP wrapped model if is_distributed
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,  # Scheduler step is usually inside train_loop or after epoch
                l1_lambda=L1_parameter,
                device=device,
                # Pass rank and world_size if train_loop needs to do DDP-aware operations (e.g. averaging metrics)
                rank=global_rank,
                world_size=world_size,
            )
        )
        scheduler.step()

        epoch_training_end_time = time.time()
        if global_rank == 0 or not is_distributed:  # Log time from rank 0
            print(
                f"{epoch_log_prefix} training time: {epoch_training_end_time - epoch_training_start_time:.2f} seconds"
            )

        # Aggregate metrics if train_loop doesn't handle DDP aggregation
        # For DDP, metrics from train_loop might be process-local. They need to be averaged.
        # If train_loop already averages across DDP processes, this is fine.
        # Otherwise, you'll need torch.distributed.all_reduce or similar here.
        # Assuming train_loop returns already averaged/rank-0 metrics for now.
        if global_rank == 0 or not is_distributed:
            train_losses.append(train_loss)
            learning_rates.append(last_lr)
            training_accuracy.append(train_accuracy)
            train_recalls.append(train_recall)
            train_precisions.append(train_precision)
            train_f1_scores.append(train_f1)

        # Evaluation:
        # For DDP, evaluation can be done on rank 0 only, or distributed evaluation.
        # Your test_sampler setup depends on args.dist_eval.
        if (
            global_rank == 0 or not is_distributed or args.dist_eval
        ):  # Perform test if rank 0, or non-DDP, or dist_eval
            evaluation_start_time = time.time()
            test_loss, test_accuracy, test_recall, test_precision, test_f1 = test_loop(
                test_dataloader,
                model.module
                if is_distributed
                else model,  # Pass unwrapped model for testing if preferred, or DDP model
                loss_fn,
                device=device,
                log_wrong_type=(
                    args.log_wrong_type and (t == epochs - 1)
                ),  # Log on last epoch
                # Pass rank and world_size if test_loop needs to do DDP-aware operations
                # rank=global_rank,
                # world_size=world_size
            )
            evaluation_end_time = time.time()
            if global_rank == 0 or not is_distributed:  # Log time from rank 0
                print(
                    f"{epoch_log_prefix} evaluation time: {evaluation_end_time - evaluation_start_time:.2f} seconds"
                )

            # Aggregate test metrics if test_loop doesn't handle DDP aggregation
            # Assuming test_loop returns already averaged/rank-0 metrics for now.
            if global_rank == 0 or not is_distributed:
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                test_recalls.append(test_recall)
                test_precisions.append(test_precision)
                test_f1_scores.append(test_f1)

                max_acc = max(test_accuracies) if test_accuracies else 0
                print(f"Max accuracy so far: {max_acc:>0.1f}%")
                print(
                    f"Epoch {t + 1} - Test Acc: {test_accuracy:>0.1f}%, Test Loss: {test_loss:>8f}"
                )

        # Checkpointing (only on rank 0 for DDP)
        if global_rank == 0 or not is_distributed:
            if (t + 1) % 10 == 0 or (
                t + 1
            ) == epochs:  # Save every 10 epochs or on the last epoch
                checkpoint_dir = "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"model_epoch{t + 1}.pth"
                )  # Simpler naming for DDP
                torch.save(
                    (
                        model.module if is_distributed else model
                    ).state_dict(),  # Save unwrapped model
                    checkpoint_path,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

        if is_distributed:
            dist.barrier()  # Ensure all processes sync before next epoch or finishing

    # End of training loop
    if global_rank == 0 or not is_distributed:
        end_time = time.time()
        total_train_time = end_time - start_time
        avg_epoch_time = total_train_time / epochs if epochs > 0 else 0
        print(
            f"Total training time: {total_train_time:.2f} seconds, "
            f"Average time per epoch: {avg_epoch_time:.2f} seconds"
        )

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join("results", dt_string)
        os.makedirs(results_dir, exist_ok=True)

        # Plot metrics (only on rank 0)
        if train_losses:  # Check if lists are populated
            plot_metrics(
                train_losses,
                train_recalls,
                train_precisions,
                train_f1_scores,
                training_accuracy,
                learning_rates,
                test_losses,
                test_accuracies,
                test_recalls,
                test_precisions,
                test_f1_scores,
                img_path=results_dir,
            )

        # Save final model (only on rank 0)
        final_model_path = os.path.join(results_dir, "model_final.pth")
        torch.save(
            (model.module if is_distributed else model).state_dict(), final_model_path
        )
        print(f"Saved final model to {final_model_path}")

        # Save metrics to a file (only on rank 0)
        metrics_data = {
            "train_losses": train_losses,
            "train_recalls": train_recalls,
            "train_precisions": train_precisions,
            "train_f1_scores": train_f1_scores,
            "training_accuracy": training_accuracy,
            "learning_rates": learning_rates,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
            "test_recalls": test_recalls,
            "test_precisions": test_precisions,
            "test_f1_scores": test_f1_scores,
            "args": vars(args),
            "total_training_time_seconds": total_train_time,
            "average_epoch_time_seconds": avg_epoch_time,
            "max_test_accuracy": max(test_accuracies) if test_accuracies else None,
            "final_test_accuracy": test_accuracies[-1] if test_accuracies else None,
            "final_test_loss": test_losses[-1] if test_losses else None,
        }
        metrics_file_path = os.path.join(results_dir, "metrics.json")
        try:
            with open(metrics_file_path, "w") as f:
                json.dump(metrics_data, f, indent=4)
            print(f"Saved metrics to {metrics_file_path}")
        except Exception as e:
            print(f"Error saving metrics: {e}")

    if is_distributed:
        dist.destroy_process_group()
        print(f"DDP: Rank {global_rank} process group destroyed.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
