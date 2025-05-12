import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel
import torch.backends.cudnn as cudnn

# import matplotlib.pyplot as plt # Uncomment if plot_metrics uses matplotlib directly
import numpy as np
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Assuming fvcore is installed: pip install fvcore
# If not, comment out the FlopCountAnalysis lines
try:
    from fvcore.nn import FlopCountAnalysis

    fvcore_available = True
except ImportError:
    fvcore_available = False
    print("fvcore not found. FLOP analysis will be skipped.")
    print("Install fvcore for FLOP analysis: pip install fvcore")


import multiprocessing  # Often not needed directly if using torch.distributed

# Make sure these imports point to the correct files/modules in your project structure
from models.cnn import CNN
from dataset.balance_data import PartOfData
import argparse
from engine import (
    train_loop,
    test_loop,
    plot_metrics,
)  # Ensure engine.py exists and has these functions
import os
import json
from datetime import datetime


def get_args_parser():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--full_dataset",
        type=bool,  # Consider using action='store_true' for boolean flags
        default=True,  # Default is True, use --full_dataset False to disable
        help="Use full dataset or not",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--L1_parameter",
        type=float,
        default=1e-5,
        help="L1 regularization parameter",
    )
    parser.add_argument(
        "--L2_parameter",
        type=float,
        default=1e-5,  # This is weight_decay for SGD/AdamW
        help="L2 regularization parameter (weight decay)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for the SGD optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training and testing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        # Default to 0 or a smaller number if multiprocessing issues arise
        default=min(
            8,
            multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() > 1 else 1,
        ),
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",  # Ensure PartOfData handles this string
        help="Dataset to use (e.g., MNIST)",
    )
    parser.add_argument(
        "--milestones",
        type=int,  # Changed type to int
        nargs="+",  # Allows multiple values: --milestones 20 28
        default=[20, 28],
        help="Milestones for learning rate scheduler (list of epoch indices)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CNN",  # Ensure this matches your model class name or logic
        help="Model architecture to use (e.g., CNN)",
    )
    parser.add_argument(
        "--flopanalysis",
        action="store_true",  # Use action='store_true' for flags
        help="Enable FLOP analysis (requires fvcore)",
    )
    parser.add_argument(
        "--log_wrong_type",
        action="store_true",  # Use action='store_true' for flags
        help="Log wrong type predictions during testing",
    )
    # Commented out distributed argument as local_rank is used for detection
    # parser.add_argument(
    #     "--distributed", # Name conflict with internal variable
    #     action="store_true",
    #     help="Use distributed training",
    # )
    parser.add_argument(
        "--dist-eval",  # Keep this if needed for specific evaluation logic
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (if different from training)",
    )
    # Arguments for torch.distributed.launch or torchrun
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,  # Default to -1, will be set by launcher
        help="number of distributed processes (set automatically by launcher)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,  # Default to -1 indicates non-distributed mode unless launched
        help="Local rank for distributed training (set automatically by launcher)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results (plots, metrics, checkpoints)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    return parser


def set_seed(seed):
    """Sets the random seed for reproducibility across libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Important for multi-GPU
        # Disabling benchmark and enabling deterministic mode can impact performance
        # Only use if exact reproducibility is critical
        cudnn.deterministic = True
        cudnn.benchmark = False
    print(f"Set seed to {seed}")


def setup_ddp(local_rank):
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)  # Pin the process to a specific GPU
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"Initialized DDP: Rank {rank}/{world_size} on GPU {local_rank}")
    return world_size, rank


def main(args):
    """Main training and evaluation function."""
    print("Starting training script...")
    print("Arguments:", args)

    # --- Distributed Setup ---
    # Check if launched in distributed mode (local_rank is set)
    distributed = args.local_rank != -1
    rank = 0  # Default rank for non-distributed or main process
    world_size = 1  # Default world size

    if distributed:
        try:
            world_size, rank = setup_ddp(args.local_rank)
        except Exception as e:
            print(f"Error setting up DDP: {e}. Running in non-distributed mode.")
            distributed = False
            args.local_rank = -1
    else:
        print("Running in non-distributed mode.")

    # --- Seed and Device ---
    set_seed(args.seed + rank)

    if torch.cuda.is_available():
        if distributed:
            device = torch.device(f"cuda:{args.local_rank}")
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Check for Apple Metal Performance Shaders
        device = torch.device("mps")
        print("Warning: MPS support might be experimental for some operations.")
    else:
        device = torch.device("cpu")
    print(f"Process Rank {rank}: Using device: {device}")

    # --- Data Loading ---
    print(f"Process Rank {rank}: Loading dataset '{args.dataset}'...")
    try:
        data = PartOfData(args.full_dataset)
        train_dataset = data.get_training_data()
        test_dataset = data.get_testing_data()
        print(
            f"Process Rank {rank}: Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}"
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return  # Exit if data loading fails

    # Create samplers for distributed training
    train_sampler = None
    test_sampler = None
    shuffle_train = True
    if distributed:
        # DistributedSampler ensures each process gets a different part of the data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        # Usually, evaluation doesn't need shuffling and can use the same sampler logic
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle_train = False  # Sampler handles shuffling

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,  # Shuffle only if not using DistributedSampler
        num_workers=args.num_workers,
        pin_memory=True,  # Improves data transfer speed to GPU
        drop_last=True if distributed else False,  # Drop last incomplete batch in DDP
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,  # Can often use a larger batch size for evaluation
        sampler=test_sampler,
        shuffle=False,  # No need to shuffle test data
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Process Rank {rank}: DataLoaders created.")

    # --- Model Initialization ---
    print(f"Process Rank {rank}: Initializing model '{args.model}'...")
    # Add logic here if you have multiple model types based on args.model
    if args.model.upper() == "CNN":
        # Assuming input channels=1 (like MNIST), and 10 output classes
        model = CNN(in_c=1, conv1_c=20, conv2_c=15, out_dim=10).to(device)
    else:
        print(f"Error: Model type '{args.model}' not recognized.")
        return

    # Wrap model with DDP if distributed
    model_without_ddp = model  # Keep a reference to the original model
    if distributed:
        # find_unused_parameters=True can sometimes be needed if not all model
        # parameters are used in the forward pass of a given iteration
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
        model_without_ddp = (
            model.module
        )  # Access original model attributes through .module
        print(f"Process Rank {rank}: Wrapped model with DDP.")

    # --- FLOP Analysis (Optional, on rank 0) ---
    if rank == 0 and args.flopanalysis and fvcore_available:
        try:
            # Use a sample input tensor matching the model's expected input
            # Ensure the sample tensor is on the correct device
            sample_input = torch.randn(1, 1, 28, 28).to(
                device
            )  # Example for MNIST-like input
            flops_analysis = FlopCountAnalysis(model_without_ddp, sample_input)
            total_flops = flops_analysis.total()
            print(f"FLOPs Analysis (Rank 0): {total_flops / 1e9:.2f} GFLOPs")
        except Exception as e:
            print(f"FLOP analysis failed: {e}")

    # --- EMA Model (Optional) ---
    # AveragedModel can improve generalization
    # Initialize EMA model - apply after DDP wrapping if using DDP
    # ema_decay = 0.999 # Example decay factor, adjust as needed
    # model_ema = AveragedModel(model, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged:
    #                            ema_decay * averaged_model_parameter + (1 - ema_decay) * model_parameter)
    # model_ema.to(device) # Ensure EMA model is on the correct device
    # print(f"Process Rank {rank}: EMA model initialized.")
    # Note: You'll need to update model_ema during training loop (e.g., model_ema.update())
    # and potentially use it for evaluation. The provided code doesn't use EMA yet.

    # --- Loss Function, Optimizer, Scheduler ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),  # Use DDP model's parameters if distributed
        lr=args.learning_rate,
        weight_decay=args.L2_parameter,  # L2 regularization
        momentum=args.momentum,
    )
    # Learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    print(f"Process Rank {rank}: Optimizer and Scheduler created.")

    # --- Training Loop ---
    print(f"Process Rank {rank}: Starting training for {args.epochs} epochs...")
    start_time = time.time()

    # History tracking (only really needed on rank 0 for aggregated results)
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_recall": [],
        "train_precision": [],
        "train_f1": [],
        "test_loss": [],
        "test_acc": [],
        "test_recall": [],
        "test_precision": [],
        "test_f1": [],
        "lr": [],
    }

    # --- Results Directory Setup (Rank 0 only) ---
    results_dir = None
    checkpoint_dir = None
    if rank == 0:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{args.model}_{args.dataset}_{dt_string}"
        results_dir = os.path.join(args.results_dir, run_name)
        checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Rank 0: Saving results to: {results_dir}")
        print(f"Rank 0: Saving checkpoints to: {checkpoint_dir}")
        # Save args to results dir
        args_file_path = os.path.join(results_dir, "args.json")
        try:
            with open(args_file_path, "w") as f:
                json.dump(vars(args), f, indent=4)
            print(f"Rank 0: Saved arguments to {args_file_path}")
        except Exception as e:
            print(f"Rank 0: Error saving arguments: {e}")

    for t in range(args.epochs):
        epoch_start_time = time.time()
        if distributed:
            # Ensure data shuffling is different each epoch with DistributedSampler
            train_sampler.set_epoch(t)
            print(f"Rank {rank} - Epoch {t + 1}/{args.epochs} - Set sampler epoch")

        # --- Training Phase ---
        epoch_train_start_time = time.time()
        # Ensure train_loop handles DDP correctly (e.g., gradient sync)
        # Pass the potentially DDP-wrapped model
        train_metrics = train_loop(
            train_dataloader,
            model,  # Pass the potentially DDP-wrapped model
            loss_fn=loss_fn,
            optimizer=optimizer,
            # scheduler=scheduler, # Step scheduler *after* optimizer step
            l1_lambda=args.L1_parameter,
            device=device,
            epoch=t,
            rank=rank,  # Pass rank for logging purposes if needed inside train_loop
            world_size=world_size,
            is_distributed=distributed,
        )
        epoch_train_end_time = time.time()

        # --- Step Scheduler ---
        # Important: Step the scheduler *after* the optimizer step within the epoch
        # Typically done once per epoch, but depends on the scheduler type
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]
        if rank == 0 and current_lr != new_lr:
            print(f"Rank 0 - Epoch {t + 1}: Learning rate changed to {new_lr:.6f}")

        # --- Evaluation Phase ---
        # Evaluation can be done on all ranks or just rank 0 after gathering results
        # Using distributed evaluation (all ranks compute, results gathered later)
        epoch_eval_start_time = time.time()
        test_metrics = test_loop(
            test_dataloader,
            model,  # Pass the potentially DDP-wrapped model
            loss_fn,
            device=device,
            log_wrong_type=args.log_wrong_type
            and (t == args.epochs - 1),  # Log only last epoch maybe?
            epoch=t,
            rank=rank,
            world_size=world_size,
            is_distributed=distributed,
            dist_eval=args.dist_eval,  # Pass flag if specific distributed eval logic is needed
        )
        epoch_eval_end_time = time.time()

        # --- Aggregate and Log Metrics (Rank 0) ---
        if rank == 0:
            # If train_loop/test_loop already return aggregated metrics (e.g., avg loss/acc), use them directly.
            # If they return metrics per rank, you need to gather them here using dist.gather or dist.all_reduce.
            # Assuming train_metrics and test_metrics are dictionaries containing aggregated results:
            history["train_loss"].append(train_metrics.get("loss", float("nan")))
            history["train_acc"].append(train_metrics.get("accuracy", float("nan")))
            history["train_recall"].append(train_metrics.get("recall", float("nan")))
            history["train_precision"].append(
                train_metrics.get("precision", float("nan"))
            )
            history["train_f1"].append(train_metrics.get("f1", float("nan")))
            history["lr"].append(current_lr)  # Log LR before scheduler step

            history["test_loss"].append(test_metrics.get("loss", float("nan")))
            history["test_acc"].append(test_metrics.get("accuracy", float("nan")))
            history["test_recall"].append(test_metrics.get("recall", float("nan")))
            history["test_precision"].append(
                test_metrics.get("precision", float("nan"))
            )
            history["test_f1"].append(test_metrics.get("f1", float("nan")))

            epoch_time = time.time() - epoch_start_time
            train_time = epoch_train_end_time - epoch_train_start_time
            eval_time = epoch_eval_end_time - epoch_eval_start_time

            print(f"--- Epoch {t + 1}/{args.epochs} Summary (Rank 0) ---")
            print(
                f"  Train Loss: {train_metrics.get('loss', 'N/A'):>8f} | Train Acc: {train_metrics.get('accuracy', 'N/A'):>0.2f}%"
            )
            print(
                f"  Test Loss:  {test_metrics.get('loss', 'N/A'):>8f} | Test Acc:  {test_metrics.get('accuracy', 'N/A'):>0.2f}%"
            )
            print(
                f"  Train Time: {train_time:.2f}s | Eval Time: {eval_time:.2f}s | Epoch Time: {epoch_time:.2f}s"
            )
            print("-" * (20 + len(str(args.epochs)) * 2 + 10))  # Separator

        # --- Checkpointing (Rank 0) ---
        # Save checkpoints periodically and/or at the end
        if rank == 0 and checkpoint_dir:
            # Example: Save every 10 epochs and at the last epoch
            if (t + 1) % 10 == 0 or (t + 1) == args.epochs:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"model_epoch_{t + 1}.pth"
                )
                save_obj = {
                    "epoch": t + 1,
                    # Save the state dict of the underlying model, not the DDP wrapper
                    "model_state_dict": model_without_ddp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": args,  # Optionally save args
                    # Add any other state you need to resume (e.g., EMA model state)
                }
                try:
                    torch.save(save_obj, checkpoint_path)
                    print(f"Rank 0: Saved checkpoint to {checkpoint_path}")
                except Exception as e:
                    print(f"Rank 0: Error saving checkpoint: {e}")

        # Synchronization barrier for distributed training
        if distributed:
            dist.barrier()  # Ensure all processes finish the epoch before starting the next

    # --- End of Training ---
    total_training_time = time.time() - start_time
    print(f"Process Rank {rank}: Finished training.")

    # --- Final Actions (Rank 0) ---
    if rank == 0:
        print("\n--- Training Complete (Rank 0) ---")
        print(f"Total training time: {total_training_time:.2f} seconds")
        avg_epoch_time = total_training_time / args.epochs if args.epochs > 0 else 0
        print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

        max_test_acc = max(history["test_acc"]) if history["test_acc"] else 0
        final_test_acc = history["test_acc"][-1] if history["test_acc"] else 0
        final_test_loss = (
            history["test_loss"][-1] if history["test_loss"] else float("nan")
        )

        print(f"Max Test Accuracy: {max_test_acc:>0.2f}%")
        print(f"Final Test Accuracy: {final_test_acc:>0.2f}%")
        print(f"Final Test Loss: {final_test_loss:>8f}")

        # --- Plotting ---
        if results_dir:
            try:
                print("Plotting metrics...")
                # Pass only the necessary lists from history to plot_metrics
                plot_metrics(
                    history["train_loss"],
                    history["train_recall"],
                    history["train_precision"],
                    history["train_f1"],
                    history["train_acc"],
                    history["lr"],
                    history["test_loss"],
                    history["test_acc"],
                    history["test_recall"],
                    history["test_precision"],
                    history["test_f1"],
                    img_path=results_dir,  # Pass the directory path
                )
                print(f"Saved plots to {results_dir}")
            except Exception as e:
                print(f"Error plotting metrics: {e}")
        else:
            print("Results directory not specified, skipping plotting.")

        # --- Save Final Model ---
        if checkpoint_dir:
            final_model_path = os.path.join(checkpoint_dir, "model_final.pth")
            try:
                # Save just the model state dict for inference/fine-tuning
                torch.save(model_without_ddp.state_dict(), final_model_path)
                print(f"Saved final model state dict to {final_model_path}")
            except Exception as e:
                print(f"Error saving final model: {e}")
        else:
            print("Checkpoint directory not specified, skipping final model save.")

        # --- Save Metrics ---
        if results_dir:
            metrics_file_path = os.path.join(results_dir, "metrics.json")
            metrics_data = {
                "history": history,
                "args": vars(args),
                "total_training_time_seconds": total_training_time,
                "average_epoch_time_seconds": avg_epoch_time,
                "max_test_accuracy": max_test_acc,
                "final_test_accuracy": final_test_acc,
                "final_test_loss": final_test_loss,
                # Add FLOPs if calculated
                "gflops": total_flops / 1e9 if "total_flops" in locals() else None,
            }
            try:
                with open(metrics_file_path, "w") as f:
                    # Use a custom encoder for non-serializable types if necessary
                    json.dump(
                        metrics_data,
                        f,
                        indent=4,
                        default=lambda o: "<not serializable>",
                    )
                print(f"Saved metrics to {metrics_file_path}")
            except Exception as e:
                print(f"Error saving metrics: {e}")
        else:
            print("Results directory not specified, skipping metrics save.")

    # --- Cleanup DDP ---
    if distributed:
        dist.destroy_process_group()
        print(f"Process Rank {rank}: Destroyed process group.")

    print(f"Process Rank {rank}: Script finished.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # --- Environment Variable Check for DDP ---
    # torchrun/torch.distributed.launch usually sets these.
    # If running manually, you might need to set them.
    if args.local_rank == -1 and ("LOCAL_RANK" in os.environ):
        print("Found LOCAL_RANK environment variable, setting args.local_rank.")
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # Basic check for CUDA availability if requested
    if args.local_rank != -1 and not torch.cuda.is_available():
        print(
            "Warning: Distributed training requested (local_rank != -1) but CUDA is not available. Exiting."
        )
        exit()  # Or fallback to CPU? Depends on intent.
    elif args.local_rank != -1 and torch.cuda.device_count() <= args.local_rank:
        print(
            f"Error: local_rank ({args.local_rank}) is invalid for available CUDA device count ({torch.cuda.device_count()}). Exiting."
        )
        exit()

    main(args)
