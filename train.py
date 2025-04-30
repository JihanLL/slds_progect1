import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    from fvcore.nn import FlopCountAnalysis
    flopanalysis = True
except ImportError:
    flopanalysis = False
import multiprocessing

from models.cnn import CNN
from dataset.balance_data import PartOfData
import argparse
from engine import train_loop, test_loop, plot_metrics
import os
import json
from datetime import datetime


def get_args_parser():
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
        default=2,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--full_dataset",
        type=bool,
        default=True,
        help="Use full dataset or not",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
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
        default=1e-5,
        help="L2 regularization parameter",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for the optimizer",
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
        default=8,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="Dataset to use (e.g., MNIST)",
    )
    parser.add_argument(
        "--milestones",
        type=list,
        default=[20, 28],
        help="Milestones for learning rate scheduler",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CNN",
        help="Model architecture to use (e.g., CNN)",
    )
    parser.add_argument(
        "--flopanalysis",
        action="store_true",
        help="Enable FLOP analysis",
    )
    parser.add_argument(
        "--log_wrong_type",
        action="store_true",
        help="Log wrong type predictions",
    )
    return parser


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def main(args):
    print("Starting training script...")
    print(args)
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Initialize the data class, and define
    # the number of training and test samples
    data = PartOfData(args.full_dataset)
    training_data = data.get_training_data()
    test_data = data.get_testing_data()
    # Initialize DataLoader
    num_workers = multiprocessing.cpu_count()
    train_dataloader = DataLoader(
        training_data, batch_size=100, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_data, batch_size=100, shuffle=True, num_workers=num_workers
    )
    # Initialize the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    model = CNN(in_c=1, conv1_c=20, conv2_c=15, out_dim=10).to(device)
    # Print model summary using the base model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    if flopanalysis:
        # Pass base_model to FlopCountAnalysis
        flops_analysis = FlopCountAnalysis(model, torch.randn(8, 1, 28, 28).to(device))
        print(f"FLOPs: {flops_analysis.total() / 1e6} M")
    # Initialize the AveragedModel using the base model
    model_ema = AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(1e-6),
    ).to(device)  # EMA model also needs to be on device
    # Pass both base_model and model_ema

    loss_fn = nn.CrossEntropyLoss()
    # 3. Optimizer targets the base_model's parameters
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.L2_parameter,
        momentum=args.momentum,
    )
    scheduler = MultiStepLR(optimizer, milestones=[20, 28], gamma=0.1)
    epochs = args.epochs
    L1_parameter = args.L1_parameter

    # --- Training loop ---
    train_losses = []
    train_recalls = []
    train_precisions = []
    train_f1_scores = []
    training_accuracy = []
    learning_rates = []
    training_accuracy = []
    test_losses = []
    test_accuracies = []
    test_recalls = []
    test_precisions = []
    test_f1_scores = []
    
    start_time = time.time()
    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}")
        # Pass both models to train_loop
        epoch_training_start_time = time.time()
        train_loss, train_accuracy, train_recall, train_precision, train_f1, last_lr = (
            train_loop(
                train_dataloader,
                model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                l1_lambda=L1_parameter,
                device=device,
            )
        )
        epoch_training_end_time = time.time()
        print(
            f"Epoch {t + 1} training time: {epoch_training_end_time - epoch_training_start_time:.2f} seconds"
        )
        train_losses.append(train_loss)
        learning_rates.append(last_lr)
        training_accuracy.append(train_accuracy)
        train_recalls.append(train_recall)
        train_precisions.append(train_precision)
        train_f1_scores.append(train_f1)

        evaluation_start_time = time.time()
        test_loss, test_accuracy, test_recall, test_precision, test_f1 = test_loop(
            test_dataloader,
            model,
            loss_fn,
            device=device,
            log_wrong_type=args.log_wrong_type,
        )  # Log wrong type only on the last epoch
        evaluation_end_time = time.time()
        print(
            f"Epoch {t + 1} evaluation time: {evaluation_end_time - evaluation_start_time:.2f} seconds"
        )
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_recalls.append(test_recall)
        test_precisions.append(test_precision)
        test_f1_scores.append(test_f1)

        # Save the EMA model state dict
        if t % 10 == 0:
            torch.save(model_ema.state_dict(), "checkpoints/model.pth")
            print("Saved model state to model.pth")

        # Plot metrics after training
        max_acc = max(test_accuracies) if test_accuracies else 0  # Handle empty list
        print(f"Max accuracy: {max_acc:>0.1f}%")
        if test_accuracies:
            print(
                f"top1 accuracy: {test_accuracies[-1]:>0.1f}%, val loss: {test_losses[-1]:>8f} \n"
            )
        else:
            print("No test results recorded.")

    end_time = time.time()
    print(
        f"Total training time: {end_time - start_time:.2f} seconds, "
        f"Average time per epoch: {(end_time - start_time) / epochs:.2f} seconds"
    )
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", dt_string)
    os.makedirs(results_dir, exist_ok=True)
    # Plot metrics
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
    # Save the model state dict
    torch.save(model.state_dict(), "model.pth")
    # Save metrics to a file

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
        "args": vars(args),  # Save arguments as well
        "total_training_time_seconds": end_time - start_time,
        "average_epoch_time_seconds": (end_time - start_time) / epochs,
        "max_test_accuracy": max_acc,
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


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
