import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_loop(
    dataloader,
    model,
    loss_fn=None,
    optimizer=None,
    scheduler=None,
    l1_lambda=0,
    device=None,
):
    model.train()  # Set base model to train mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    learning_rates = []

    preds = []
    labels = []
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)

        # Use base_model for prediction and loss calculation for backprop
        pred = model(X)
        loss = loss_fn(pred, y)
        preds.extend(pred.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        regularized_loss = loss + l1_lambda * l1_norm

        optimizer.zero_grad()  # Zero gradients before backward pass
        regularized_loss.backward()
        optimizer.step()  # Update base_model parameters

        # Record metrics for the batch
        total_loss += loss.item()  # Use original loss for reporting
        total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_samples += y.size(0)

        # Record learning rate at each step if scheduler updates per step
        # If scheduler updates per epoch, this will be the same value throughout
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    # Return the average learning rate or the list of LRs used
    # Here we return the last one, assuming it might be most relevant if updated per epoch
    last_lr = learning_rates[-1] if learning_rates else None

    return avg_loss, accuracy, recall, precision, f1, last_lr


def test_loop(dataloader, model, loss_fn=None, device=None, log_wrong_type=False):
    model.eval()  # Ensure EMA model is in eval mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(dataloader)):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # 收集预测和真实标签
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if log_wrong_type:
                wrong_mask = pred.argmax(1) != y
                misclassified_samples = []
                if wrong_mask.any():
                    misclassified_samples.extend(
                        list(
                            zip(
                                X[wrong_mask].cpu(),
                                y[wrong_mask].cpu(),
                                pred.argmax(1)[wrong_mask].cpu(),
                            )
                        )
                    )

    test_loss /= num_batches
    accuracy = correct / size

    # 计算各项指标
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # 保存错误分类样本的代码保持不变
    if log_wrong_type and misclassified_samples:
        if os.path.exists("wrong_results"):
            shutil.rmtree("wrong_results")
        os.makedirs("wrong_results", exist_ok=True)

        with open("misclassified_samples.txt", "w") as f:
            f.write("真实标签,预测标签\n")
            for sample in misclassified_samples:
                true_label = sample[1].item()
                pred_label = sample[2].item()
                f.write(f"{true_label},{pred_label}\n")

        for i, sample in enumerate(misclassified_samples):
            image = sample[0].squeeze().numpy()
            plt.imsave(
                f"wrong_results/misclassified_{i}_true{sample[1].item()}_pred{sample[2].item()}.png",
                image,
                cmap="gray",
            )
    return test_loss, accuracy, recall, precision, f1


def plot_metrics(
    train_losses,
    learning_rates,
    step_count,
    training_accuracy,
    test_losses,
    test_accuracies,
    test_recalls,
    test_precisions,
    test_f1_scores,
    title=None,
):
    """Plot training metrics"""
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # 展开子图数组
    ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axs.ravel()

    # Plot training loss
    ax1.plot(step_count, train_losses)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss vs Steps")

    # Plot learning rate
    ax2.plot(step_count, learning_rates)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate vs Steps")

    # Plot test metrics
    epochs_x = np.arange(1, len(test_losses) + 1)
    ax3.plot(epochs_x, test_losses, "b-", label="Test Loss")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Test Loss")

    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs_x, test_accuracies, "r-", label="Accuracy")
    ax3_twin.set_ylabel("Accuracy (%)")

    ax4.plot(epochs_x, test_recalls, "g-", label="Recall")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Recall")

    ax5.plot(epochs_x, test_precisions, "m-", label="Precision")
    ax5.set_xlabel("Epochs")
    ax5.set_ylabel("Precision")

    ax6.plot(epochs_x, test_f1_scores, "c-", label="F1 Score")
    ax6.set_xlabel("Epochs")
    ax6.set_ylabel("F1 Score")

    ax7.plot(epochs_x, training_accuracy, "y-", label="Training Accuracy")
    ax7.set_xlabel("Epochs")
    ax7.set_ylabel("Training Accuracy")

    # Combine the legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # 添加图例到其他子图
    ax4.legend(loc="best")
    ax5.legend(loc="best")
    ax6.legend(loc="best")

    # 设置总标题
    fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()
