import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel
from torchvision import datasets
from torchvision.transforms.v2 import (
    Compose,
    RandomRotation,
    RandomAffine,
    ToImage,
    ToDtype,
)
import matplotlib.pyplot as plt
import numpy as np

try:
    from fvcore.nn import FlopCountAnalysis 
    flopanalysis = True
except ImportError:
    flopanalysis = False
import multiprocessing

from models.cnn import CNN
from models.balance_data import PartOfData
import os
import shutil
from sklearn.metrics import recall_score, precision_score, f1_score


def train_loop(dataloader, base_model, model_ema, loss_fn, optimizer, scheduler, epoch,l1_lambda):
    base_model.train() # Set base model to train mode
    # model_ema usually stays in eval mode or its state is managed internally by update_parameters

    for batch, (X, y) in enumerate(dataloader):
        # Current step count (global)
        step = epoch * len(dataloader) + batch
        X, y = X.to(device), y.to(device)

        # Use base_model for prediction and loss calculation for backprop
        pred = base_model(X)
        loss = loss_fn(pred, y)
        l1_norm = sum(p.abs().sum() for p in base_model.parameters())
        loss = loss + l1_lambda * l1_norm
        
        optimizer.zero_grad() # Zero gradients before backward pass
        loss.backward()
        optimizer.step() # Update base_model parameters

        # Update EMA model parameters based on the base_model
        model_ema.update_parameters(base_model)

        # Record loss and learning rate at each step
        current_lr = scheduler.get_last_lr()[0]
        if batch % 100 == 0:
            # Record loss from base_model's computation
            train_losses.append(loss.item())
            learning_rates.append(current_lr)
            step_count.append(step)

    scheduler.step()



def test_loop(dataloader, model_ema, loss_fn, log_wrong_type=False):
    model_ema.eval() # Ensure EMA model is in eval mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    # 用于存储所有预测和真实标签
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model_ema(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # 收集预测和真实标签
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if log_wrong_type:
                wrong_mask = (pred.argmax(1) != y)
                misclassified_samples = []
                if wrong_mask.any():
                    misclassified_samples.extend(
                        list(zip(X[wrong_mask].cpu(), y[wrong_mask].cpu(), pred.argmax(1)[wrong_mask].cpu()))
                    )

    test_loss /= num_batches
    correct /= size

    # 计算各项指标
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 记录测试指标
    test_losses.append(test_loss)
    test_accuracies.append(100 * correct)
    test_recalls.append(100 * recall)
    test_precisions.append(100 * precision)
    test_f1_scores.append(100 * f1)



    # 保存错误分类样本的代码保持不变
    if log_wrong_type and misclassified_samples:
        if os.path.exists('wrong_results'):
            shutil.rmtree('wrong_results')
        os.makedirs('wrong_results', exist_ok=True)
        
        with open('misclassified_samples.txt', 'w') as f:
            f.write("真实标签,预测标签\n")
            for sample in misclassified_samples:
                true_label = sample[1].item()
                pred_label = sample[2].item()
                f.write(f"{true_label},{pred_label}\n")
        
        for i, sample in enumerate(misclassified_samples):
            image = sample[0].squeeze().numpy()
            plt.imsave(f'wrong_results/misclassified_{i}_true{sample[1].item()}_pred{sample[2].item()}.png', image, cmap='gray')
def plot_metrics(
    train_losses, learning_rates, step_count, test_losses, test_accuracies, test_recalls,test_precisions, test_f1_scores,title
):
    """Plot training metrics"""
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # 展开子图数组
    ax1, ax2, ax3, ax4, ax5, ax6 = axs.ravel()

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
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






if __name__=='__main__':
    for loop in range(1):
        '''
        建立模型、数据的列表，每次试验选择不同组合。

        '''
        
        set_seed(114514) # Set random seed for reproducibility
        num_workers = multiprocessing.cpu_count()  # on my laptop, this is 20
        print(f"Number of workers: {num_workers}")


        # in task 3.3 there is no need to enhence the data
        # for minist, i don't think cutmix is necessary, so i only use geometric augmentations
        # train_transform = Compose(
        #     [
        #         RandomRotation(degrees=10),  # randomly rotate the image by 10 degrees
        #         RandomAffine(
        #             degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)
        #         ),  # randomly translate and scale the image
        #         ToImage(),
        #         ToDtype(torch.float32, scale=True),
        #     ]
        # )

        # test_transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])

        # training_data = datasets.MNIST(
        #     root="data",
        #     train=True,
        #     download=False,
        #     transform=train_transform if task=="3.3" else None,
        # )

        # test_data = datasets.MNIST(
        #     root="data",
        #     train=False,
        #     download=False,
        #     transform=test_transform if task=="3.3" else None,
        # )

        # hyper parameter
        batch_size = 100
        learning_rate = 1e-1
        epochs = 5
        L2_parameter = 1e-6
        momentum = 0.9

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {device}")

        # Lists to record metrics
        train_losses = []
        learning_rates = []
        step_count = []
        test_losses = []
        test_accuracies = []
        test_recalls = []
        test_f1_scores = []
        test_precisions = []

        data = PartOfData() # Initialize the data class, and define the number of training and test samples
        training_data = data.get_training_data()
        test_data = data.get_testing_data()
        
        # seed?  
        train_dataloader = DataLoader(
            training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataloader = DataLoader(
            test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        # 1. Instantiate the base model and move to device
        base_model = CNN(in_c=1, conv1_c=20, conv2_c=15, out_dim=10).to(device)

        # Print model summary using the base model
        total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"Number of parameters: {total_params}")
        if flopanalysis:
            # Pass base_model to FlopCountAnalysis
            flops_analysis = FlopCountAnalysis(base_model, torch.randn(8, 1, 28, 28).to(device))
            print(f"FLOPs: {flops_analysis.total() / 1e6} M")

        # 2. Instantiate the AveragedModel using the base model
        model_ema = AveragedModel(
            base_model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(L2_parameter)
        ).to(device) # EMA model also needs to be on device


        # Pass both base_model and model_ema

        L1_parameter = 0


        loss_fn = nn.CrossEntropyLoss()
        # 3. Optimizer targets the base_model's parameters
        optimizer = torch.optim.SGD(
            base_model.parameters(), lr=learning_rate, weight_decay=L2_parameter, momentum=momentum
        )
        steps = len(train_dataloader)
        scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

        for t in range(epochs):
            print(f"Epoch {t + 1}/{epochs}")
            # Pass both models to train_loop
            train_loop(train_dataloader, base_model, model_ema, loss_fn, optimizer, scheduler, t, L1_parameter)
            # Pass the EMA model to test_loop
            test_loop(test_dataloader, model_ema, loss_fn,t==epochs-1) # Log wrong type only on the last epoch

        # 4. Save the EMA model state dict
        torch.save(model_ema.state_dict(), "model.pth")
        print("Saved EMA model state to model_ema.pth")

        # Plot metrics after training
        max_acc = max(test_accuracies) if test_accuracies else 0 # Handle empty list
        print(f"Max EMA accuracy: {max_acc:>0.1f}%")
        if test_accuracies:
            print(f"Final EMA accuracy: {test_accuracies[-1]:>0.1f}%, Final EMA test loss: {test_losses[-1]:>8f} \n")
        else:
            print("No test results recorded.")

        # Call plot_metrics if lists are not empty
        if train_losses and learning_rates and step_count and test_losses and test_accuracies:
            plot_metrics(train_losses, learning_rates, step_count, test_losses, test_accuracies,test_recalls,test_precisions,test_f1_scores, title="MNIST Classification Training Results")#可以在此处声明超参数、数据集、模型等
        else:
            print("Metrics lists are empty, skipping plotting.")