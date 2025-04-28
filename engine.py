import torch
from sklearn.metrics import recall_score, precision_score, f1_score


def train_loop(
    dataloader, base_model, model_ema, loss_fn, optimizer, scheduler, epoch, l1_lambda, device
):
    base_model.train()  # Set base model to train mode
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

        optimizer.zero_grad()  # Zero gradients before backward pass
        loss.backward()
        optimizer.step()  # Update base_model parameters

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


def test_loop(dataloader, model_ema, loss_fn, device,log_wrong_type=False):
    model_ema.eval()  # Ensure EMA model is in eval mode
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
    correct /= size

    # 计算各项指标
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # 记录测试指标
    test_losses.append(test_loss)
    test_accuracies.append(100 * correct)
    test_recalls.append(100 * recall)
    test_precisions.append(100 * precision)
    test_f1_scores.append(100 * f1)

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